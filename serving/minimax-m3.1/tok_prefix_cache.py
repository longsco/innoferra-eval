"""Prefix-aware prompt encoding for agentic traffic (innoferra 09-30).

Agentic sessions resend the whole conversation every turn; the engine re-tokenizes the full rendered prompt (80k tokens median on
real M3.1 traffic: ~150 ms, up to ~700 ms, all inside the request's TTFT) although almost all of it was tokenized for the previous
turn. This encoder splits the rendered prompt at special-token boundaries (added special tokens are cut out before
pre-tokenization, so text on either side tokenizes independently), finds the longest boundary prefix it has already encoded and only
tokenizes the tail. Entries are kept per process in an LRU; verification mode compares every result with a full encode.

Env: SGLANG_TOKENIZE_PREFIX_CACHE=1 (enable), SGLANG_TOKENIZE_PREFIX_CACHE_ENTRIES (LRU size, default 512),
     SGLANG_TOKENIZE_PREFIX_CACHE_VERIFY=1 (also run the full encode and return it, logging any mismatch)."""
import hashlib
import logging
import os
import re
from array import array
from collections import OrderedDict

logger = logging.getLogger(__name__)
ENABLED = os.environ.get("SGLANG_TOKENIZE_PREFIX_CACHE", "0") == "1"
VERIFY = os.environ.get("SGLANG_TOKENIZE_PREFIX_CACHE_VERIFY", "0") == "1"
MAX_ENTRIES = int(os.environ.get("SGLANG_TOKENIZE_PREFIX_CACHE_ENTRIES", "512"))


class PrefixCachedEncoder:
    def __init__(self, tokenizer, max_entries=MAX_ENTRIES):
        self.tok = tokenizer
        specials = {}
        for tid, at in (getattr(tokenizer, "added_tokens_decoder", None) or {}).items():
            if getattr(at, "special", False) and at.content:
                specials[at.content] = int(tid)
        self.special_ids = set(specials.values())
        self.regex = re.compile("|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True))) if specials else None
        self.entries = OrderedDict()   # entry key -> array('i') of the full ids of one encoded prompt
        self.index = {}                # boundary digest -> (entry key, token offset of that boundary)
        self.entry_keys = {}           # entry key -> list of boundary digests it owns (for eviction)
        self.max_entries = max_entries
        self.stats = {"calls": 0, "hits": 0, "saved_chars": 0, "mismatch": 0, "uncacheable": 0}

    def _full(self, text, kw):
        return self.tok.encode(text, **kw)

    def encode(self, text, **kw):
        self.stats["calls"] += 1
        if self.regex is None:
            return self._full(text, kw)
        pos = [m.start() for m in self.regex.finditer(text)]
        if len(pos) < 2:
            return self._full(text, kw)
        h = hashlib.blake2b(digest_size=16); prev = 0; digs = []
        for p in pos:
            h.update(text[prev:p].encode("utf-8", "surrogatepass")); prev = p
            digs.append(h.copy().digest())
        ids = None
        for k in range(len(pos) - 1, 0, -1):
            hit = self.index.get(digs[k])
            if not hit:
                continue
            ekey, off = hit
            e = self.entries.get(ekey)
            if e is None:
                continue
            self.entries.move_to_end(ekey)
            ids = e[:off].tolist() + self.tok.encode(text[pos[k]:], add_special_tokens=False)
            self.stats["hits"] += 1; self.stats["saved_chars"] += pos[k]
            break
        if VERIFY and ids is not None:
            full = self._full(text, kw)
            if full != ids:
                self.stats["mismatch"] += 1
                logger.warning("tokenize prefix cache mismatch (%d vs %d tokens); using the full encode", len(ids), len(full))
            ids = full
        if ids is None:
            ids = self._full(text, kw)
        self._insert(digs, ids)
        return ids

    def _insert(self, digs, ids):
        toks = [i for i, t in enumerate(ids) if t in self.special_ids]
        if len(toks) != len(digs):          # a special string did not map to exactly one special id: do not cache
            self.stats["uncacheable"] += 1
            return
        ekey = digs[-1]
        if ekey in self.entries:
            self.entries.move_to_end(ekey); return
        self.entries[ekey] = array("i", ids)
        self.entry_keys[ekey] = digs
        for d, off in zip(digs, toks):
            self.index[d] = (ekey, off)
        while len(self.entries) > self.max_entries:
            old, _ = self.entries.popitem(last=False)
            for d in self.entry_keys.pop(old, ()):
                if self.index.get(d, (None,))[0] == old:
                    del self.index[d]


_ENCODERS = {}


def encode(tokenizer, text, **kw):
    """Drop-in for tokenizer.encode(text, **kw) on rendered chat prompts."""
    if not ENABLED:
        return tokenizer.encode(text, **kw)
    enc = _ENCODERS.get(id(tokenizer))
    if enc is None:
        enc = _ENCODERS[id(tokenizer)] = PrefixCachedEncoder(tokenizer)
    return enc.encode(text, **kw)
