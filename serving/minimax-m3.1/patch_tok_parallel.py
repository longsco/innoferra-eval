#!/usr/bin/env python3
"""innoferra 10-01: parallel tokenisation inside tok_prefix_cache.PrefixCachedEncoder. The median-TTFT floor is per-request
re-tokenisation (full encode 0.14 / 0.34 / 0.55 s at <120k / 120-200k / 200k+ tokens; the per-process prefix cache only hits when a
session's next turn reaches the same tokenizer worker). Text between added special tokens tokenises independently (the property the
prefix cache already relies on), so a cache miss - or a long uncached tail after a hit - is split at special-token boundaries into
SGLANG_TOKENIZE_PARALLEL_CHUNKS pieces and encoded with the Rust tokenizer's encode_batch (rayon threads), then concatenated.
Offline on 16 real prompts: bit-identical, median 0.281 -> 0.073 s (16 chunks). Off unless SGLANG_TOKENIZE_PARALLEL_CHUNKS > 1.
Usage: patch_tok_parallel.py <sglang python root> [...]"""
import pathlib, py_compile, shutil, sys
MOD_OLD = 'MAX_ENTRIES = int(os.environ.get("SGLANG_TOKENIZE_PREFIX_CACHE_ENTRIES", "512"))\n'
MOD_NEW = MOD_OLD + ('PAR_CHUNKS = int(os.environ.get("SGLANG_TOKENIZE_PARALLEL_CHUNKS", "0"))          # innoferra 10-01: >1 = parallel encode\n'
                     'PAR_MIN_CHARS = int(os.environ.get("SGLANG_TOKENIZE_PARALLEL_MIN_CHARS", "40000"))  # only texts at least this long\n')
HIT_OLD = '            ids = e[:off].tolist() + self.tok.encode(text[pos[k]:], add_special_tokens=False)\n'
HIT_NEW = '            ids = e[:off].tolist() + self._tail(text, pos, k)\n'
MISS_OLD = '        if ids is None:\n            ids = self._full(text, kw)\n        self._insert(digs, ids)\n'
MISS_NEW = ('        if ids is None:\n'
            '            ids = self._miss(text, pos, kw)\n'
            '        self._insert(digs, ids)\n')
METHODS = '''
    # innoferra 10-01: parallel encode of [start:] split at special-token boundaries (identical ids, rayon threads)
    def _par(self, text, cuts):
        target = max(1, len(text) // PAR_CHUNKS); segs = []; start = 0
        for p in cuts:
            if p - start >= target:
                segs.append(text[start:p]); start = p
        segs.append(text[start:])
        if len(segs) < 2:
            return self.tok.encode(text, add_special_tokens=False)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        out = []
        for e in self.tok._tokenizer.encode_batch(segs, add_special_tokens=False):
            out.extend(e.ids)
        self.stats["parallel"] = self.stats.get("parallel", 0) + 1
        return out

    def _tail(self, text, pos, k):
        tail = text[pos[k]:]
        if PAR_CHUNKS > 1 and len(tail) >= PAR_MIN_CHARS and getattr(self.tok, "_tokenizer", None) is not None:
            return self._par(tail, [p - pos[k] for p in pos[k + 1:]])
        return self.tok.encode(tail, add_special_tokens=False)

    def _miss(self, text, pos, kw):
        if PAR_CHUNKS > 1 and len(text) >= PAR_MIN_CHARS and getattr(self.tok, "_tokenizer", None) is not None:
            if not kw.get("add_special_tokens", True) or not self._adds_specials():
                return self._par(text, [p for p in pos if p > 0])
        return self._full(text, kw)

    def _adds_specials(self):
        if not hasattr(self, "_adds"):
            self._adds = self.tok.encode("a", add_special_tokens=True) != self.tok.encode("a", add_special_tokens=False)
        return self._adds
'''
ANCHOR = '    def _insert(self, digs, ids):\n'
for root in sys.argv[1:]:
    p = pathlib.Path(root) / "sglang/srt/entrypoints/openai/tok_prefix_cache.py"; s = p.read_text()
    if "innoferra 10-01: parallel encode" in s: print("already patched:", p); continue
    for old in (MOD_OLD, HIT_OLD, MISS_OLD, ANCHOR):
        assert s.count(old) == 1, f"pattern not found: {old[:60]!r} in {p}"
    shutil.copy2(p, str(p) + ".pre-tokpar")
    s = s.replace(MOD_OLD, MOD_NEW).replace(HIT_OLD, HIT_NEW).replace(MISS_OLD, MISS_NEW).replace(ANCHOR, METHODS.lstrip("\n") + "\n" + ANCHOR)
    p.write_text(s); py_compile.compile(str(p), doraise=True); print("patched:", p)
