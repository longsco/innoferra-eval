#!/usr/bin/env python3
"""innoferra 10-01: share tok_prefix_cache across the engine's tokenizer processes through files in the container's /dev/shm.
The per-process cache nearly always hits at SOME boundary (system prompt + tools are common to many sessions), but it reaches the
previous turn of the same session only when that turn was tokenised by the same process (~1 in 8 with TOKW=8; uvicorn workers take
connections at random), so most turns still encode the whole conversation (0.14 / 0.34 / 0.55 s at <120k / 120-200k / 200k+ tokens),
the median-TTFT floor. Every encoded prompt is published as e_<entry>.bin (int32 ids) plus index files i_<boundary digest> =
'<entry> <token offset>' for its last SHARED_K special-token boundaries (atomic tmp+rename). A lookup takes the longest boundary
found locally or in the shared directory (shared probes only above the local hit, at most SHARED_K boundaries from the end). A
publishing process writes into the generation directory g<time // GEN_S>; lookups probe the newest SHARED_KEEP generations and
the first publish of a new generation renames older generations away and removes them with an external rm -rf (no directory scans
or GIL-heavy loops inside the tokenizer process; memory ~ SHARED_KEEP x GEN_S of traffic). Same digests and offsets as
the in-process cache, so the result is identical. Off unless SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR is set.
Usage: patch_tok_shared.py <sglang python root> [...] (applies to the pristine .pre-tokshared copy when present)"""
import pathlib, py_compile, shutil, sys
MOD_OLD = 'MAX_ENTRIES = int(os.environ.get("SGLANG_TOKENIZE_PREFIX_CACHE_ENTRIES", "512"))\n'
MOD_NEW = MOD_OLD + ('SHARED_DIR = os.environ.get("SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR", "")      # innoferra 10-01: cross-process cache\n'
                     'SHARED_K = int(os.environ.get("SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_BOUNDARIES", "64"))\n'
                     'SHARED_GEN_S = int(os.environ.get("SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_GEN_S", "600"))\n'
                     'SHARED_KEEP = int(os.environ.get("SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_KEEP", "3"))\n')
LOOKUP_OLD = '''        ids = None
        for k in range(len(pos) - 1, 0, -1):
            hit = self.index.get(digs[k])
            if not hit:
                continue
            ekey, off = hit
            e = self.entries.get(ekey)
            if e is None:
                continue
            self.entries.move_to_end(ekey)
            ids = e[:off].tolist() + self._tail(text, pos, k)
            self.stats["hits"] += 1; self.stats["saved_chars"] += pos[k]
            break
'''
LOOKUP_NEW = '''        ids = None
        kl = next((k for k in range(len(pos) - 1, 0, -1) if self.index.get(digs[k], (None,))[0] in self.entries), 0)
        if SHARED_DIR:      # innoferra 10-01: a longer prefix published by another tokenizer process beats the local hit
            ids = self._shared_lookup(text, pos, digs, kl + 1)
        if ids is None and kl:
            ekey, off = self.index[digs[kl]]
            self.entries.move_to_end(ekey)
            ids = self.entries[ekey][:off].tolist() + self._tail(text, pos, kl)
            self.stats["hits"] += 1; self.stats["saved_chars"] += pos[kl]
'''
INSERT_OLD = '''        self._insert(digs, ids)
        return ids
'''
INSERT_NEW = '''        self._insert(digs, ids)
        if SHARED_DIR:
            self._shared_publish(digs, ids)
        return ids
'''
METHODS = '''
    # innoferra 10-01: cross-process cache in SHARED_DIR (see patch_tok_shared.py)
    def _shared_lookup(self, text, pos, digs, kmin):
        gen = int(time.time() // SHARED_GEN_S)
        dirs = [os.path.join(SHARED_DIR, "g%d" % g) for g in range(gen, gen - SHARED_KEEP, -1)]
        for k in range(len(pos) - 1, max(kmin, len(pos) - SHARED_K) - 1, -1):
            name = "i_" + digs[k].hex()
            for d in dirs:
                try:
                    with open(os.path.join(d, name), "r") as f:
                        ekey, off = f.read().split()
                    off = int(off)
                    with open(os.path.join(d, "e_" + ekey + ".bin"), "rb") as f:
                        head = array("i"); head.frombytes(f.read(4 * off))
                    if len(head) != off:
                        continue
                except (OSError, ValueError):
                    continue
                self.stats["shared_hits"] = self.stats.get("shared_hits", 0) + 1; self.stats["saved_chars"] += pos[k]
                return head.tolist() + self._tail(text, pos, k)
        return None

    def _shared_publish(self, digs, ids):
        try:
            toks = self._last_toks            # computed by _insert() for the same ids
            if len(toks) != len(digs) or not digs:
                return
            gen = int(time.time() // SHARED_GEN_S); gdir = os.path.join(SHARED_DIR, "g%d" % gen)
            if gen != getattr(self, "_shared_gen", None):
                self._shared_gen = gen
                os.makedirs(gdir, exist_ok=True)
                _shared_retire(gen)
            ekey = digs[-1].hex(); ep = os.path.join(gdir, "e_" + ekey + ".bin")
            if not os.path.exists(ep):
                tmp = ep + ".%d.tmp" % os.getpid()
                with open(tmp, "wb") as f:
                    f.write(array("i", ids).tobytes())
                os.replace(tmp, ep)
            for d, off in list(zip(digs, toks))[-SHARED_K:]:
                ip = os.path.join(gdir, "i_" + d.hex()); tmp = ip + ".%d.tmp" % os.getpid()
                with open(tmp, "w") as f:
                    f.write("%s %d" % (ekey, off))
                os.replace(tmp, ip)
        except OSError:
            pass

'''
EVICT = '''

def _shared_retire(gen):
    """innoferra 10-01: move generations older than SHARED_KEEP aside (atomic rename: one process wins) and delete them outside Python."""
    try:
        for name in os.listdir(SHARED_DIR):
            if name.startswith("g") and name[1:].isdigit() and int(name[1:]) <= gen - SHARED_KEEP:
                dead = os.path.join(SHARED_DIR, "x%s.%d" % (name, os.getpid()))
                try:
                    os.rename(os.path.join(SHARED_DIR, name), dead)
                except OSError:
                    continue
                subprocess.Popen(["rm", "-rf", dead], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    except OSError:
        pass
'''
ANCHOR = '    def _insert(self, digs, ids):\n'
TAIL_ANCHOR = '\n\n_ENCODERS = {}\n'
TOKS_OLD = '        if len(toks) != len(digs):          # a special string did not map to exactly one special id: do not cache\n'
TOKS_NEW = '        self._last_toks = toks\n' + TOKS_OLD
for root in sys.argv[1:]:
    p = pathlib.Path(root) / "sglang/srt/entrypoints/openai/tok_prefix_cache.py"
    if not p.exists(): print("no tok_prefix_cache in", root); continue
    orig = pathlib.Path(str(p) + ".pre-tokshared")
    if orig.exists(): shutil.copy2(orig, p)
    else: shutil.copy2(p, orig)
    s = p.read_text()
    for old in (MOD_OLD, LOOKUP_OLD, INSERT_OLD, ANCHOR, TAIL_ANCHOR, TOKS_OLD, "import re\n"):
        assert s.count(old) == 1, f"pattern count {s.count(old)} for {old[:50]!r} in {p}"
    s = (s.replace("import re\n", "import re\nimport subprocess\nimport time\n").replace(MOD_OLD, MOD_NEW).replace(LOOKUP_OLD, LOOKUP_NEW)
          .replace(INSERT_OLD, INSERT_NEW).replace(ANCHOR, METHODS.lstrip("\n") + ANCHOR).replace(TAIL_ANCHOR, EVICT + TAIL_ANCHOR).replace(TOKS_OLD, TOKS_NEW))
    p.write_text(s); py_compile.compile(str(p), doraise=True); print("patched:", p)
