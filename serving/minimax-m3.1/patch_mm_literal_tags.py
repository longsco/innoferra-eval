#!/usr/bin/env python3
"""innoferra 10-01: M3.1 multimodal processor treats literal '<image>', '<|image|>', '<|image_pad|>', '<video>', '<|video|>' in
USER TEXT as media placeholders. Real traffic contains them (HTML <video> tags, prompts about other VLMs), so a request with images
plus such text fails: 'More IMAGE tokens found than corresponding data provided' -> HTTP 500 (v3 session with 6 literal <image>),
or 'No data iterator found for token: <video>' -> HTTP 400. Production answers both with 200. M3.1's chat template emits only the
native tokens ']<]image[>[' / ']<]video[>[' (ids 200025/200026) and the generic tags are not special tokens in its tokenizer, so the
placeholder regexes are narrowed to the native tokens; the literal tags stay ordinary text.
Usage: patch_mm_literal_tags.py <sglang python root> [...]"""
import pathlib, py_compile, shutil, sys
OLD_I = 'image_token_regex=re.compile(\n                r"<image>|<\\|image\\|>|<\\|image_pad\\|>|\\]\\<\\]image\\[\\>\\["\n            ),'
NEW_I = 'image_token_regex=re.compile(r"\\]\\<\\]image\\[\\>\\["),   # innoferra 10-01: native token only (literal <image> is user text)'
OLD_V = 'video_token_regex=re.compile(r"<video>|<\\|video\\|>|\\]\\<\\]video\\[\\>\\["),'
NEW_V = 'video_token_regex=re.compile(r"\\]\\<\\]video\\[\\>\\["),   # innoferra 10-01: native token only (literal <video> is user text)'
for root in sys.argv[1:]:
    p = pathlib.Path(root) / "sglang/srt/multimodal/processors/minimax_m3_vl.py"; s = p.read_text()
    if "innoferra 10-01: native token only" in s: print("already patched:", p); continue
    assert s.count(OLD_I) == 1 and s.count(OLD_V) == 1, f"pattern not found in {p}"
    shutil.copy2(p, str(p) + ".pre-mmtags"); p.write_text(s.replace(OLD_I, NEW_I).replace(OLD_V, NEW_V))
    py_compile.compile(str(p), doraise=True); print("patched:", p)
