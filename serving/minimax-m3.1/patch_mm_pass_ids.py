#!/usr/bin/env python3
"""innoferra 10-01: M3.1 is multimodal (minimax_m3_vl), so serving_chat renders + encodes the prompt (tok_prefix_cache), DECODES the ids
back to text and passes text; the tokenizer manager then re-encodes the whole conversation (tokenizer([text])) on every request. For a
request without image/video/audio data the multimodal processor does not run anyway, so pass the encoded ids straight through (as for a
text-only model) and skip the decode. mm_ids_verify.py: identical ids on 300/300 real turns; removes decode + re-encode, p50 0.13 s,
p90 0.70 s (0.73 s median above 200k tokens). Requests with media keep the text path. Off unless SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1.
Usage: patch_mm_pass_ids.py <sglang python root> [...]"""
import pathlib, py_compile, shutil, sys
IMP_OLD = "import copy\nimport json\n"
IMP_NEW = "import copy\nimport json\nimport os\n"
FLAG_ANCHOR = "\nlogger = logging.getLogger(__name__)\n"
FLAG = FLAG_ANCHOR + ('_MM_PASS_IDS = os.environ.get("SGLANG_MM_PASS_IDS_WITHOUT_MEDIA", "0") == "1"   '
                      '# innoferra 10-01 (patch_mm_pass_ids.py)\n')
DEC_OLD = '''            if is_multimodal:
                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)
'''
DEC_NEW = '''            if is_multimodal and not (_MM_PASS_IDS and not (image_data or video_data or audio_data)):   # innoferra 10-01: no decode without media
                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)
'''
KW_OLD = '''            if (
                self.chat_encoding_spec == "inkling"
                and isinstance(processed_messages.prompt_ids, list)
                and processed_messages.prompt_ids
            ):
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"text": processed_messages.prompt}
'''
KW_NEW = '''            if (
                _MM_PASS_IDS      # innoferra 10-01: no media -> the MM processor would not run; skip the full re-encode of the text
                and not (processed_messages.image_data or processed_messages.video_data or processed_messages.audio_data)
                and isinstance(processed_messages.prompt_ids, list)
                and processed_messages.prompt_ids
            ):
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
            elif (
                self.chat_encoding_spec == "inkling"
                and isinstance(processed_messages.prompt_ids, list)
                and processed_messages.prompt_ids
            ):
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"text": processed_messages.prompt}
'''
for root in sys.argv[1:]:
    p = pathlib.Path(root) / "sglang/srt/entrypoints/openai/serving_chat.py"
    s = p.read_text()
    if "patch_mm_pass_ids.py" in s: print("already patched:", p); continue
    for old in (IMP_OLD, FLAG_ANCHOR, DEC_OLD, KW_OLD):
        assert s.count(old) == 1, f"pattern count {s.count(old)} for {old[:60]!r} in {p}"
    shutil.copy2(p, str(p) + ".pre-mmpassids")
    s = s.replace(IMP_OLD, IMP_NEW).replace(FLAG_ANCHOR, FLAG).replace(DEC_OLD, DEC_NEW).replace(KW_OLD, KW_NEW)
    p.write_text(s); py_compile.compile(str(p), doraise=True); print("patched:", p)
