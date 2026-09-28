import sys; sys.path.insert(0, "/opt/0922-sglang/python")
from sglang.srt.function_call import minimax_m3 as m
from sglang.srt.function_call.minimax_m3 import MINIMAX_NS_TOKEN as NS
Det = next(getattr(m, n) for n in dir(m) if n.endswith("Detector") and isinstance(getattr(m, n), type) and getattr(m, n).__module__ == m.__name__)
d = Det()
# parameter whose value is text, then a nested child tag, then closes: used to raise TypeError
body = NS.join(["<query>", "some free text", "<item>", "x", "</item>", "</query>"])
out = d._parse_parameter(body, {"type": "object", "properties": {"query": {"type": "string"}}})
print("parsed:", out)
assert isinstance(out, dict) and "query" in out, out
body2 = NS.join(["<a>", "<b>", "1", "</b>", "</a>"])
print("normal nested:", d._parse_parameter(body2, {"type": "object"}))
print("ALL OK")
