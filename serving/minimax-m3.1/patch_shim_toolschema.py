#!/usr/bin/env python3
"""Gateway shim: drop null-valued JSON-schema keywords inside tool parameter schemas (TOOL_SCHEMA_DROP_NULL=1).
Real M3.1 traffic sends e.g. {"type": "number", "maximum": null}; production accepts it, while the 0922 engine's
Draft202012Validator.check_schema rejects the request with 400 ("None is not of type 'number'"). A null keyword carries no
constraint, so removing it keeps the tool's meaning. 'default'/'const'/'examples' may legally be null and are kept, as are
property names (keys under properties/patternProperties/$defs/definitions are schemas, not keywords). Usage: patch_shim_toolschema.py shim.py"""
import sys
p = sys.argv[1]; s = open(p).read()
if "TOOL_SCHEMA_DROP_NULL" in s: print("already patched"); sys.exit(0)
anchor = "def translate(body):\n"
assert anchor in s
func = '''TOOL_SCHEMA_DROP_NULL = os.environ.get("TOOL_SCHEMA_DROP_NULL", "0") == "1"   # drop null-valued keywords in tool schemas (prod accepts them; engine 400s)
_NULL_OK = {"default", "const", "examples"}
_SCHEMA_MAPS = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
def _drop_null_kw(node, depth=0):
    if depth > 64: return 0
    n = 0
    if isinstance(node, dict):
        for k in [k for k, v in node.items() if v is None and k not in _NULL_OK]: del node[k]; n += 1
        for k, v in node.items():
            if k in _SCHEMA_MAPS and isinstance(v, dict):
                for sub in v.values(): n += _drop_null_kw(sub, depth + 1)
            elif k not in _NULL_OK: n += _drop_null_kw(v, depth + 1)
    elif isinstance(node, list):
        for v in node: n += _drop_null_kw(v, depth + 1)
    return n
def drop_null_tool_schema(body):
    tools = body.get("tools")
    if not isinstance(tools, list): return 0
    n = 0
    for t in tools:
        f = t.get("function") if isinstance(t, dict) else None
        if isinstance(f, dict) and isinstance(f.get("parameters"), dict): n += _drop_null_kw(f["parameters"])
    return n

'''
s = s.replace(anchor, func + anchor + "    if TOOL_SCHEMA_DROP_NULL: drop_null_tool_schema(body)\n", 1)
open(p, "w").write(s); print("patched")
