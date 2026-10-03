#!/usr/bin/env python3
"""Gateway shim: remove invalid 'type' values inside tool parameter schemas (TOOL_SCHEMA_FIX_TYPE=1, default on in run_gateway.sh).
Real M3.1 traffic (fair test v3.2, buckets b02+b03, 10-03 00:25 PDT) has a client that sends 218 tools per request, 44 of them with a
property schema {"type": {}}. Production accepts and renders it; the 0922 engine's Draft202012Validator.check_schema rejects the WHOLE
request with 400 ("{} is not valid under any of the given schemas"), so all 60 requests of that session failed (1.1% errors, 13 of 15
minutes lost). A 'type' that is not a string or a list of strings carries no usable constraint, so deleting it keeps the tool's
meaning (the property then accepts any value). String types are left alone (the engine normalises aliases such as "int" itself);
a list keeps its string members and is deleted only when none remain. Property names (keys under properties/patternProperties/
$defs/definitions/dependentSchemas) are schemas, not keywords, and are never touched.
Usage: patch_shim_toolschema_type.py /data01/minimax31/gateway   (edits shim.py and run_gateway.sh; backups *.pre-fixtype)"""
import os, shutil, sys
d = sys.argv[1] if len(sys.argv) > 1 else "/data01/minimax31/gateway"
def edit(path, pairs, tag):
    s = open(path).read()
    if tag in s: print("already patched", path); return
    bak = path + ".pre-fixtype"
    if not os.path.exists(bak): shutil.copy2(path, bak)
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    open(path + ".tmp", "w").write(s); shutil.copymode(path, path + ".tmp"); os.replace(path + ".tmp", path); print("patched", path)
FUNC = '''TOOL_SCHEMA_FIX_TYPE = os.environ.get("TOOL_SCHEMA_FIX_TYPE", "0") == "1"   # innoferra 10-03: drop non-string 'type' values (prod accepts; engine 400s)
def _fix_type_kw(node, depth=0):
    if depth > 64: return 0
    n = 0
    if isinstance(node, dict):
        t = node.get("type", "")
        if "type" in node and not isinstance(t, str):
            keep = [x for x in t if isinstance(x, str)] if isinstance(t, list) else []
            if keep and len(keep) == len(t): pass
            elif keep: node["type"] = keep; n += 1
            else: del node["type"]; n += 1
        for k, v in node.items():
            if k in _SCHEMA_MAPS and isinstance(v, dict):
                for sub in v.values(): n += _fix_type_kw(sub, depth + 1)
            elif k not in _NULL_OK and k != "type": n += _fix_type_kw(v, depth + 1)
    elif isinstance(node, list):
        for v in node: n += _fix_type_kw(v, depth + 1)
    return n
def fix_type_tool_schema(body):
    tools = body.get("tools")
    if not isinstance(tools, list): return 0
    n = 0
    for t in tools:
        f = t.get("function") if isinstance(t, dict) else None
        if isinstance(f, dict) and isinstance(f.get("parameters"), dict): n += _fix_type_kw(f["parameters"])
    return n

'''
edit(os.path.join(d, "shim.py"), [
    ("def translate(body):\n", FUNC + "def translate(body):\n"),
    ("    if TOOL_SCHEMA_DROP_NULL: drop_null_tool_schema(body)\n",
     "    if TOOL_SCHEMA_DROP_NULL: drop_null_tool_schema(body)\n"
     "    if TOOL_SCHEMA_FIX_TYPE: fix_type_tool_schema(body)          # innoferra 10-03\n"),
], "TOOL_SCHEMA_FIX_TYPE")
edit(os.path.join(d, "run_gateway.sh"), [
    ('-e TOOL_SCHEMA_DROP_NULL="${TOOL_SCHEMA_DROP_NULL:-0}" \\\n',
     '-e TOOL_SCHEMA_DROP_NULL="${TOOL_SCHEMA_DROP_NULL:-0}" -e TOOL_SCHEMA_FIX_TYPE="${TOOL_SCHEMA_FIX_TYPE:-1}" \\\n'),
], "TOOL_SCHEMA_FIX_TYPE")
