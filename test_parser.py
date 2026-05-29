#!/usr/bin/env python3
"""Quick test for the dual-format tool call parser."""

import sys
sys.path.insert(0, '/root/opencode-openai-proxy')

from main import _parse_tool_calls_from_text

# Test 1: JSON format (original)
json_text = """Let me check that.

<tool_call>
{"name": "terminal", "arguments": {"command": "ls -la"}}
</tool_call>"""

remaining, calls = _parse_tool_calls_from_text(json_text)
print("Test 1 - JSON format:")
print(f"  Remaining: {remaining!r}")
print(f"  Calls: {len(calls)}")
if calls:
    print(f"  First call: {calls[0]['function']['name']}")
    print(f"  Arguments: {calls[0]['function']['arguments']}")
print()

# Test 2: XML format (new)
xml_text = """Let me check that.

<tool_call>
<function=terminal>
<parameter=command>ls -la</parameter>
</function>
</tool_call>"""

remaining, calls = _parse_tool_calls_from_text(xml_text)
print("Test 2 - XML format:")
print(f"  Remaining: {remaining!r}")
print(f"  Calls: {len(calls)}")
if calls:
    print(f"  First call: {calls[0]['function']['name']}")
    print(f"  Arguments: {calls[0]['function']['arguments']}")
print()

# Test 3: Multiple parameters in XML
xml_multi = """<tool_call>
<function=read_file>
<parameter=path>/etc/hosts</parameter>
<parameter=encoding>utf-8</parameter>
</function>
</tool_call>"""

remaining, calls = _parse_tool_calls_from_text(xml_multi)
print("Test 3 - XML with multiple parameters:")
print(f"  Calls: {len(calls)}")
if calls:
    print(f"  Function: {calls[0]['function']['name']}")
    print(f"  Arguments: {calls[0]['function']['arguments']}")
print()

# Test 4: Missing closing tags (tolerance test)
xml_incomplete = """<tool_call>
<function=bash>
<parameter=command>echo hello"""

remaining, calls = _parse_tool_calls_from_text(xml_incomplete)
print("Test 4 - Incomplete XML (should still parse):")
print(f"  Calls: {len(calls)}")
if calls:
    print(f"  Function: {calls[0]['function']['name']}")
    print(f"  Arguments: {calls[0]['function']['arguments']}")
print()

print("✅ All tests completed")
