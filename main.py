"""
OpenAI-compatible REST proxy for OpenCode CLI.

Translates OpenAI `/v1/chat/completions` calls into OpenCode session API calls,
and streams OpenCode's SSE events back as OpenAI-format chunks.

Prereq:
    opencode serve --port 4096

Env:
    OPENCODE_BASE_URL          default http://localhost:4096
    OPENCODE_SERVER_USERNAME   default "opencode" (only used if password set)
    OPENCODE_SERVER_PASSWORD   if set, enables HTTP Basic auth to OpenCode

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import os
import re
import time
import uuid
from typing import AsyncGenerator, Optional, Tuple, Union

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

OPENCODE_BASE = os.environ.get("OPENCODE_BASE_URL", "http://localhost:4096")
OPENCODE_USER = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
OPENCODE_PASS = os.environ.get("OPENCODE_SERVER_PASSWORD")
AUTH = (OPENCODE_USER, OPENCODE_PASS) if OPENCODE_PASS else None

# PROXY_AGENT_MODE controls whether opencode's agent loop is engaged.
#   "off" (default) — disable all tools so opencode acts as a pure model gateway.
#                     Use this when the host (claude code / hermes / Kiro) already
#                     has its own tool ecosystem and just wants opencode as the LLM.
#   "on"            — let opencode run its full agent and stream tool calls inline.
#                     Use this only when you want opencode's built-in tools
#                     (websearch / bash / read / ...) instead of the host's.
AGENT_MODE = os.environ.get("PROXY_AGENT_MODE", "off").lower()

# Inline rendering of agent events as content text. Only relevant when AGENT_MODE=on.
# Defaults are off so that even if someone flips AGENT_MODE=on, opencode's
# internal reasoning / tool noise doesn't leak into the host's response stream.
SHOW_TOOLS = os.environ.get("PROXY_SHOW_TOOLS", "0") not in ("0", "false", "False")
SHOW_REASONING = os.environ.get("PROXY_SHOW_REASONING", "0") not in ("0", "false", "False")
TOOL_RESULT_MAX = int(os.environ.get("PROXY_TOOL_RESULT_MAX", "400"))

# Cached list of opencode tool IDs — used to build the all-false map in AGENT_MODE=off.
_tool_ids_cache: Optional[list[str]] = None

app = FastAPI(title="OpenCode → OpenAI Proxy", version="0.2.0")


class UpstreamError(Exception):
    """Non-2xx response from opencode session API."""
    def __init__(self, status: int, body: str):
        super().__init__(f"opencode {status}: {body[:300]}")
        self.status = status
        self.body = body

# ---- Provider config cache (from OpenCode /config/providers) ----
_provider_cache: Optional[dict] = None
_provider_cache_ts: float = 0


async def _refresh_providers():
    global _provider_cache, _provider_cache_ts
    if _provider_cache and time.time() - _provider_cache_ts < 60:
        return
    try:
        async with httpx.AsyncClient(auth=AUTH, timeout=10) as client:
            r = await client.get(f"{OPENCODE_BASE}/config/providers")
            r.raise_for_status()
            data = r.json()
            providers = data.get("providers") if isinstance(data, dict) else data
            config: dict[str, dict] = {}
            for p in providers or []:
                pid = p.get("id")
                if not pid:
                    continue
                models: dict[str, dict] = {}
                for mid, mcfg in (p.get("models") or {}).items():
                    api = mcfg.get("api") or {}
                    models[mid] = {
                        "url": api.get("url") or "",
                        "api_key": api.get("apiKey") or p.get("key") or (p.get("options") or {}).get("apiKey") or "",
                    }
                config[pid] = {
                    "key": p.get("key") or "",
                    "base_url": p.get("base_url") or "",
                    "models": models,
                }
            _provider_cache = config
            _provider_cache_ts = time.time()
    except Exception:
        pass


def _model_api_info(provider: str, model: str) -> Optional[Tuple[str, str]]:
    """Return (api_base_url, api_key) for direct API calling, or None."""
    cfg = _provider_cache
    if not cfg or provider not in cfg:
        return None
    prov = cfg[provider]
    # Per-model URL takes priority
    if model in prov.get("models", {}):
        url = prov["models"][model].get("url")
        if url:
            return url.rstrip("/"), prov["models"][model].get("api_key") or prov.get("key")
    # Provider-level base_url
    if prov.get("base_url"):
        return prov["base_url"].rstrip("/"), prov.get("key")
    return None


# ---- Models ----
class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str
    messages: list[Msg]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None

    class Config:
        extra = "allow"


class AnthropicMsg(BaseModel):
    role: str
    # Anthropic allows either a plain string OR a list of content blocks.
    # Block shapes: text / image / document / tool_use / tool_result.
    content: Union[str, list]


class AnthropicReq(BaseModel):
    model: str
    messages: list[AnthropicMsg]
    # max_tokens is required by the Anthropic spec; keep optional here so
    # malformed clients still get a useful error from downstream rather than
    # a Pydantic 422 — we default it when forwarding.
    max_tokens: Optional[int] = None
    system: Optional[Union[str, list]] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    # Tools are accepted but opencode runs its own toolset — see _render_tools_hint.
    tools: Optional[list[dict]] = None
    tool_choice: Optional[dict] = None
    metadata: Optional[dict] = None

    class Config:
        extra = "allow"


def parse_model(model_id: str) -> tuple[str, str]:
    if "/" in model_id:
        p, m = model_id.split("/", 1)
        return p, m
    return "anthropic", model_id


def now() -> int:
    return int(time.time())


def cid() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def render_prompt(messages: list[Msg]) -> tuple[Optional[str], str]:
    # OpenCode sessions own their own state, but OpenAI requests are stateless
    # (full history every call), so we replay the conversation as one prompt.
    # System messages are sent via the `system` field of the message POST.
    system_parts: list[str] = []
    convo_parts: list[str] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            convo_parts.append(f"[{m.role.upper()}]\n{m.content}")
    system = "\n\n".join(system_parts) if system_parts else None
    return system, "\n\n".join(convo_parts)


# ---- Tool-calling injection for session path ----
# When the host (Hermes) passes tools but we can only route through OpenCode's
# session API (no direct API key), we inject tool definitions into the system
# prompt and instruct the model to emit tool calls in a parseable XML format.
# We then parse the model's text output and convert matching XML into OpenAI
# format tool_calls chunks.

_TOOL_SYSTEM_SUFFIX = """

## Available Tools

You have access to the following tools. When you want to call a tool, you MUST output the tool call blocks in the following XML format:

<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}
</tool_call>

CRITICAL RULES:
1. Each <tool_call> block must contain exactly ONE JSON object.
2. ALWAYS close every <tool_call> block with a matching </tool_call> tag on its own line. A missing </tool_call> will cause the tool call to fail to dispatch.
3. If you need to call multiple tools, use multiple separate <tool_call>...</tool_call> blocks — each one fully opened and closed.
4. After outputting tool call blocks, STOP IMMEDIATELY. Do NOT output any other text after the </tool_call> tags.
5. Do NOT predict or fabricate tool results. Wait for the actual results to be provided.
6. If you want to include a brief explanation before calling tools, put it BEFORE the <tool_call> blocks.

Here are the available tools:

"""


def _render_tools_for_prompt(tools: list[dict]) -> str:
    """Render OpenAI-format tool definitions into a text block for the system prompt."""
    parts: list[str] = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool.get("function", {})
        else:
            # Some clients pass tools without the wrapper
            fn = tool
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        parts.append(f"### {name}")
        if desc:
            parts.append(f"Description: {desc}")
        if params:
            # Show a compact JSON schema
            props = params.get("properties", {})
            required = params.get("required", [])
            if props:
                parts.append("Parameters:")
                for pname, pdef in props.items():
                    ptype = pdef.get("type", "any")
                    pdesc = pdef.get("description", "")
                    req_mark = " (required)" if pname in required else ""
                    parts.append(f"  - {pname}: {ptype}{req_mark} — {pdesc}")
        parts.append("")
    return "\n".join(parts)


def _inject_tools_into_system(system: Optional[str], tools: list[dict]) -> str:
    """Append tool definitions to the system prompt."""
    tool_text = _TOOL_SYSTEM_SUFFIX + _render_tools_for_prompt(tools)
    if system:
        return system + tool_text
    return tool_text.strip()


# Marker that opens a tool_call block. We don't require a paired </tool_call>
# close tag — many models drop it under length pressure or get truncated mid-
# stream, and the original strict-regex parser silently fell back to emitting
# the buffer as plain text in that case.
_TOOL_CALL_OPEN = re.compile(r"<\s*tool_call\s*>", re.IGNORECASE)


def _make_tool_call(obj: dict) -> Optional[dict]:
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    name = obj.get("name") or "unknown"
    arguments = obj.get("arguments", {})
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _parse_xml_tool_call(text: str, start: int) -> tuple[Optional[dict], int]:
    """Parse XML-format tool call: <function=name><parameter=key>value</parameter></function>

    Returns (tool_call_dict, end_position) or (None, start) if parsing fails.
    """
    i = start
    # Skip whitespace
    while i < len(text) and text[i] in " \t\r\n":
        i += 1

    # Look for <function=name> or <function name="...">
    if i >= len(text) or text[i] != '<':
        return None, start

    # Find the function tag
    func_match = re.match(r'<\s*function\s*=\s*([^>\s]+)\s*>', text[i:], re.IGNORECASE)
    if not func_match:
        # Try alternate format: <function name="...">
        func_match = re.match(r'<\s*function\s+name\s*=\s*["\']([^"\']+)["\']\s*>', text[i:], re.IGNORECASE)

    if not func_match:
        return None, start

    func_name = func_match.group(1)
    i += func_match.end()

    # Parse parameters
    arguments = {}
    while i < len(text):
        # Skip whitespace
        while i < len(text) and text[i] in " \t\r\n":
            i += 1

        if i >= len(text):
            break

        # Check for closing </function>
        if text[i:i+10].lower().startswith('</function'):
            close_match = re.match(r'</\s*function\s*>', text[i:], re.IGNORECASE)
            if close_match:
                i += close_match.end()
                break
            else:
                break

        # Parse <parameter=key>value</parameter> or <parameter name="key">value</parameter>
        param_match = re.match(r'<\s*parameter\s*=\s*([^>\s]+)\s*>', text[i:], re.IGNORECASE)
        if not param_match:
            param_match = re.match(r'<\s*parameter\s+name\s*=\s*["\']([^"\']+)["\']\s*>', text[i:], re.IGNORECASE)

        if not param_match:
            break

        param_name = param_match.group(1)
        i += param_match.end()

        # Find closing </parameter>
        close_tag = '</parameter>'
        close_idx = text.lower().find(close_tag, i)
        if close_idx == -1:
            # No closing tag, take rest until next tag or end
            next_tag = text.find('<', i)
            if next_tag == -1:
                param_value = text[i:].strip()
                i = len(text)
            else:
                param_value = text[i:next_tag].strip()
                i = next_tag
        else:
            param_value = text[i:close_idx].strip()
            i = close_idx + len(close_tag)

        arguments[param_name] = param_value

    if not func_name:
        return None, start

    return _make_tool_call({"name": func_name, "arguments": arguments}), i


def _parse_tool_calls_from_text(text: str) -> tuple[str, list[dict]]:
    """Parse tool_call blocks from model text output.

    Supports two formats:
    1. JSON: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. XML: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>

    Tolerant of:
    - Missing </tool_call> closing tags (the common failure mode)
    - Multiple tool calls after one marker
    - Stray non-JSON text between blocks
    - Truncation mid-JSON/XML (just drops the partial call)

    Returns (remaining_text, list_of_tool_calls).
    """
    if not _TOOL_CALL_OPEN.search(text):
        return text, []

    decoder = json.JSONDecoder()
    tool_calls: list[dict] = []
    first_real_marker: Optional[int] = None
    pos = 0

    while pos < len(text):
        m = _TOOL_CALL_OPEN.search(text, pos)
        if not m:
            break
        i = m.end()
        parsed_any = False

        # Skip whitespace to detect format
        while i < len(text) and text[i] in " \t\r\n":
            i += 1

        if i >= len(text):
            break

        # Detect format: JSON starts with {, XML starts with <
        if text[i] == '{':
            # JSON format (original logic)
            while i < len(text):
                while i < len(text) and text[i] in " \t\r\n":
                    i += 1
                if i >= len(text) or text[i] != "{":
                    break
                try:
                    obj, end_idx = decoder.raw_decode(text, i)
                except json.JSONDecodeError:
                    break
                tc = _make_tool_call(obj)
                if tc is None:
                    break
                tool_calls.append(tc)
                parsed_any = True
                i = end_idx
        elif text[i] == '<':
            # XML format (new logic)
            tc, end_idx = _parse_xml_tool_call(text, i)
            if tc:
                tool_calls.append(tc)
                parsed_any = True
                i = end_idx

        if parsed_any and first_real_marker is None:
            first_real_marker = m.start()
        # Always advance past this marker so we don't loop forever on a stray
        # <tool_call> with no JSON after it.
        pos = max(i, m.end())

    if not tool_calls:
        return text, []

    remaining = text[:first_real_marker].strip() if first_real_marker is not None else ""
    return remaining, tool_calls


def _short(s: str, limit: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= limit else s[:limit] + f"... [+{len(s) - limit} chars]"


def render_tool_call(tool: str, inp: dict) -> str:
    # Pick the most-informative single field per known tool, fall back to JSON.
    if not isinstance(inp, dict):
        return f"\n\n🔧 **{tool}**\n"
    summary = (
        inp.get("command")
        or inp.get("filePath")
        or inp.get("file_path")
        or inp.get("path")
        or inp.get("pattern")
        or inp.get("query")
        or inp.get("url")
    )
    if not summary:
        summary = json.dumps(inp, ensure_ascii=False)
    return f"\n\n🔧 **{tool}** `{_short(str(summary), 200)}`\n"


def render_tool_result(tool: str, content, ok: bool, err=None) -> str:
    if not ok:
        err_text = err if isinstance(err, str) else json.dumps(err, ensure_ascii=False) if err else "?"
        return f"❌ {tool} failed: {_short(err_text, 200)}\n"
    text = ""
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                text += c.get("text", "")
            elif isinstance(c, str):
                text += c
    elif isinstance(content, str):
        text = content
    elif isinstance(content, dict):
        text = json.dumps(content, ensure_ascii=False)
    text = _short(text, TOOL_RESULT_MAX)
    if text:
        return f"```\n{text}\n```\n"
    return f"✅ {tool}\n"


def chunk(
    id_: str,
    model: str,
    delta: Optional[str] = None,
    role: Optional[str] = None,
    finish: Optional[str] = None,
) -> str:
    d: dict = {}
    if role:
        d["role"] = role
    if delta is not None:
        d["content"] = delta
    payload = {
        "id": id_,
        "object": "chat.completion.chunk",
        "created": now(),
        "model": model,
        "choices": [{"index": 0, "delta": d, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def opencode_create_session(
    client: httpx.AsyncClient, provider: str, model: str
) -> str:
    # Pass an explicit title so opencode skips its built-in title generation,
    # which would otherwise hit claude-haiku-4-5 and fail / waste tokens.
    payload = {"title": "proxy", "model": {"providerID": provider, "id": model}}
    r = await client.post(f"{OPENCODE_BASE}/session", json=payload)
    r.raise_for_status()
    data = r.json()
    return data.get("id") or data["info"]["id"]


async def opencode_tool_ids(client: httpx.AsyncClient) -> list[str]:
    global _tool_ids_cache
    if _tool_ids_cache is not None:
        return _tool_ids_cache
    try:
        r = await client.get(f"{OPENCODE_BASE}/experimental/tool/ids")
        r.raise_for_status()
        ids = r.json()
        if isinstance(ids, list):
            _tool_ids_cache = [str(x) for x in ids]
            return _tool_ids_cache
    except Exception:
        pass
    # Fallback to the documented set if the endpoint is unavailable. Better to
    # over-disable than under-disable — unknown keys are ignored by opencode.
    _tool_ids_cache = [
        "bash", "read", "glob", "grep", "edit", "write", "task",
        "webfetch", "todowrite", "websearch", "skill", "apply_patch",
    ]
    return _tool_ids_cache


async def opencode_send(
    client: httpx.AsyncClient,
    session_id: str,
    provider: str,
    model: str,
    system: Optional[str],
    prompt: str,
) -> None:
    payload: dict = {
        "model": {"providerID": provider, "modelID": model},
        "parts": [{"type": "text", "text": prompt}],
    }
    if system:
        payload["system"] = system
    if AGENT_MODE == "off":
        tool_ids = await opencode_tool_ids(client)
        payload["tools"] = {tid: False for tid in tool_ids}
    r = await client.post(
        f"{OPENCODE_BASE}/session/{session_id}/message",
        json=payload,
    )
    if r.status_code >= 400:
        raise UpstreamError(r.status_code, r.text)


async def open_event_pump(
    client: httpx.AsyncClient,
) -> tuple[asyncio.Queue, asyncio.Task, asyncio.Event]:
    """Open the SSE stream eagerly and pump events into a queue.

    Returns (queue, task, connected_event). Caller should `await connected.wait()`
    before sending the prompt — otherwise fast models (e.g. opencode free tier)
    may finish before the SSE connection is established and we miss every delta.
    The queue receives parsed event dicts; a `None` sentinel signals end of stream.
    """
    q: asyncio.Queue = asyncio.Queue()
    connected = asyncio.Event()

    async def pump():
        try:
            async with client.stream(
                "GET",
                f"{OPENCODE_BASE}/global/event",
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()
                connected.set()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    try:
                        evt = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    await q.put(evt)
        except Exception:
            pass
        finally:
            connected.set()  # unblock waiters even on failure
            await q.put(None)

    task = asyncio.create_task(pump())
    return q, task, connected


async def stream_chat(req: ChatReq) -> AsyncGenerator[str, None]:
    id_ = cid()
    provider, model = parse_model(req.model)
    system, prompt = render_prompt(req.messages)

    # If the request includes tools, inject them into the system prompt so the
    # model knows what tools are available and how to call them.
    has_tools = bool(req.tools)
    if has_tools and req.tools:
        system = _inject_tools_into_system(system, req.tools)

    async with httpx.AsyncClient(auth=AUTH, timeout=None) as client:
        # Open the SSE stream eagerly (and wait for the connection to be live)
        # BEFORE creating the session / posting the prompt — otherwise fast
        # models finish before we subscribe and every delta is missed.
        events_q, pump_task, connected = await open_event_pump(client)
        await connected.wait()

        try:
            session_id = await opencode_create_session(client, provider, model)
        except Exception as e:
            yield chunk(id_, req.model, role="assistant")
            yield chunk(id_, req.model, delta=f"[proxy error creating session: {e}]")
            yield chunk(id_, req.model, finish="stop")
            yield "data: [DONE]\n\n"
            pump_task.cancel()
            return

        yield chunk(id_, req.model, role="assistant")

        send_task = asyncio.create_task(
            opencode_send(client, session_id, provider, model, system, prompt)
        )

        # opencode v1.15.x streams text in two complementary channels:
        #   - message.part.delta with field="text" carries token-level deltas
        #   - message.part.updated carries cumulative snapshots (start, end, sometimes mid-flight)
        # We track part type from message.part.updated, then emit incremental
        # content from BOTH channels — using the running "emitted length" to
        # avoid double-emitting if a cumulative snapshot arrives mid-stream.
        part_types: dict[str, str] = {}       # part_id → "text" | "reasoning" | "tool" | ...
        part_msg: dict[str, str] = {}         # part_id → messageID
        text_emitted: dict[str, int] = {}     # part_id → chars already streamed to client
        # message.id → role. We only emit text/reasoning for assistant messages;
        # the user message also produces a TextPart that would otherwise echo
        # the prompt back into the response.
        msg_roles: dict[str, str] = {}
        # Tool calls: emit headers/results on state transitions only.
        tool_parts: dict[str, str] = {}  # part_id → last status emitted

        def text_role_for_part(pid: str) -> Optional[str]:
            """Return "text"/"reasoning" if we should stream this part's text, else None."""
            ptype = part_types.get(pid)
            if ptype not in ("text", "reasoning"):
                return None
            if ptype == "reasoning" and not SHOW_REASONING:
                return None
            mid = part_msg.get(pid)
            if mid and msg_roles.get(mid) != "assistant":
                return None
            return ptype

        emitted_finish = False

        async def drain_send_error() -> Optional[str]:
            """If send_task failed, return a renderable error string. Else None."""
            if not send_task.done():
                return None
            exc = send_task.exception()
            if exc is None:
                return None
            if isinstance(exc, UpstreamError):
                return f"[opencode {exc.status}] {exc.body[:300]}"
            return f"[proxy send error: {exc}]"

        try:
            while True:
                evt = await events_q.get()
                if evt is None:
                    # SSE stream ended — fall through to finish
                    break
                # Surface send failures the moment they happen.
                err_str = await drain_send_error()
                if err_str:
                    yield chunk(id_, req.model, delta=err_str)
                    yield chunk(id_, req.model, finish="stop")
                    emitted_finish = True
                    yield "data: [DONE]\n\n"
                    return

                # GlobalEvent wraps the actual event in `payload`.
                pl = evt.get("payload") or evt
                t = pl.get("type", "")
                props = pl.get("properties") or {}

                # Skip events not for this session.
                sid = props.get("sessionID")
                if sid and sid != session_id:
                    continue

                if t == "message.part.delta":
                    # Token-level streaming channel. We've cached part_type from
                    # the matching message.part.updated that opens the part.
                    if props.get("field") not in ("text", "reasoning_content"):
                        continue
                    pid = props.get("partID")
                    if not pid:
                        continue
                    delta = props.get("delta") or ""
                    if not delta:
                        continue
                    if text_role_for_part(pid) is None:
                        # Either not a text/reasoning part, or messageID is not
                        # the assistant (the user's prompt also produces a text
                        # part — its deltas would echo the prompt back).
                        continue
                    text_emitted[pid] = text_emitted.get(pid, 0) + len(delta)
                    yield chunk(id_, req.model, delta=delta)

                elif t == "message.part.updated":
                    part = props.get("part") or {}
                    ptype = part.get("type")
                    pid = part.get("id")
                    if not pid:
                        continue
                    # Cache the part's type and messageID so message.part.delta
                    # events that reference partID can be routed correctly.
                    part_types[pid] = ptype
                    mid = part.get("messageID")
                    if mid:
                        part_msg[pid] = mid

                    if ptype in ("text", "reasoning"):
                        # Cumulative snapshot — only emit the suffix beyond what
                        # deltas have already streamed. If deltas were missed
                        # (e.g. for a provider that doesn't emit them), this is
                        # the fallback that fills in the gap.
                        if text_role_for_part(pid) is None:
                            continue
                        full = part.get("text") or ""
                        already = text_emitted.get(pid, 0)
                        if len(full) > already:
                            delta = full[already:]
                            text_emitted[pid] = len(full)
                            if delta:
                                yield chunk(id_, req.model, delta=delta)

                    elif ptype == "tool" and SHOW_TOOLS:
                        state = part.get("state") or {}
                        status = state.get("status")
                        last = tool_parts.get(pid)

                        if status == "running" and last != "running":
                            tool_parts[pid] = "running"
                            yield chunk(
                                id_,
                                req.model,
                                delta=render_tool_call(
                                    part.get("tool", "?"), state.get("input") or {}
                                ),
                            )
                        elif status == "completed" and last != "completed":
                            tool_parts[pid] = "completed"
                            yield chunk(
                                id_,
                                req.model,
                                delta=render_tool_result(
                                    part.get("tool", "?"),
                                    state.get("output") or state.get("content"),
                                    ok=True,
                                ),
                            )
                        elif status == "error" and last != "error":
                            tool_parts[pid] = "error"
                            yield chunk(
                                id_,
                                req.model,
                                delta=render_tool_result(
                                    part.get("tool", "?"),
                                    None,
                                    ok=False,
                                    err=state.get("error") or state.get("output"),
                                ),
                            )

                elif t == "message.updated":
                    # Watch for AssistantMessage.error — that's where APIError
                    # (statusCode, message) surfaces when the upstream LLM call fails.
                    info = props.get("info") or {}
                    info_id = info.get("id")
                    info_role = info.get("role")
                    if info_id and info_role:
                        msg_roles[info_id] = info_role
                    if info_role != "assistant":
                        continue
                    err = info.get("error")
                    if not err:
                        continue
                    data = err.get("data") or {}
                    msg = data.get("message") or err.get("name") or "unknown upstream error"
                    status = data.get("statusCode")
                    label = f"[upstream {status}] " if status else "[upstream error] "
                    yield chunk(id_, req.model, delta=label + str(msg))
                    yield chunk(id_, req.model, finish="stop")
                    emitted_finish = True
                    yield "data: [DONE]\n\n"
                    return

                elif t == "session.error":
                    err = props.get("error") or {}
                    data = err.get("data") or {}
                    msg = data.get("message") or err.get("name") or "unknown"
                    status = data.get("statusCode")
                    label = f"[session {status}] " if status else "[session error] "
                    yield chunk(id_, req.model, delta=label + str(msg))
                    yield chunk(id_, req.model, finish="stop")
                    emitted_finish = True
                    yield "data: [DONE]\n\n"
                    return

                elif t == "session.idle":
                    # Final chance to surface a send failure that finished racing
                    # against session.idle (e.g., send returned 4xx after idle fires).
                    err_str = await drain_send_error()
                    if err_str:
                        yield chunk(id_, req.model, delta=err_str)
                    yield chunk(id_, req.model, finish="stop")
                    emitted_finish = True
                    yield "data: [DONE]\n\n"
                    return
        finally:
            if not send_task.done():
                send_task.cancel()
            try:
                await send_task
            except (asyncio.CancelledError, Exception):
                pass
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass

        if not emitted_finish:
            yield chunk(id_, req.model, finish="stop")
            yield "data: [DONE]\n\n"


async def _direct_stream(req: ChatReq, api_base: str, api_key: str) -> AsyncGenerator[str, None]:
    """Call the provider's own OpenAI-compatible chat API directly (streaming).
    Tool calls pass through as native OpenAI tool_calls — the original dialog handles them."""
    id_ = cid()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    _, model_name = parse_model(req.model)
    payload = req.model_dump(exclude={"extra"}, exclude_none=True)
    payload["model"] = model_name
    payload["stream"] = True

    yield chunk(id_, req.model, role="assistant")
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream("POST", f"{api_base}/chat/completions", json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    yield chunk(id_, req.model, delta=f"[API error {resp.status_code}: {err.decode(errors='replace')[:300]}]")
                    yield chunk(id_, req.model, finish="stop")
                    yield "data: [DONE]\n\n"
                    return
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield "data: [DONE]\n\n"
                        else:
                            yield f"data: {data}\n\n"
    except Exception as e:
        yield chunk(id_, req.model, delta=f"[proxy: direct API error — {e}]")
        yield chunk(id_, req.model, finish="stop")
        yield "data: [DONE]\n\n"


def _anthropic_text(content: Union[str, list, None]) -> str:
    """Render an Anthropic message/system content field as a flat string.

    Anthropic uses typed content blocks (text/image/tool_use/tool_result/document).
    OpenCode's session API only accepts plain text in `parts[].text`, so we
    linearize the blocks with explicit delimiters so the model can still see
    the tool-call structure across multi-turn conversations.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(str(block.get("text") or ""))
        elif btype == "tool_use":
            name = block.get("name") or "tool"
            tid = block.get("id") or ""
            try:
                inp_str = json.dumps(block.get("input") or {}, ensure_ascii=False)
            except (TypeError, ValueError):
                inp_str = str(block.get("input"))
            parts.append(f'<tool_use name="{name}" id="{tid}">{inp_str}</tool_use>')
        elif btype == "tool_result":
            tid = block.get("tool_use_id") or ""
            inner = block.get("content")
            if isinstance(inner, list):
                inner_text = "".join(
                    str(c.get("text") or "")
                    for c in inner
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            elif isinstance(inner, str):
                inner_text = inner
            else:
                inner_text = ""
            err_attr = ' is_error="true"' if block.get("is_error") else ""
            parts.append(f'<tool_result for="{tid}"{err_attr}>{inner_text}</tool_result>')
        elif btype in ("image", "document"):
            parts.append(f"[{btype} attachment — not forwarded]")
    return "".join(parts)


def anthropic_to_chat(req: AnthropicReq) -> ChatReq:
    messages: list[Msg] = []
    system = _anthropic_text(req.system)
    if system:
        messages.append(Msg(role="system", content=system))
    for msg in req.messages:
        messages.append(Msg(role=msg.role, content=_anthropic_text(msg.content)))
    return ChatReq(
        model=req.model,
        messages=messages,
        stream=True,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )


def anthropic_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# OpenAI finish_reason → Anthropic stop_reason. Anything we don't recognise
# falls back to "end_turn" — that's the safest signal for clients.
_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
    "function_call": "tool_use",
}


def _estimate_input_tokens(req: AnthropicReq) -> int:
    # Rough char/4 estimate — we don't get real numbers from opencode.
    n = 0
    if req.system:
        n += len(_anthropic_text(req.system))
    for m in req.messages:
        n += len(_anthropic_text(m.content))
    return max(1, n // 4)


async def _iter_openai_chunks(req: AnthropicReq) -> AsyncGenerator[dict, None]:
    """Drive stream_chat and yield parsed OpenAI chunk dicts (skip role-only)."""
    async for line in stream_chat(anthropic_to_chat(req)):
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            continue


async def stream_anthropic(req: AnthropicReq) -> AsyncGenerator[str, None]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = _estimate_input_tokens(req)

    yield anthropic_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": req.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )
    yield anthropic_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    # Optional but recommended by Anthropic — some SDKs use it as a keep-alive marker.
    yield anthropic_event("ping", {"type": "ping"})

    output_chars = 0
    stop_reason = "end_turn"

    async for obj in _iter_openai_chunks(req):
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            text = d.get("content")
            if text:
                output_chars += len(text)
                yield anthropic_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            finish = ch.get("finish_reason")
            if finish:
                stop_reason = _FINISH_TO_STOP.get(finish, "end_turn")

    output_tokens = max(1, output_chars // 4) if output_chars else 0

    yield anthropic_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": 0},
    )
    yield anthropic_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield anthropic_event("message_stop", {"type": "message_stop"})


async def collect_anthropic_message(req: AnthropicReq) -> dict:
    """Non-streaming path: buffer the stream into a single Anthropic Message JSON."""
    text_acc: list[str] = []
    stop_reason = "end_turn"
    async for obj in _iter_openai_chunks(req):
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            text = d.get("content")
            if text:
                text_acc.append(text)
            finish = ch.get("finish_reason")
            if finish:
                stop_reason = _FINISH_TO_STOP.get(finish, "end_turn")
    body = "".join(text_acc)
    output_tokens = max(1, len(body) // 4) if body else 0
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": req.model,
        "content": [{"type": "text", "text": body}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": _estimate_input_tokens(req),
            "output_tokens": output_tokens,
        },
    }


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


async def _collect_chat_completion(req: ChatReq) -> dict:
    """Non-streaming path: buffer the stream into a single chat completion JSON.

    Used by clients for auxiliary tasks like title generation where they send
    stream=false and expect a complete JSON response.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    finish_reason = "stop"

    if req.tools:
        # Buffer and parse tool calls
        async for line in stream_chat(req):
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                text = d.get("content")
                if text:
                    text_parts.append(text)

        full_text = "".join(text_parts)
        remaining, tool_calls = _parse_tool_calls_from_text(full_text)
        if tool_calls:
            finish_reason = "tool_calls"
            text_parts = [remaining] if remaining else []
        else:
            text_parts = [full_text] if full_text else []
    else:
        async for line in stream_chat(req):
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                text = d.get("content")
                if text:
                    text_parts.append(text)
                fr = ch.get("finish_reason")
                if fr:
                    finish_reason = fr

    content = "".join(text_parts) if text_parts else None
    message: dict = {"role": "assistant"}
    if content:
        message["content"] = content
    else:
        message["content"] = None
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": cid(),
        "object": "chat.completion",
        "created": now(),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def _stream_chat_with_tool_parsing(req: ChatReq) -> AsyncGenerator[str, None]:
    """Stream text in real-time, switching to buffer mode only when <tool_call> is detected.

    Text before any tool_call block is streamed token-by-token. Once the
    <tool_call> marker appears, the remainder is buffered and parsed into
    OpenAI-format tool_calls chunks.
    """
    id_ = cid()
    MARKER = "<tool_call>"
    MARKER_LEN = len(MARKER)

    yield chunk(id_, req.model, role="assistant")

    buffer = ""
    tool_buffer = ""
    in_tool_mode = False

    async for line in stream_chat(req):
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            text = d.get("content")
            if not text:
                continue

            if in_tool_mode:
                tool_buffer += text
                continue

            buffer += text

            marker_pos = buffer.find(MARKER)
            if marker_pos != -1:
                before = buffer[:marker_pos]
                if before:
                    yield chunk(id_, req.model, delta=before)
                tool_buffer = buffer[marker_pos:]
                in_tool_mode = True
                buffer = ""
                continue

            # Hold back any suffix that could be the start of MARKER
            hold_back = 0
            for i in range(min(MARKER_LEN - 1, len(buffer)), 0, -1):
                if buffer.endswith(MARKER[:i]):
                    hold_back = i
                    break

            safe_end = len(buffer) - hold_back
            if safe_end > 0:
                yield chunk(id_, req.model, delta=buffer[:safe_end])
                buffer = buffer[safe_end:]

    # Stream ended — flush
    if in_tool_mode:
        _, tool_calls = _parse_tool_calls_from_text(tool_buffer)
        if tool_calls:
            for idx, tc in enumerate(tool_calls):
                tc_delta: dict = {
                    "index": idx,
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": ""},
                }
                payload = {
                    "id": id_,
                    "object": "chat.completion.chunk",
                    "created": now(),
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": [tc_delta]},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                tc_args_delta: dict = {
                    "index": idx,
                    "function": {"arguments": tc["function"]["arguments"]},
                }
                payload = {
                    "id": id_,
                    "object": "chat.completion.chunk",
                    "created": now(),
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": [tc_args_delta]},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            yield chunk(id_, req.model, finish="tool_calls")
        else:
            if tool_buffer:
                yield chunk(id_, req.model, delta=tool_buffer)
            yield chunk(id_, req.model, finish="stop")
    else:
        if buffer:
            yield chunk(id_, req.model, delta=buffer)
        yield chunk(id_, req.model, finish="stop")

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatReq):
    await _refresh_providers()
    provider, model_name = parse_model(req.model)
    api_info = _model_api_info(provider, model_name)

    # Direct API path: only when we have a base URL AND a real provider API key
    # (sk-... shape). OAuth-style tokens (e.g. opencode-issued `fe_oa_...` for
    # free-tier anthropic, or `***` masked keys) need opencode's own header /
    # protocol handling — using them as bare Bearer against /chat/completions
    # gets the request treated as an anonymous free-tier call (FreeUsageLimitError).
    if api_info:
        api_base, api_key = api_info
        if api_base and api_key and api_key.startswith("sk-"):
            return StreamingResponse(
                _direct_stream(req, api_base, api_key),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )

    # Non-streaming path: buffer the response into a single JSON object.
    # Used by clients for title generation, embeddings, etc.
    if not req.stream:
        return await _collect_chat_completion(req)

    # Fallback: route through OpenCode agent.
    # When tools are present, we buffer the response and parse tool calls.
    if req.tools:
        return StreamingResponse(
            _stream_chat_with_tool_parsing(req),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    return StreamingResponse(
        stream_chat(req),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/v1/messages")
async def anthropic_messages(req: AnthropicReq):
    if req.stream:
        return StreamingResponse(
            stream_anthropic(req),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
    # Non-streaming clients (Anthropic SDK with stream=False) expect a single
    # Message JSON, not SSE — buffer the stream into one response.
    return await collect_anthropic_message(req)


@app.get("/v1/models")
async def models():
    try:
        async with httpx.AsyncClient(auth=AUTH, timeout=10) as client:
            r = await client.get(f"{OPENCODE_BASE}/config/providers")
            r.raise_for_status()
            data = r.json()
    except Exception:
        return {"object": "list", "data": []}

    # /config/providers returns {"providers": [...], "default": {...}}
    providers = data.get("providers") if isinstance(data, dict) else data
    out = []
    for p in providers or []:
        pid = p.get("id") or p.get("name")
        for m in (p.get("models") or {}).keys() if isinstance(p.get("models"), dict) else (p.get("models") or []):
            mid = m if isinstance(m, str) else (m.get("id") if isinstance(m, dict) else None)
            if mid and pid:
                out.append(
                    {
                        "id": f"{pid}/{mid}",
                        "object": "model",
                        "created": now(),
                        "owned_by": pid,
                    }
                )
    return {"object": "list", "data": out}


@app.get("/health")
async def health():
    return {"ok": True, "opencode_base": OPENCODE_BASE, "agent_mode": AGENT_MODE}
