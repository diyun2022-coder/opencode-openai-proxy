import asyncio
import unittest
from unittest.mock import patch

import main


async def _collect_stream(gen):
    chunks = []
    async for c in gen:
        chunks.append(c)
    return "".join(chunks)


class TestReasoningContentHandling(unittest.TestCase):
    def setUp(self):
        self.orig_show = main.SHOW_REASONING

    def tearDown(self):
        main.SHOW_REASONING = self.orig_show

    def _fake_req(self, model="m"):
        return type("R", (), {"model": model, "messages": [main.Msg(role="user", content="hi")], "stream": True})()

    # ── stream_chat path ────────────────────────────────────────────

    def test_reasoning_part_text_emitted_as_content(self):
        """Reasoning part's text should produce content deltas."""
        async def fake_events():
            # message.updated must arrive BEFORE part.updated so msg_roles is set
            yield {
                "payload": {
                    "type": "message.updated",
                    "properties": {
                        "info": {"id": "m1", "role": "assistant"},
                    },
                },
            }
            yield {
                "payload": {
                    "type": "message.part.updated",
                    "properties": {
                        "part": {
                            "id": "p1", "type": "reasoning",
                            "messageID": "m1", "text": "thinking text",
                        },
                    },
                },
            }
            yield {
                "payload": {
                    "type": "session.idle",
                    "properties": {},
                },
            }

        gen = main.stream_chat(self._fake_req())

        with (
            patch.object(main, "opencode_events", return_value=fake_events()),
            patch.object(main, "opencode_create_session", return_value="sess-1"),
            patch.object(main, "opencode_send", return_value=None),
        ):
            body = asyncio.run(_collect_stream(gen))

        self.assertIn("thinking text", body)

    def test_reasoning_content_delta_not_filtered(self):
        """Delta with field=reasoning_content should NOT be dropped."""
        async def fake_events():
            yield {
                "payload": {
                    "type": "message.part.updated",
                    "properties": {
                        "part": {
                            "id": "p1", "type": "reasoning",
                            "messageID": "m1", "text": "",
                        },
                    },
                },
            }
            yield {
                "payload": {
                    "type": "message.updated",
                    "properties": {
                        "info": {"id": "m1", "role": "assistant"},
                    },
                },
            }
            yield {
                "payload": {
                    "type": "message.part.delta",
                    "properties": {
                        "partID": "p1", "field": "reasoning_content", "delta": "thinking...",
                    },
                },
            }
            yield {
                "payload": {
                    "type": "session.idle",
                    "properties": {},
                },
            }

        gen = main.stream_chat(self._fake_req())

        with (
            patch.object(main, "opencode_events", return_value=fake_events()),
            patch.object(main, "opencode_create_session", return_value="sess-1"),
            patch.object(main, "opencode_send", return_value=None),
        ):
            body = asyncio.run(_collect_stream(gen))

        self.assertIn("thinking...", body)

    def test_text_delta_still_works(self):
        """Regular text deltas continue working unchanged."""
        async def fake_events():
            yield {
                "payload": {
                    "type": "message.part.updated",
                    "properties": {
                        "part": {
                            "id": "p1", "type": "text",
                            "messageID": "m1",
                        },
                    },
                },
            }
            yield {
                "payload": {
                    "type": "message.updated",
                    "properties": {
                        "info": {"id": "m1", "role": "assistant"},
                    },
                },
            }
            yield {
                "payload": {
                    "type": "message.part.delta",
                    "properties": {
                        "partID": "p1", "field": "text", "delta": "hello world",
                    },
                },
            }
            yield {
                "payload": {
                    "type": "session.idle",
                    "properties": {},
                },
            }

        gen = main.stream_chat(self._fake_req())

        with (
            patch.object(main, "opencode_events", return_value=fake_events()),
            patch.object(main, "opencode_create_session", return_value="sess-1"),
            patch.object(main, "opencode_send", return_value=None),
        ):
            body = asyncio.run(_collect_stream(gen))

        self.assertIn("hello world", body)

    # ── chat_completions route -------------------------------------------------

    def test_masked_api_key_uses_session_path(self):
        """api_key='***' should route through stream_chat, not _direct_stream."""
        from fastapi.testclient import TestClient
        client = TestClient(main.app)

        original_cache = main._provider_cache
        main._provider_cache = {
            "opencode": {
                "key": "",
                "base_url": "",
                "models": {
                    "deepseek-v4-flash-free": {
                        "url": "https://opencode.ai/zen/v1",
                        "api_key": "***",
                    },
                },
            },
        }

        call_path = []

        async def fake_stream(req):
            call_path.append("stream_chat")
            yield 'data: [DONE]\n\n'

        async def fake_direct(req, base, key):
            call_path.append("_direct_stream")

        try:
            with (
                patch.object(main, "stream_chat", fake_stream),
                patch.object(main, "_direct_stream", fake_direct),
                patch.object(main, "_refresh_providers"),
            ):
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": "opencode/deepseek-v4-flash-free",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                ) as resp:
                    resp.read()
        finally:
            main._provider_cache = original_cache

        self.assertEqual(call_path, ["stream_chat"],
                         "Should fall to stream_chat, not _direct_stream")

    def test_valid_api_key_uses_direct_path(self):
        """Real api_key should route through _direct_stream."""
        from fastapi.testclient import TestClient
        client = TestClient(main.app)

        original_cache = main._provider_cache
        main._provider_cache = {
            "anthropic": {
                "key": "sk-real-key-123",
                "base_url": "https://api.anthropic.com/v1",
                "models": {
                    "claude-sonnet-4-6": {
                        "url": "",
                        "api_key": "",
                    },
                },
            },
        }

        call_path = []

        async def fake_stream(req):
            call_path.append("stream_chat")
            yield 'data: [DONE]\n\n'

        async def fake_direct(req, base, key):
            call_path.append("_direct_stream")
            if False:
                yield  # pragma: no cover — async generator

        try:
            with (
                patch.object(main, "stream_chat", fake_stream),
                patch.object(main, "_direct_stream", fake_direct),
                patch.object(main, "_refresh_providers"),
            ):
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": "anthropic/claude-sonnet-4-6",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                ) as resp:
                    resp.read()
        finally:
            main._provider_cache = original_cache

        self.assertEqual(call_path, ["_direct_stream"],
                         "Should use _direct_stream for valid key")


if __name__ == '__main__':
    unittest.main()
