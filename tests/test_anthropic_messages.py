import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class TestAnthropicMessagesEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_stream_true_returns_anthropic_sse_events(self):
        async def fake_stream_chat(req):
            yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"anthropic/claude-sonnet-4-6","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}' + "\n\n"
            yield 'data: [DONE]\n\n'

        with patch.object(main, 'stream_chat', fake_stream_chat):
            with self.client.stream(
                'POST',
                '/v1/messages',
                json={
                    'model': 'anthropic/claude-sonnet-4-6',
                    'max_tokens': 32,
                    'stream': True,
                    'messages': [{'role': 'user', 'content': 'Say hi'}],
                },
            ) as resp:
                body = resp.read().decode()

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['content-type'].split(';')[0], 'text/event-stream')
        self.assertIn('event: message_start', body)
        self.assertIn('event: content_block_start', body)
        self.assertIn('event: content_block_delta', body)
        self.assertIn('event: ping', body)
        self.assertIn('"text": "Hi"', body)
        self.assertIn('event: message_delta', body)
        self.assertIn('event: message_stop', body)

    def test_stream_false_returns_message_json(self):
        """Without stream=true, Anthropic spec requires a single Message JSON response."""
        async def fake_stream_chat(req):
            yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"opencode/deepseek-v4-flash-free","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}' + "\n\n"
            yield 'data: [DONE]\n\n'

        with patch.object(main, 'stream_chat', fake_stream_chat):
            resp = self.client.post(
                '/v1/messages',
                json={
                    'model': 'opencode/deepseek-v4-flash-free',
                    'max_tokens': 32,
                    'messages': [{'role': 'user', 'content': 'hi'}],
                },
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers['content-type'].split(';')[0], 'application/json')
        body = resp.json()
        self.assertEqual(body['type'], 'message')
        self.assertEqual(body['role'], 'assistant')
        self.assertEqual(body['content'], [{'type': 'text', 'text': 'ok'}])
        self.assertEqual(body['stop_reason'], 'end_turn')
        self.assertIn('input_tokens', body['usage'])
        self.assertIn('output_tokens', body['usage'])

    def test_content_blocks_are_linearized(self):
        """Multi-block content (text + tool_use + tool_result) must be flattened to a string prompt."""
        captured = {}

        async def fake_stream_chat(req):
            captured['req'] = req
            yield 'data: {"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}' + "\n\n"
            yield 'data: [DONE]\n\n'

        with patch.object(main, 'stream_chat', fake_stream_chat):
            resp = self.client.post(
                '/v1/messages',
                json={
                    'model': 'anthropic/claude-sonnet-4-6',
                    'max_tokens': 32,
                    'system': [{'type': 'text', 'text': 'sys'}],
                    'messages': [
                        {'role': 'user', 'content': 'find files'},
                        {'role': 'assistant', 'content': [
                            {'type': 'text', 'text': 'I will search'},
                            {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Bash', 'input': {'command': 'ls'}},
                        ]},
                        {'role': 'user', 'content': [
                            {'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': 'a.txt'},
                        ]},
                    ],
                },
            )
        self.assertEqual(resp.status_code, 200)
        prompt_msgs = captured['req'].messages
        # system + user + assistant + user
        self.assertEqual([m.role for m in prompt_msgs], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(prompt_msgs[0].content, 'sys')
        assistant_text = prompt_msgs[2].content
        self.assertIn('I will search', assistant_text)
        self.assertIn('<tool_use name="Bash" id="toolu_1">', assistant_text)
        self.assertIn('"command"', assistant_text)
        result_text = prompt_msgs[3].content
        self.assertIn('<tool_result for="toolu_1">', result_text)
        self.assertIn('a.txt', result_text)

    def test_finish_reason_length_maps_to_max_tokens(self):
        async def fake_stream_chat(req):
            yield 'data: {"choices":[{"index":0,"delta":{"content":"abc"},"finish_reason":"length"}]}' + "\n\n"
            yield 'data: [DONE]\n\n'

        with patch.object(main, 'stream_chat', fake_stream_chat):
            resp = self.client.post(
                '/v1/messages',
                json={
                    'model': 'anthropic/claude-sonnet-4-6',
                    'max_tokens': 3,
                    'messages': [{'role': 'user', 'content': 'hi'}],
                },
            )
        self.assertEqual(resp.json()['stop_reason'], 'max_tokens')


if __name__ == '__main__':
    unittest.main()
