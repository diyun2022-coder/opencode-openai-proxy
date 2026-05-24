# opencode-openai-proxy

把 [opencode](https://opencode.ai) 包装成 OpenAI / Anthropic 兼容的 HTTP 接口，让任何支持 OpenAI 或 Anthropic SDK 的客户端都能直接调用 opencode 背后的模型（含 agent 能力）。

## Endpoints

| Method | Path | 用途 |
| ------ | ---- | ---- |
| POST | `/v1/chat/completions` | OpenAI 兼容 chat completions，支持流式 |
| POST | `/v1/messages` | Anthropic Messages API 兼容 |
| GET  | `/v1/models` | 列出 opencode 已配置的模型 |
| GET  | `/health` | 健康检查 |

## 依赖

- Python 3.10+
- `opencode` CLI（用于 `opencode serve`）

## 安装与启动

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./start.sh
```

`start.sh` 会自动拉起 `opencode serve --port 4096`，然后用 uvicorn 启动代理（默认 `0.0.0.0:8000`）。

## 环境变量

| 变量 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `OPENCODE_BASE_URL` | `http://localhost:4096` | opencode server 地址 |
| `OPENCODE_SERVER_USERNAME` | `opencode` | HTTP Basic 用户名（仅在设置了密码时启用）|
| `OPENCODE_SERVER_PASSWORD` | — | 设置后启用 HTTP Basic auth |
| `PROXY_PORT` | `8000` | 代理监听端口 |
| `PROXY_AGENT_MODE` | `off` | `off` 仅作模型网关（host 自带工具链时用这个）；`on` 启用 opencode agent + 工具 |
| `PROXY_SHOW_TOOLS` | `0` | agent 模式下是否把工具调用渲染进文本（`1` 开启） |
| `PROXY_SHOW_REASONING` | `0` | agent 模式下是否把推理过程渲染进文本（`1` 开启） |
| `PROXY_TOOL_RESULT_MAX` | `400` | 工具结果文本截断长度 |

## 调用示例

OpenAI 客户端：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "opencode/big-pickle",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }'
```

Anthropic 客户端：

```bash
curl http://localhost:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "opencode/big-pickle",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

## 测试

```bash
pytest tests/
```
