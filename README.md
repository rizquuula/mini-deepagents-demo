# mini-deepagents

A minimal agent harness that works inside an **8K-token context window**.
Today it is a small web chat with a hard 8K cap, SQLite history, and replies that stream to the browser over SSE.

## Why this project exists

[Deep Agents](https://github.com/langchain-ai/deepagents) is a strong agent harness, but it is context hungry.
It assumes a model with a window of 100K tokens or more.
Here is what I measured on `deepagents==0.7.39`:

| What | Cost |
|---|---|
| The 8 default tool schemas, sent before the user writes one word | about 2,480 tokens |
| The default summarization trigger, when the model has no profile | 170,000 tokens |
| The size at which a tool result moves out of the context to a file | 20,000 tokens |

On an 8K model, the tool schemas alone use about 30% of the window.
Summarization never starts, so the conversation overflows after a few steps.

Many small and local models have a window of 8K tokens or less.
I want an agent harness that works well on them.

My bet: a small context window can still give good results with a good harness and a good agent loop.
The model does not need 100K+ tokens.
It needs the right tokens.
This project is my attempt to prove that.

## How the 8K cap works

Every request to the model must fit in `CONTEXT_WINDOW` tokens (default 8,192), prompt plus output.

1. The app reserves `MAX_OUTPUT_TOKENS` (default 1,024) for the reply.
   The rest, 7,168 tokens, is the budget for the prompt.
2. The app estimates the tokens of each message as `ceil(utf8_bytes / 3) + 4`.
   English text averages about 4 bytes per token, so this estimate errs on the safe side.
3. The app sends the newest messages that fit the budget.
   It drops the oldest messages first.
   The window always starts with a user message.
4. If the new message alone is larger than the budget, the app returns HTTP 413 and saves nothing.
5. SQLite always keeps the full history.
   Only the copy that goes to the model is trimmed.

**Live result:** in a 15-turn test with long English messages, trimming started at turn 9.
The highest `prompt_tokens` that the API reported was 4,543, under the 7,168 budget.
Code, emoji, and CJK text can tokenize differently and are not tested yet.

## Quick start

Requirements:

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12+ for you)
- `make`
- An [OpenCode Go](https://opencode.ai/docs/en/go/) API key

Steps:

1. Copy the env template: `cp .env.example .env`
2. Set `OPENCODE_API_KEY` in `.env`.
3. Start the server: `make run`
4. Open http://127.0.0.1:8000

Run `make` to see every target.

| Command | What it does |
|---|---|
| `make` or `make help` | Lists the targets. |
| `make run` | Starts the server from `.env`. Override the address with `HOST` and `PORT`, for example `make run PORT=9000`. |
| `make logs` | Follows `logs/app.log`. |
| `make test` | Runs the backend tests (pytest) and the frontend tests (`bun test`), each in under 1 second. |

Requirements for `make test`: [bun](https://bun.sh), in addition to the list above.

## Configuration

All settings come from environment variables. `make run` reads them from `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `OPENCODE_API_KEY` | (required) | The API key for OpenCode Go. |
| `LLM_PROVIDER` | `opencode` | The model provider, by registry name. |
| `STORAGE_BACKEND` | `sqlite` | The history store, by registry name. |
| `OPENCODE_BASE_URL` | `https://opencode.ai/inference/go/openai/v1` | The OpenAI-compatible base URL. |
| `MODEL` | `deepseek-v4.1-flash` | The model ID. |
| `CONTEXT_WINDOW` | `8192` | The hard cap for one request: prompt plus output tokens. |
| `MAX_OUTPUT_TOKENS` | `1024` | The tokens reserved for the reply. |
| `DB_PATH` | `chat.db` | The SQLite file for the chat history. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |
| `LOG_FILE` | `logs/app.log` | The log file. It rotates at 5 MB and keeps 3 old files. |

## Logs

Each event is one line, in the terminal and in `LOG_FILE`.
Each line holds the time, the level, the event, and `key=value` fields.
All lines of one request share one `request_id`.
The web page shows that ID next to an error, so you can find the matching lines.

One real request, with the timestamps and some fields removed:

```
INFO  mini_deepagents.app request received request_id=f1da5b3acaca method=POST path=/api/sessions/1/messages
INFO  mini_deepagents.app context fitted request_id=f1da5b3acaca session_id=1 sent_messages=1 dropped_messages=0 estimated_tokens=19 budget_tokens=7168
INFO  mini_deepagents.llm model call succeeded request_id=f1da5b3acaca prompt_tokens=43 completion_tokens=31 duration_ms=2789
INFO  mini_deepagents.app exchange saved request_id=f1da5b3acaca session_id=1 reply_chars=13
INFO  mini_deepagents.app request completed request_id=f1da5b3acaca method=POST path=/api/sessions/1/messages status=200 duration_ms=2793
```

- `context fitted` shows the trimming decision for each request.
- `model call succeeded` shows the real token counts from the API.
- Errors appear as `ERROR model call failed`, `WARNING message rejected reason=too_long`, or `ERROR unhandled error` with a traceback.
- The logs never hold message text or the API key.

## API

| Method and path | Result |
|---|---|
| `GET /` | The chat page. |
| `GET /api/v1/sessions` | All sessions, newest first. |
| `POST /api/v1/sessions` | A new session (201). |
| `GET /api/v1/sessions/{id}/messages` | The full history of one session. |
| `POST /api/v1/sessions/{id}/messages` with `{"content": "..."}` | The reply as a stream: `200 text/event-stream` with `delta`, `done`, and `error` frames. Errors before the stream: 404 unknown session, 413 message too long, 422 invalid body. |

Every response carries an `X-Request-ID` header. `ARCHITECTURE.md` defines the SSE frames.

## Project layout

```
src/
  backend/mini_deepagents/   Python package, hexagonal design
    main.py                  composition root: wires every part
    chat/                    domain: types, ports, 8K budget, ChatService
    adapters/llm/            model providers (opencode), by registry name
    adapters/storage/        history stores (sqlite + migrations), by registry name
    adapters/http/           v1 routes, SSE encoder, request middleware
    observability/           key=value logs with a request ID
  frontend/                  vanilla JS, no build step
    index.html, css/app.css
    js/main.js, api.js, sse.js, state.js, ui/*.js
tests/
  backend/                   pytest: contract tests per port, service, HTTP
  frontend/                  bun test: the SSE parser
```

`ARCHITECTURE.md` explains each part and each seam for growth.
