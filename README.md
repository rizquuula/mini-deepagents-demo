# mini-deepagents-demo

The reference app of the [mini-deepagents](https://github.com/rizquuula/mini-deepagents) framework.
The framework is a minimal agent harness that works inside an **8K-token context window**.
This demo shows how to implement the framework ports with real adapters.
Today it is a small web chat with a hard 8K cap, SQLite history, one tool, and replies that stream to the browser over SSE.

The framework repository now holds the domain: `AgentService` (the tool loop), the ports, the context fitting, and the domain errors.
This repository holds the adapters, the HTTP API, and the browser frontend.

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
The framework is my attempt to prove that.
This demo runs the framework against a real model.

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

## Tools

The model can call tools.
Each turn runs the agent loop of the framework, `AgentService`:

- The loop makes at most 4 model calls per turn.
- The last call sends no tools, so the model must answer in text.
- The loop cuts each tool result to 512 tokens.
- The loop saves only the user message and the final reply. It never saves the tool steps.

### The `calculate` tool

`calculate` evaluates one arithmetic expression with `+ - * / // % **` and parentheses.
It parses the expression with Python's `ast` module and walks the tree itself.
It never calls `eval`.
It has these limits:

- The expression has 200 characters or fewer.
- An exponent is 100 or less.
- An integer has 4,000 bits or fewer.

### Tool settings

- `ENABLED_TOOLS` is a comma list of tool names. The default is `calculate`. An empty value means no tools.
- `TOOL_OVERHEAD_TOKENS` is the fixed cost that the provider adds when a request carries tools. The default is 256.

### What tools cost

A request with tools costs more prompt tokens.
Measured on `deepseek-v4.1-flash` through OpenCode Go:

| Tools in the request | Prompt tokens |
|---|---|
| None | 44 |
| 1 small tool | 297 |
| 2 small tools | 343 |
| 3 small tools | 389 |

The fixed overhead is about 207 tokens.
Each small tool adds about 46 tokens.
To measure the cost yourself, send the same request with and without tools, then subtract.

The `calculate` schema is 242 bytes.
The framework estimates it at 85 tokens.
The app reserves 256 + 85 = 341 tokens of the 7,168 input budget for tools.
That leaves 6,827 tokens for the history.

### Known trade-off: tool steps are not saved

The loop does not save the tool steps.
This keeps later turns small.
It has two costs:

- After a reload, the tool note in the reply bubble is gone.
- In a later turn, the model sees only its earlier final answer, without the tool evidence.

In one live run, the model then said that it had answered "without actually running the tool".
A follow-up may save a compact tool trace.

### Add a tool

1. Write a class with a `spec: ToolSpec` and `async run(arguments) -> str`. Put it in one new file in `tools/`.
2. Add one entry for it in `TOOLS` in `tools/__init__.py`.
3. Add its name to `ENABLED_TOOLS`.
4. Count the tokens of its schema. Compare `prompt_tokens` with and without the tool.
5. Add a test in `tests/backend/tools/`.

## Quick start

Requirements:

- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12+ for you)
- `make`
- An [OpenCode Go](https://opencode.ai/docs/en/go/) API key

Steps:

1. Clone the repository: `git clone https://github.com/rizquuula/mini-deepagents-demo.git`
2. Copy the env template: `cp .env.example .env`
3. Set `OPENCODE_API_KEY` in `.env`.
4. Start the server: `make run`. The first run also installs the framework package from its Git tag.
5. Open http://127.0.0.1:8000

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
| `ENABLED_TOOLS` | `calculate` | The tools the model may call: a comma list of names. An empty value means no tools. |
| `TOOL_OVERHEAD_TOKENS` | `256` | The tokens reserved for the fixed cost of a request with tools. |
| `DB_PATH` | `chat.db` | The SQLite file for the chat history. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |
| `LOG_FILE` | `logs/app.log` | The log file. It rotates at 5 MB and keeps 3 old files. |

## Logs

Each event is one line, in the terminal and in `LOG_FILE`.
Each line holds the time, the level, the event, and `key=value` fields.
All lines of one request share one `request_id`.
The web page shows that ID next to an error, so you can find the matching lines.

One real request that called `calculate`, with the timestamps removed.
In the logger names, `...` stands for `mini_deepagents_demo.adapters`.

```
INFO  ...http.api         context fitted         request_id=1b876c3c55fa session_id=1 history_messages=0 sent_messages=1 dropped_messages=0 estimated_tokens=19 budget_tokens=6827
INFO  ...llm.opencode     model stream succeeded request_id=1b876c3c55fa model=deepseek-v4.1-flash prompt_tokens=310 completion_tokens=61 reasoning_tokens=18 total_tokens=371 first_delta_ms=1722 duration_ms=1929 tool_calls=1
INFO  ...http.sse         tool finished          request_id=1b876c3c55fa session_id=1 tool_name=calculate ok=True result_chars=7 duration_ms=0
INFO  ...llm.opencode     model stream succeeded request_id=1b876c3c55fa model=deepseek-v4.1-flash prompt_tokens=367 completion_tokens=16 reasoning_tokens=0 total_tokens=383 first_delta_ms=1457 duration_ms=1532 tool_calls=0
INFO  ...http.sse         stream completed       request_id=1b876c3c55fa session_id=1 reply_chars=27 steps=2 peak_estimated_tokens=402 prompt_tokens=367 completion_tokens=16 total_tokens=383 first_delta_ms=3387 duration_ms=3469
```

The reply was "1234 × 5678 = **7,006,652**".

- `context fitted` shows the trimming decision for each request. With `calculate` on, the budget for the history is 6,827 tokens.
- `model stream succeeded` appears once for each model call. It shows the real token counts from the API, including the thinking tokens (`reasoning_tokens`). `tool_calls` counts the tool calls that the model asked for.
- `tool finished` appears once for each tool call. It shows the tool name, the outcome, and the length of the result.
- `stream completed` closes the turn. `steps` counts the model calls. `peak_estimated_tokens` is the largest estimate of one model request.
- Each estimate was at or above the real count. Step 1 had an estimate of 360 and a real count of 310. The peak estimate was 402, and the real count was 367.
- `first_delta_ms` is the time until the first reply text or tool call. In `stream completed`, it counts across all steps of the turn.
- `app started` lists the settings, including `enabled_tools` and `tool_overhead_tokens`.
- Errors appear as `ERROR stream failed`, `WARNING message rejected reason=too_long`, or `ERROR unhandled error` with a traceback.
- A closed browser tab shows as `stream cancelled`. Nothing is saved for that turn.
- The logs never hold message text, tool arguments, tool results, or the API key.

## API

| Method and path | Result |
|---|---|
| `GET /` | The chat page. |
| `GET /api/v1/sessions` | All sessions, newest first. |
| `POST /api/v1/sessions` | A new session (201). |
| `GET /api/v1/sessions/{id}/messages` | The full history of one session. |
| `POST /api/v1/sessions/{id}/messages` with `{"content": "..."}` | The reply as a stream: `200 text/event-stream` with `tool_call`, `tool_result`, `delta`, `done`, and `error` frames. Errors before the stream: 404 unknown session, 413 message too long, 422 invalid body. |

Every response carries an `X-Request-ID` header. `ARCHITECTURE.md` defines the SSE frames.

The tool frames carry only the call ID, the tool name, and the outcome:

```
event: tool_call
data: {"v":1,"call_id":"…","name":"calculate"}

event: tool_result
data: {"v":1,"call_id":"…","ok":true}
```

The frames never carry tool arguments or tool results.
The reply bubble shows "Using calculate…" for a `tool_call` frame.
A `tool_result` frame changes the note to "Used calculate" or "calculate failed".

An `error` frame with the code `context_full` has the message "the conversation is too long for the context window".
It means that the tool steps outgrew the budget in the middle of the stream.

## Project layout

```
src/
  backend/mini_deepagents_demo/   Python package, hexagonal design
    main.py                       composition root: wires every part
    config.py                     Settings, built from environment variables
    errors.py                     ConfigError, the one demo-only error
    tools/                        tool registry (TOOLS, create_tools) + calculate.py
    adapters/llm/                 model providers (opencode), by registry name
    adapters/storage/             history stores (sqlite + migrations), by registry name
    adapters/http/                v1 routes, SSE encoder, request middleware
    observability/                key=value logs with a request ID
  frontend/                       vanilla JS, no build step
    index.html, css/app.css
    js/main.js, api.js, sse.js, state.js, ui/*.js
tests/
  backend/                        pytest: contract tests per port, HTTP, the framework seam
  frontend/                       bun test: the SSE parser
```

The domain (types, ports, the 8K budget, `AgentService`) comes from the `mini_deepagents` package.
`pyproject.toml` pins that package to a Git tag of the [framework repository](https://github.com/rizquuula/mini-deepagents).
The demo uses tag v0.2.0.

`ARCHITECTURE.md` explains each part and each seam for growth.

## License

MIT. See `LICENSE`.
