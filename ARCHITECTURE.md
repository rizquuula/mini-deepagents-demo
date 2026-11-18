# Architecture

mini-deepagents-demo is the reference app of the [mini-deepagents](https://github.com/rizquuula/mini-deepagents) framework. It has two parts:

- `src/backend/`: a Python package, `mini_deepagents_demo`, with a hexagonal design.
- `src/frontend/`: plain HTML, CSS, and vanilla JavaScript modules, with no build step.

The two parts share no code. The frontend talks to the backend only through the HTTP API under `/api/v1` and its SSE stream.

The domain core is not in this repository. It comes from the framework package `mini_deepagents`. `pyproject.toml` pins that package to a Git tag of the framework repository.

## Backend layers

```
mini_deepagents (framework package, installed from a Git tag)
  AgentService         prepare one turn, run the tool loop, stream it, save it
  ChatModel, ChatRepository, TokenEstimator, Tool   the ports (Protocols)
  ToolSpec, ToolSet, ToolCall                  tool schemas and calls
  ToolCallRequested, ToolStarted, ToolFinished the tool events
  ByteTokenEstimator, fit_messages            the 8K budget
  Message, Session, Usage, TextDelta, StreamEnd, ReplyDone
  MiniDeepAgentsError and the domain errors
  testing              FakeChatModel, InMemoryChatRepository

src/backend/mini_deepagents_demo/
  main.py              composition root: reads Settings, builds and wires every part
  config.py            Settings, built from environment variables
  errors.py            ConfigError, the one demo-only error
  tools/               tool registry (TOOLS, create_tools) + calculate.py
  adapters/            implementations of the framework ports, and the HTTP entry point
    llm/               provider registry + opencode.py (streaming client)
    storage/           backend registry + sqlite.py + migrations.py
    http/              api.py (v1 routes), sse.py (wire format), middleware.py
  observability/
    logging.py         key=value log lines with a request ID
```

Dependencies point inward:

- `adapters/` and `main.py` import the framework.
- The framework never imports the demo.
- The demo imports the framework only through its top-level module, `mini_deepagents`. Tests may also import `mini_deepagents.testing`.
- The test `tests/backend/test_architecture.py` enforces the import rule. It also checks that the demo has no `chat/` package of its own.

To change the domain, change the framework repository and tag a new version. Then bump the tag in `pyproject.toml` with `uv add`.

## One chat turn

```
 Browser (js/api.js)
    | POST /api/v1/sessions/{id}/messages {"content": "..."}
    v
 http/middleware.py   set request_id, log "request received"
    v
 http/api.py          service.prepare()  -> 404 / 413 as JSON, before any stream
    |                 log "context fitted"
    v
 AgentService         [framework] stream_reply(): the tool loop, at most 4 model calls
    |                 each call: model.stream(tools) -> TextDelta ... ToolCallRequested ... StreamEnd
    |                 each tool call: yield ToolStarted, tool.run(), yield ToolFinished,
    |                   cut the result to 512 tokens, call the model again
    |                 the last call sends no tools, so the model answers in text
    |                 on the final StreamEnd: save the user message and the final reply,
    |                   yield ReplyDone
    v
 http/sse.py          tool_call / tool_result / delta / done / error frames
    |                 log "tool finished" per tool call, log "stream completed"
    v
 Browser (js/sse.js)  parse frames -> js/ui/messages.js shows the tool note and appends the text
```

The service saves the exchange only after the final model stream completes. An error, a client disconnect, or a cancellation saves nothing.

The service never saves the tool steps. A later turn sees only the user message and the final reply. This keeps the history small, but the model loses the tool evidence.

`opencode.py` sends the enabled tool schemas in each request, except the last one. It joins the streamed tool-call pieces into `ToolCall` objects and logs `tool_calls` in "model stream succeeded".

## Seams

A seam is a boundary where new behaviour arrives as a new file, with no change to existing code.
The framework defines each port. The registries and the adapters stay in the demo.

| Seam | Today | Growth path |
|---|---|---|
| Framework port `ChatModel` + demo `adapters/llm` registry (`LLM_PROVIDER`) | `opencode` | A local model, such as Ollama or llama.cpp: add `adapters/llm/<name>.py` and one entry in `PROVIDERS`. |
| Framework port `ChatRepository` + demo `adapters/storage` registry (`STORAGE_BACKEND`) | `sqlite` | Postgres or another store: add `adapters/storage/<name>.py` and one entry in `BACKENDS`. |
| Framework port `Tool` + demo `tools/` registry (`ENABLED_TOOLS`) | `calculate` | A new tool: add one file in `tools/` and one entry in `TOOLS`, behind the framework `Tool` port. |
| Framework port `TokenEstimator` | `ByteTokenEstimator` from the framework (bytes ÷ 3) | The real tokenizer of a model: add a class in the demo and pass it in `main.py`. |
| Framework events `ChatEvent` → demo `http/sse.py` | SSE | Another transport, such as WebSocket or a CLI, consumes the same events. |
| Framework base error `MiniDeepAgentsError` | 3 framework subclasses + `ConfigError` in the demo | A new domain error goes into the framework. A demo-only error derives from `MiniDeepAgentsError` in `errors.py`. Adapters translate vendor errors into domain errors. |

The framework ships the fakes `FakeChatModel` and `InMemoryChatRepository` in `mini_deepagents.testing`. Each adapter runs the shared contract test in `tests/backend/adapters/`, next to the fake.

## Versioned contracts

| Contract | Version | Rule for a change |
|---|---|---|
| HTTP API | path prefix `/api/v1` | A breaking change goes under `/api/v2`. |
| SSE payload | field `"v": 1` in every frame | A breaking change sets `"v": 2`. The frontend checks the frames that it reads. |
| SQLite schema | `PRAGMA user_version` (now 1) | Add a new entry to `MIGRATIONS` in `adapters/storage/migrations.py`. Never edit an old entry. |

## SSE contract v1

`POST /api/v1/sessions/{id}/messages` returns `200 text/event-stream` with these frames:

```
event: tool_call
data: {"v":1,"call_id":"…","name":"calculate"}

event: tool_result
data: {"v":1,"call_id":"…","ok":true}

event: delta
data: {"v":1,"text":"Hel"}

event: done
data: {"v":1,"reply_chars":13}

event: error
data: {"v":1,"code":"model_failed","message":"model request failed","request_id":"3f2a9c1b7d0e"}
```

The stream ends after one `done` or one `error` frame.

The `tool_call` and `tool_result` frames are additive. A v1 client that ignores unknown events still works. The frames never carry tool arguments or tool results.

The `error` frame has one of these codes:

- `context_full`: "the conversation is too long for the context window". The tool steps outgrew the budget in the middle of the stream.
- `model_failed`: "model request failed".
- `internal_error`: "internal server error".

Errors that happen before the stream starts come back as plain JSON:

- 404: unknown session
- 413: message too long
- 422: invalid body

## Frontend modules

```
src/frontend/
  index.html          markup only
  css/app.css         all styles
  js/main.js          boot: wires the store, the API, and the UI modules
  js/api.js           REST calls and streamMessage() over fetch
  js/sse.js           pure SSE frame parser (tested with bun test)
  js/state.js         the one app-state store, with subscribe()
  js/ui/sessions.js   the session list
  js/ui/messages.js   the message list and the streaming bubble
  js/ui/composer.js   the textarea and the Send button
```

`main.js` is the only module that knows all the others. Each UI module owns one region of the DOM. All server and model text goes into the DOM through `textContent`.
