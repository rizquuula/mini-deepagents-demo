# Architecture

mini-deepagents has two parts:

- `src/backend/`: a Python package, `mini_deepagents`, with a hexagonal design.
- `src/frontend/`: plain HTML, CSS, and vanilla JavaScript modules, with no build step.

The two parts share no code. The frontend talks to the backend only through the HTTP API under `/api/v1` and its SSE stream.

## Backend layers

```
src/backend/mini_deepagents/
  main.py              composition root: reads Settings, builds and wires every part
  config.py            Settings, built from environment variables
  errors.py            MiniDeepAgentsError and its subclasses
  chat/                the domain: no framework, vendor, or logging imports
    domain.py          Message, Session, Usage, TextDelta, StreamEnd, ReplyDone
    ports.py           ChatModel, ChatRepository, TokenEstimator (Protocols)
    context.py         ByteTokenEstimator, fit_messages (the 8K budget)
    service.py         ChatService: prepare one turn, stream it, save it
  adapters/            implementations of the ports, and the HTTP entry point
    llm/               provider registry + opencode.py (streaming client)
    storage/           backend registry + sqlite.py + migrations.py
    http/              api.py (v1 routes), sse.py (wire format), middleware.py
  observability/
    logging.py         key=value log lines with a request ID
```

Dependencies point inward:

- `adapters/` and `main.py` import `chat/`.
- `chat/` never imports `adapters/`, `main.py`, FastAPI, httpx, sqlite3, or `logging`. The test `tests/backend/test_architecture.py` enforces this rule.

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
 chat/service.py      stream_reply(): model.stream() -> TextDelta ... StreamEnd
    |                 on StreamEnd: repository.add_exchange(), yield ReplyDone
    v
 http/sse.py          delta / done / error frames, log "stream completed"
    v
 Browser (js/sse.js)  parse frames -> js/ui/messages.js appends the text
```

The service saves the exchange only after the model stream completes. An error, a client disconnect, or a cancellation saves nothing.

## Seams

A seam is a boundary where new behaviour arrives as a new file, with no change to existing code.

| Seam | Today | Growth path |
|---|---|---|
| `ChatModel` port + `adapters/llm` registry (`LLM_PROVIDER`) | `opencode` | A local model, such as Ollama or llama.cpp: add `adapters/llm/<name>.py` and one entry in `PROVIDERS`. |
| `ChatRepository` port + `adapters/storage` registry (`STORAGE_BACKEND`) | `sqlite` | Postgres or another store: add `adapters/storage/<name>.py` and one entry in `BACKENDS`. |
| `TokenEstimator` port | `ByteTokenEstimator` (bytes ÷ 3) | The real tokenizer of a model: add a class and pass it in `main.py`. |
| `ChatEvent` domain events → `http/sse.py` | SSE | Another transport, such as WebSocket or a CLI, consumes the same events. |
| `MiniDeepAgentsError` | 4 subclasses | Each new domain error derives from it. Adapters translate vendor errors into it. |

Each port has a fake in `tests/backend/fakes.py`. Each adapter runs the shared contract test in `tests/backend/adapters/`, next to the fake.

## Versioned contracts

| Contract | Version | Rule for a change |
|---|---|---|
| HTTP API | path prefix `/api/v1` | A breaking change goes under `/api/v2`. |
| SSE payload | field `"v": 1` in every frame | A breaking change sets `"v": 2`. The frontend checks the frames that it reads. |
| SQLite schema | `PRAGMA user_version` (now 1) | Add a new entry to `MIGRATIONS` in `adapters/storage/migrations.py`. Never edit an old entry. |

## SSE contract v1

`POST /api/v1/sessions/{id}/messages` returns `200 text/event-stream` with these frames:

```
event: delta
data: {"v":1,"text":"Hel"}

event: done
data: {"v":1,"reply_chars":13}

event: error
data: {"v":1,"code":"model_failed","message":"model request failed","request_id":"3f2a9c1b7d0e"}
```

The stream ends after one `done` or one `error` frame.

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
