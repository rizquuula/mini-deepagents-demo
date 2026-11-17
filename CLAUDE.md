# mini-deepagents

A minimal agent harness that works inside an 8K-token context window. Today it is a web chat with a hard 8K cap, SQLite history, and replies streamed over SSE. The backend is Python, managed with `uv`. The frontend is vanilla JavaScript.

## Why this project exists

- Deep Agents assumes a window of 100K+ tokens. Its 8 default tool schemas alone cost about 2,480 tokens, about 30% of an 8K window.
- Many small and local models have a window of 8K tokens or less. This project targets them.
- The owner's bet: a small context window can still give good results with a good harness and a good agent loop.
- This project exists to prove that bet. See `README.md` for the full story and the measured numbers.

## Design rules that follow from the motivation

- Every model request must fit in `CONTEXT_WINDOW` tokens, prompt plus output. Never break this cap.
- Treat each prompt token as a cost. Before you add prompt text, a system prompt, or a tool schema, count its tokens and state the count.
- Measure, do not guess. Check a change against the real `prompt_tokens` in the `model stream succeeded` log line.
- Build the simplest working version first. Add a feature only when the user asks for it.

## Architecture: modular from the start

Read `ARCHITECTURE.md` before you change the structure. It lists every seam and its growth path.

- Keep the two parts apart. `src/backend/` is the Python package. `src/frontend/` is the browser code. They share no code and talk only through the HTTP API under `/api/v1`.
- Stream model replies from the backend with SSE. The send route returns `text/event-stream` with `delta`, `done`, and `error` frames, as `ARCHITECTURE.md` defines. Errors before the stream starts are plain JSON (404, 413, 422).
- Keep the domain package `chat/` pure: no FastAPI, Starlette, httpx, sqlite3, Pydantic, or `logging` imports. `tests/backend/test_architecture.py` enforces this. Log in the adapters.
- Define each port in `chat/ports.py`, on the consumer side.
- Add a new LLM provider or storage backend as one new file in `adapters/llm/` or `adapters/storage/`, plus one registry entry. Never add an `if`/`elif` on a type name.
- Read configuration only in `main.py`, through `Settings.from_env()`. Pass the typed `Settings` to the parts that need it.
- Derive each domain error from `MiniDeepAgentsError` in `errors.py`. Translate vendor errors into domain errors inside the adapter.
- Version every contract: `/api/v1` for HTTP, `"v": 1` in each SSE frame, and `PRAGMA user_version` for SQLite. To change the schema, add a migration in `adapters/storage/migrations.py`. Never edit an old migration.
- Give every port a fake in `tests/backend/fakes.py`. Run each adapter against the shared contract test in `tests/backend/adapters/`.

## Frontend: vanilla JavaScript only

- Use plain ES modules (`<script type="module">`). Use no framework, no bundler, no build step, no npm packages, and no CDN.
- Give each module one job: `api.js` for HTTP, `sse.js` for SSE parsing, `state.js` for state, and one file per UI region in `js/ui/`. Only `main.js` wires the modules together.
- Insert server and model text with `textContent`. Never use `innerHTML`.
- Keep all styles in `css/app.css` and all markup in `index.html`.

## Commands

- Show the targets: `make` (the same as `make help`)
- Run the server: `make run` (reads `.env`; set `HOST` or `PORT` to override)
- Follow the logs: `make logs` (tails `logs/app.log`)
- Run all tests: `make test` (backend: `uv run pytest -q`; frontend: `bun test tests/frontend`)

## Tests: keep them lean and quick

- Keep the full suite under 5 seconds. A slow test is a bug: fix it or delete it.
- Make no real network call. Use `httpx.MockTransport` for the LLM adapter and the fakes in `tests/backend/fakes.py` for everything else.
- Use a file DB under `tmp_path` for SQLite. Never share a DB between tests.
- Write one test per behaviour. Name the test by that behaviour.
- Use `pytest.mark.parametrize` for input variants. Do not copy a test to change one value.
- Prefer small fakes over `unittest.mock`. Do not patch private functions.
- Test our code, not the libraries. Do not test FastAPI validation, Pydantic, or SQLite itself.
- Do not use `sleep`, real clocks, or timing-based asserts.
- Do not add hooks or flags to production code only for tests.
- Add a test only for a new behaviour or a fixed bug. Delete a test that checks the same thing as another test.
- Use pytest for the backend and `bun test` for the frontend. Both run with no extra packages. Ask before you add a test dependency.
- Drive async code with `asyncio.run(...)` or FastAPI's `TestClient`. Do not add `pytest-asyncio`.
- The live API check (real key, real model) is manual. Keep it out of the suite.
