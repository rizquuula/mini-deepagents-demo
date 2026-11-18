# mini-deepagents-demo

The reference app of the [mini-deepagents](https://github.com/rizquuula/mini-deepagents) framework. The framework is a minimal agent harness that works inside an 8K-token context window. This demo shows how to implement the framework ports with real adapters. Today it is a web chat with a hard 8K cap, SQLite history, and replies streamed over SSE. The backend is Python, managed with `uv`. The frontend is vanilla JavaScript.

This repository is https://github.com/rizquuula/mini-deepagents-demo. The framework repository now holds the domain.

## Why this project exists

- Deep Agents assumes a window of 100K+ tokens. Its 8 default tool schemas alone cost about 2,480 tokens, about 30% of an 8K window.
- Many small and local models have a window of 8K tokens or less. This project targets them.
- The owner's bet: a small context window can still give good results with a good harness and a good agent loop.
- The framework exists to prove that bet. This demo runs it against a real model. See `README.md` for the full story and the measured numbers.

## Design rules that follow from the motivation

- Every model request must fit in `CONTEXT_WINDOW` tokens, prompt plus output. Never break this cap.
- Treat each prompt token as a cost. Before you add prompt text, a system prompt, or a tool schema, count its tokens and state the count.
- Measure, do not guess. Check a change against the real `prompt_tokens` in the `model stream succeeded` log line.
- Build the simplest working version first. Add a feature only when the user asks for it.

## Architecture: modular from the start

Read `ARCHITECTURE.md` before you change the structure. It lists every seam and its growth path.

- Keep the two parts apart. `src/backend/` is the Python package. `src/frontend/` is the browser code. They share no code and talk only through the HTTP API under `/api/v1`.
- Stream model replies from the backend with SSE. The send route returns `text/event-stream` with `delta`, `done`, and `error` frames, as `ARCHITECTURE.md` defines. Errors before the stream starts are plain JSON (404, 413, 422).
- The domain core lives in the framework package `mini_deepagents`: `ChatService`, the ports, the context fitting, and the domain errors. Do not copy it back into this repository.
- To change the domain, change the framework repository and tag a new version. Then bump the tag in `pyproject.toml` with `uv add`.
- Implement the framework ports in the demo adapters. Log in the adapters.
- Import the framework only through its top-level API: `from mini_deepagents import ...`. Tests may also import `mini_deepagents.testing`. `tests/backend/test_architecture.py` enforces both rules.
- Add a new LLM provider or storage backend as one new file in `adapters/llm/` or `adapters/storage/`, plus one registry entry. Never add an `if`/`elif` on a type name.
- Read configuration only in `main.py`, through `Settings.from_env()`. Pass the typed `Settings` to the parts that need it.
- Keep `ConfigError` in the demo, in `errors.py`. It derives from the framework's `MiniDeepAgentsError`.
- Translate vendor errors into the framework's domain errors inside the adapter.
- Version every contract: `/api/v1` for HTTP, `"v": 1` in each SSE frame, and `PRAGMA user_version` for SQLite. To change the schema, add a migration in `adapters/storage/migrations.py`. Never edit an old migration.
- Use the fakes from `mini_deepagents.testing` in tests. Run each adapter against the shared contract test in `tests/backend/adapters/`.

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
- Make no real network call. Use `httpx.MockTransport` for the LLM adapter and the fakes in `mini_deepagents.testing` for everything else.
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
