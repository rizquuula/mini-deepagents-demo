# mini-deepagents

A minimal agent harness that works inside an 8K-token context window. Today it is a web chat with a hard 8K cap and SQLite history. Python, managed with `uv`.

## Why this project exists

- Deep Agents assumes a window of 100K+ tokens. Its 8 default tool schemas alone cost about 2,480 tokens, about 30% of an 8K window.
- Many small and local models have a window of 8K tokens or less. This project targets them.
- The owner's bet: a small context window can still give good results with a good harness and a good agent loop.
- This project exists to prove that bet. See `README.md` for the full story and the measured numbers.

## Design rules that follow from the motivation

- Every model request must fit in `CONTEXT_WINDOW` tokens, prompt plus output. Never break this cap.
- Treat each prompt token as a cost. Before you add prompt text, a system prompt, or a tool schema, count its tokens and state the count.
- Measure, do not guess. Check a change against the real `prompt_tokens` in the `model call succeeded` log line.
- Build the simplest working version first. Add a part only when the user asks for it.

## Commands

- Show the targets: `make` (the same as `make help`)
- Run the server: `make run` (reads `.env`; set `HOST` or `PORT` to override)
- Follow the logs: `make logs` (tails `logs/app.log`)
- Run the tests: `uv run pytest -q`

## Tests: keep them lean and quick

- Keep the full suite under 5 seconds. A slow test is a bug: fix it or delete it.
- Make no real network call. Use `httpx.MockTransport` for `llm.py` and a fake `ChatModel` for the app.
- Use a file DB under `tmp_path` for SQLite. Never share a DB between tests.
- Write one test per behaviour. Name the test by that behaviour.
- Use `pytest.mark.parametrize` for input variants. Do not copy a test to change one value.
- Prefer small fakes over `unittest.mock`. Do not patch private functions.
- Test our code, not the libraries. Do not test FastAPI validation, Pydantic, or SQLite itself.
- Do not use `sleep`, real clocks, or timing-based asserts.
- Do not add hooks or flags to production code only for tests.
- Add a test only for a new behaviour or a fixed bug. Delete a test that checks the same thing as another test.
- Use pytest only. Ask before you add a test dependency.
- The live API check (real key, real model) is manual. Keep it out of the suite.
