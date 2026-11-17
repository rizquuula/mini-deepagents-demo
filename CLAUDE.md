# mini-deepagents

A minimal web chat with an 8K-token context cap. Python, managed with `uv`.

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
