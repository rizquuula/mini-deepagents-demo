.DEFAULT_GOAL := help

HOST ?= 127.0.0.1
PORT ?= 8000
ENV_FILE ?= .env
LOG_FILE ?= logs/app.log

.PHONY: help run logs test

help: ## Show the available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-6s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

run: ## Start the web chat server on HOST:PORT (reads .env)
	@test -f $(ENV_FILE) || { echo "$(ENV_FILE) is missing. Run: cp .env.example .env, then set OPENCODE_API_KEY."; exit 1; }
	uv run --env-file $(ENV_FILE) uvicorn mini_deepagents_demo.main:create_app --factory --host $(HOST) --port $(PORT)

logs: ## Follow the log file (LOG_FILE=logs/app.log)
	@mkdir -p $(dir $(LOG_FILE))
	@touch $(LOG_FILE)
	tail -n 50 -F $(LOG_FILE)

test: ## Run the backend (pytest) and frontend (bun test) suites
	uv run pytest -q
	bun test tests/frontend
