.PHONY: install train test lint run docker-up docker-down clean

VENV = .venv
PYTHON = $(VENV)/bin/python
PYTEST = $(VENV)/bin/pytest
RUFF = $(VENV)/bin/ruff
UVICORN = $(VENV)/bin/uvicorn

install:
	uv venv $(VENV)
	uv pip install --python $(PYTHON) -e ".[dev]"

train:
	$(PYTHON) -m src.train_and_eval

test:
	$(PYTEST) -v

lint:
	$(RUFF) check .

run:
	$(UVICORN) src.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ src/**/__pycache__ tests/__pycache__
