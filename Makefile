.PHONY: test check build
test:
	python -m pytest -q
check:
	python -m ruff check src tests scripts
	python -m ruff format --check src tests scripts
build:
	python -m build
