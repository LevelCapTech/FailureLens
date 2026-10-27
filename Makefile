# FailureLens developer tasks.
#
# These targets wrap the real tools configured in pyproject.toml so that local
# development matches CI exactly (see .github/workflows/).

PY ?= python
PKG := packages/python

.PHONY: help install lint format format-check typecheck test coverage build check demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev extras (editable)
	$(PY) -m pip install -e "$(PKG)[dev]"

lint: ## Run ruff lint checks
	ruff check .

format: ## Auto-format the codebase with ruff
	ruff format .

format-check: ## Check formatting without writing changes
	ruff format --check .

typecheck: ## Run mypy (strict)
	mypy

test: ## Run the test suite
	$(PY) -m pytest

coverage: ## Run tests with coverage reporting
	$(PY) -m pytest --cov=failurelens --cov-report=term-missing

build: ## Build the sdist and wheel
	$(PY) -m build $(PKG)

check: lint format-check typecheck test ## Run all quality gates (mirrors CI)

demo: ## Run the baseline on the bundled example fixture
	failurelens diagnose \
		--repo examples/fixtures/sample_project \
		--log examples/fixtures/sample_project/failure.log \
		--top 5

clean: ## Remove build artifacts and caches
	rm -rf dist build $(PKG)/dist $(PKG)/build
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache
