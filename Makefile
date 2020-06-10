LINT_SOURCES_DIRS = autoboosting tests

.EXPORT_ALL_VARIABLES:
POETRY ?= $(HOME)/.poetry/bin/poetry

.PHONY: install-poetry
install-poetry:
	@curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

.PHONY: configure-poetry
configure-poetry:
	@$(POETRY) config virtualenvs.path venv

.PHONY: install-deps
install-deps:
	@$(POETRY) install -vvv

.PHONY: install-deps-no-dev
install-deps-no-dev:
	@$(POETRY) install -vvv --no-dev

.PHONY: install
install: install-poetry install-deps

.PHONY: install-configure-poetry
install-configure-poetry: install-poetry configure-poetry install-deps

.PHONY: install-configure-poetry-no-dev
install-configure-poetry-no-dev: install-poetry configure-poetry install-deps-no-dev

.PHONY: lint-black
lint-black:
	@echo "\033[92m< linting using black...\033[0m"
	@$(POETRY) run black --check --diff $(LINT_SOURCES_DIRS)
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint-flake8
lint-flake8:
	@echo "\033[92m< linting using flake8...\033[0m"
	@$(POETRY) run flake8 $(LINT_SOURCES_DIRS)
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint-isort
lint-isort:
	@echo "\033[92m< linting using isort...\033[0m"
	@$(POETRY) run isort --check-only --diff --recursive $(LINT_SOURCES_DIRS)
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint-mypy
lint-mypy:
	@echo "\033[92m< linting using mypy...\033[0m"
	@$(POETRY) run mypy $(LINT_SOURCES_DIRS)
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: lint
lint: lint-black lint-flake8 lint-isort lint-mypy

.PHONY: fmt-black
fmt-black:
	@echo "\033[92m< linting using black...\033[0m"
	@$(POETRY) run black $(LINT_SOURCES_DIRS)
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: fmt-isort
fmt-isort:
	@echo "\033[92m< linting using isort...\033[0m"
	@$(POETRY) run isort --recursive $(LINT_SOURCES_DIRS)
	@echo "\033[92m> done\033[0m"
	@echo

.PHONY: fmt
fmt: fmt-black fmt-isort

.PHONY: test
test:
	@echo "\033[92m< running tests...\033[0m"
	@$(POETRY) run pytest -vv --strict --cov-config .coveragerc --cov autoboosting $(opts) $(call tests,.)