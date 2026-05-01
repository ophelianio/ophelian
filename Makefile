# Variables
SHELL := /usr/bin/env bash
PYTHON := python3
PYTHONPATH := `pwd`

# Docker variables
DOCKER_USERNAME := luisfalva
IMAGE := ophelian
VERSION := latest

# Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org | $(PYTHON) -
	@export PATH="$${HOME}/.local/bin:$$PATH"
	@printf "[Makefile] - poetry download from 'https://install.python-poetry.org' complete.\n\n"

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org | $(PYTHON) - --uninstall || echo "poetry is not installed."

# Installation
.PHONY: install
install:
	pip install --upgrade pip
	poetry config warnings.export false
	poetry lock -n && poetry export --without-hashes > requirements.txt || { echo "Failed to export requirements.txt"; exit 1; }
	poetry install --no-root -n
	@printf "[Makefile] - poetry installation and setup complete.\n\n"

.PHONY: update
update:
	poetry update
	@printf "[Makefile] - poetry updated.\n\n"

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install
	@printf "[Makefile] - pre-commit install complete.\n\n"

.PHONY: pre-commit-update
pre-commit-update:
	poetry run pre-commit autoupdate
	@printf "[Makefile] - pre-commit autoupdate complete.\n\n"

# Project setup
.PHONY: init
init: poetry-download install pre-commit-install

# Formatters
.PHONY: codestyle
codestyle:
	poetry run pyupgrade --exit-zero-even-if-changed --py38-plus **/*.py
	poetry run isort --settings-path pyproject.toml src
	poetry run black --config pyproject.toml src

.PHONY: check-codestyle
check-codestyle:
	poetry run isort --diff --check-only --settings-path pyproject.toml src
	poetry run black --diff --check --config pyproject.toml src
	poetry run darglint --verbosity 2 src tests

.PHONY: formatting
formatting: codestyle check-codestyle

# Linting
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) poetry run pytest -c pyproject.toml --cov-report=html --cov=ophelian tests/
	poetry run coverage-badge -o assets/images/coverage.svg -f

.PHONY: check
check:
	poetry check
	@printf "[Makefile] - poetry check complete.\n\n"

.PHONY: check-safety
check-safety:
	-poetry run safety check --full-report
	@printf "[Makefile] - safety check complete.\n\n"

.PHONY: check-bandit
check-bandit:
	poetry run bandit -vv -ll --recursive src
	@printf "[Makefile] - bandit check complete.\n\n"

.PHONY: env-check
env-check: check check-bandit check-safety

.PHONY: update-dev-deps
update-dev-deps:
	poetry add --group dev_latest darglint@latest "isort[colors]@latest" pydocstyle@latest pylint@latest pytest@latest coverage@latest coverage-badge@latest pytest-html@latest pytest-cov@latest
	poetry add --group dev_latest --allow-prereleases black@latest

# Docker
.PHONY: docker-build
docker-build:
	./build-push.sh $(IMAGE) $(VERSION)

# Example: make docker-remove VERSION=latest
# Example: make docker-remove IMAGE=some_name VERSION=0.1.0
.PHONY: docker-remove
docker-remove:
	@echo "Removing docker $(DOCKER_USERNAME)/$(IMAGE):$(VERSION) ..."
	docker rmi -f $(DOCKER_USERNAME)/$(IMAGE):$(VERSION)

# Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove

# Old targets to maintain backward compatibility
.PHONY: clean
clean: cleanup build-remove ## remove all build, test, coverage and Python artifacts

.PHONY: clean-build
clean-build: build-remove ## remove build artifacts

.PHONY: clean-pyc
clean-pyc: pycache-remove ## remove Python file artifacts

.PHONY: clean-test
clean-test: pytestcache-remove ## remove test and coverage artifacts
