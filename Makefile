# ============================================================================
# Ophelian — cleanup Makefile
#
# Targets focused *exclusively* on removing garbage and caches from the repo.
# Each target is atomic (`make clean-test`, `make clean-build`, ...) and
# `clean` groups the set you use day to day. `distclean` removes EVERYTHING.
#
# Conventions:
#   - Nothing touches .git/, .github/, attached_assets/ or the source code.
#   - The `find` calls use -prune so they do not descend into installed
#     packages (numpy, sklearn, ...) nor into .git.
#   - Recipes use TAB indentation (Make requirement); variable continuations
#     and .PHONY use spaces.
#   - We use `-exec rm -f {} +` instead of `-delete` because `-delete`
#     implicitly enables `-depth`, which breaks `-prune`.
# ============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Directories that find must ALWAYS ignore. If any of these were missing,
# find would descend into installed packages and wipe their __pycache__.
#   .git           -> git state
#   .venv          -> standard virtualenv (uv / venv)
#   .pythonlibs    -> site-packages used by the Replit sandbox
#   node_modules   -> in case a JS sub-project lives alongside
#   .cache         -> system-level caches (includes pre-commit envs)
#   .uv-cache      -> local uv cache (covered by the opt-in clean-uv)
#   .local         -> internal agent / harness metadata
#   attached_assets-> user files, do not touch
PRUNE := \
    -path ./.git -o \
    -path ./.venv -o \
    -path ./.pythonlibs -o \
    -path ./node_modules -o \
    -path ./.cache -o \
    -path ./.uv-cache -o \
    -path ./.local -o \
    -path ./attached_assets

.PHONY: help clean clean-pyc clean-test clean-lint clean-build clean-docs \
        clean-uv clean-precommit clean-venv distclean

help:  ## Show this help.
	@echo "Ophelian — repo cleanup"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Day-to-day target. Does NOT touch venv or the uv cache.
# ---------------------------------------------------------------------------
clean: clean-pyc clean-test clean-lint clean-build clean-docs  ## Standard cleanup (pyc + tests + lint + build + docs).
	@echo "✓ Repo clean (venv and uv caches preserved)."

# ---------------------------------------------------------------------------
# Python bytecode — always safe to remove.
# ---------------------------------------------------------------------------
clean-pyc:  ## Remove __pycache__/, *.pyc, *.pyo, *.pyd recursively.
	@echo "→ Removing Python bytecode..."
	@find . \( $(PRUNE) \) -prune -o -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@find . \( $(PRUNE) \) -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.pyd' \) -exec rm -f {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# pytest / coverage caches and reports.
# ---------------------------------------------------------------------------
clean-test:  ## Remove .pytest_cache, .coverage, coverage.xml, htmlcov/.
	@echo "→ Removing test and coverage artifacts..."
	@find . \( $(PRUNE) \) -prune -o -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov coverage.xml .coverage .coverage.*

# ---------------------------------------------------------------------------
# Linter / typechecker caches (ruff, mypy).
# ---------------------------------------------------------------------------
clean-lint:  ## Remove .ruff_cache/ and .mypy_cache/.
	@echo "→ Removing ruff and mypy caches..."
	@find . \( $(PRUNE) \) -prune -o -type d \( -name '.ruff_cache' -o -name '.mypy_cache' \) -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Packaging artifacts.
# ---------------------------------------------------------------------------
clean-build:  ## Remove dist/, build/, *.egg-info/.
	@echo "→ Removing build artifacts..."
	@rm -rf dist build
	@find . \( $(PRUNE) \) -prune -o -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Static mkdocs output.
# ---------------------------------------------------------------------------
clean-docs:  ## Remove site/ (mkdocs build output).
	@echo "→ Removing rendered docs..."
	@rm -rf site

# ---------------------------------------------------------------------------
# "Expensive" caches — opt-in. Do not include in `clean` unless you know
# you want to pay the cost of recreating them.
# ---------------------------------------------------------------------------
clean-precommit:  ## Remove .cache/pre-commit/ (cached hook environments).
	@echo "→ Removing cached pre-commit environments..."
	@rm -rf .cache/pre-commit

clean-uv:  ## Remove the local uv cache (./.uv-cache if present).
	@echo "→ Removing local uv cache..."
	@rm -rf .uv-cache

clean-venv:  ## Remove the .venv/ virtualenv (recreate it with `uv sync`).
	@echo "→ Removing .venv/..."
	@rm -rf .venv

# ---------------------------------------------------------------------------
# Full wipe. After this, rebuild everything with `uv sync --frozen --extra dev`.
# ---------------------------------------------------------------------------
distclean: clean clean-precommit clean-uv clean-venv  ## Full wipe: clean + precommit + uv + venv.
	@echo "✓ Full wipe done. Rebuild with: uv sync --frozen --extra dev"
