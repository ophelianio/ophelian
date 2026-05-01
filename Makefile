# ============================================================================
# Ophelian — Makefile de limpieza
#
# Targets enfocados *exclusivamente* en quitar basura y caches del repo.
# Cada target es atómico (`make clean-test`, `make clean-build`, ...) y
# `clean` agrupa el set que usás día a día. `distclean` borra TODO.
#
# Convenciones:
#   - Nada toca .git/, .github/, attached_assets/ ni el código fuente.
#   - Los `find` usan -prune para no descender dentro de paquetes
#     instalados (numpy, sklearn, ...) ni dentro de .git.
#   - Las recetas usan TAB (requisito de Make); las continuaciones de
#     variables y .PHONY usan espacios.
#   - Para borrar archivos usamos `-exec rm -f {} +` en vez de `-delete`
#     porque `-delete` activa `-depth` implícito, que rompe `-prune`.
# ============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Directorios que find debe ignorar SIEMPRE. Si faltara alguno, el find
# descendería dentro de paquetes instalados y borraría su __pycache__.
#   .git           -> estado de git
#   .venv          -> entorno virtual estándar (uv / venv)
#   .pythonlibs    -> site-packages que usa el sandbox de Replit
#   node_modules   -> por si conviven con un sub-proyecto JS
#   .cache         -> caches del sistema (incluye pre-commit envs)
#   .uv-cache      -> cache local de uv (cubierto por clean-uv opt-in)
#   .local         -> metadatos internos del agente / harness
#   attached_assets-> archivos del usuario, intocables
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

help:  ## Muestra esta ayuda.
	@echo "Ophelian — limpieza del repo"
	@echo ""
	@echo "Uso: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Target principal del día a día. NO toca venv ni cache de uv.
# ---------------------------------------------------------------------------
clean: clean-pyc clean-test clean-lint clean-build clean-docs  ## Limpieza estándar (pyc + tests + lint + build + docs).
	@echo "✓ Repo limpio (venv y caches de uv preservados)."

# ---------------------------------------------------------------------------
# Bytecode de Python — siempre seguro de borrar.
# ---------------------------------------------------------------------------
clean-pyc:  ## Borra __pycache__/, *.pyc, *.pyo, *.pyd recursivamente.
	@echo "→ Borrando bytecode de Python..."
	@find . \( $(PRUNE) \) -prune -o -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@find . \( $(PRUNE) \) -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.pyd' \) -exec rm -f {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Caches y reportes de pytest / cobertura.
# ---------------------------------------------------------------------------
clean-test:  ## Borra .pytest_cache, .coverage, coverage.xml, htmlcov/.
	@echo "→ Borrando artefactos de tests y cobertura..."
	@find . \( $(PRUNE) \) -prune -o -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov coverage.xml .coverage .coverage.*

# ---------------------------------------------------------------------------
# Caches de los linters/typecheckers (ruff, mypy).
# ---------------------------------------------------------------------------
clean-lint:  ## Borra .ruff_cache/ y .mypy_cache/.
	@echo "→ Borrando caches de ruff y mypy..."
	@find . \( $(PRUNE) \) -prune -o -type d \( -name '.ruff_cache' -o -name '.mypy_cache' \) -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Artifacts de packaging.
# ---------------------------------------------------------------------------
clean-build:  ## Borra dist/, build/, *.egg-info/.
	@echo "→ Borrando artefactos de build..."
	@rm -rf dist build
	@find . \( $(PRUNE) \) -prune -o -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Salida estática de mkdocs.
# ---------------------------------------------------------------------------
clean-docs:  ## Borra site/ (output de mkdocs build).
	@echo "→ Borrando docs renderizadas..."
	@rm -rf site

# ---------------------------------------------------------------------------
# Caches "caros" — opt-in. No los incluyas en `clean` salvo que sepas
# que querés pagar el costo de recrearlos.
# ---------------------------------------------------------------------------
clean-precommit:  ## Borra .cache/pre-commit/ (entornos de hooks).
	@echo "→ Borrando entornos cacheados de pre-commit..."
	@rm -rf .cache/pre-commit

clean-uv:  ## Borra el cache local de uv (./.uv-cache si existe).
	@echo "→ Borrando cache local de uv..."
	@rm -rf .uv-cache

clean-venv:  ## Borra el entorno virtual .venv/ (lo recreás con `uv sync`).
	@echo "→ Borrando .venv/..."
	@rm -rf .venv

# ---------------------------------------------------------------------------
# Wipe completo. Tras esto, recreás todo con `uv sync --frozen --extra dev`.
# ---------------------------------------------------------------------------
distclean: clean clean-precommit clean-uv clean-venv  ## Limpieza total: clean + precommit + uv + venv.
	@echo "✓ Wipe completo. Reconstruí con: uv sync --frozen --extra dev"
