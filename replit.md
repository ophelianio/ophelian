# Workspace

## Overview

This workspace contains two projects:
1. **Ophelian** — A Python PySpark ML library (the main project copied from the zip)
2. **pnpm monorepo** — TypeScript/Node.js workspace for the API server artifact

## Ophelian Python Project

**Ophelian** is a PySpark ML framework for putting ML prototypes into production.

### Structure

```
ophelian/                  # Main Python package
  ophelian_spark/          # PySpark wrappers
    evaluation/            # Metrics and feature evaluation
    ml/                    # ML pipelines (supervised, unsupervised, sampling, etc.)
    read/                  # Spark read utilities
    write/                 # Spark write utilities
    session/               # Spark session management
    streaming/             # Streaming support
    functions.py           # Core DataFrame extensions (Shape, PctChange, etc.)
    generic.py             # Generic utility functions
  start.py                 # OphelianSession entry point
tests/                     # Unit tests (pytest)
tutorials/                 # Jupyter notebooks + sample data
docs/                      # Documentation and images
pyproject.toml             # Poetry build config (v0.1.4)
requirements.txt           # Pinned dependencies
Dockerfile                 # Docker build
Makefile                   # Dev shortcuts
```

### Key Dependencies
- Python >= 3.9, < 3.12
- PySpark 3.2.2
- NumPy 1.26.4, Pandas 2.2.2
- scikit-learn, TensorFlow, SHAP, Dask, PyArrow

### Install
```sh
pip install ophelian==0.1.4
```

### Usage
```python
from ophelian.start import OphelianSession
ophelian = OphelianSession("App Name")
sc = ophelian.Spark.build_spark_context()
```

---

## pnpm Monorepo (TypeScript/Node.js)

### Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

### Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.
