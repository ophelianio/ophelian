# Standalone

```python
from ophelian import Standalone
env = Standalone(local=True)
```

`Standalone` is the local-first env. It runs every step in a
**dedicated Docker container** when a Docker daemon is available, and
falls back to running the same logic **in-process** when it isn't.

This is the env the test suite uses, and the one you'll use for 90%
of inner-loop development.

## Constructor

```python
Standalone(
    local=True,            # Required — guards against accidental remote runs.
    image="python:3.12-slim",
    artifact_dir="./.ophelian",
    extras=("sklearn",),   # Pip extras installed inside the container.
)
```

## What you get

- Per-step container isolation when Docker is up.
- Identical step contract to the cloud envs — your steps see the same
  `OPHELIAN_RUN_ID`, the same artifact paths, the same checkpoint API.
- A `LocalArtifactStore` writing under `./.ophelian/<run_id>/`.
- The full structured JSON log + summary table at the end.

## When to use it

- Inner-loop dev.
- CI without cloud credentials.
- Reproducing a bug a cloud user reported, without touching a cloud.

## When *not* to use it

- Anything that needs more than your laptop's GPU. Use `AWS`, `GCP`,
  `Azure`, or `Auto`.
