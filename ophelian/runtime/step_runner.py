"""Entry point used inside Standalone-provider containers and EC2 workers.

The Standalone provider (container mode) and the AWS EC2 driver both
write a small ``step.json`` describing the node, the kind, and any
upstream artifact references (local paths for containers, ``s3://``
URIs for EC2 workers). The container/instance then runs::

    python -m ophelian.runtime.step_runner /work/step.json

If ``OPHELIAN_ARTIFACT_BUCKET`` and ``OPHELIAN_RUN_ID`` are set in the
environment (always true for EC2 workers spawned by ``EC2Driver``),
``step_runner``:

1. downloads any ``s3://...`` upstream artifacts into ``/work/_upstream``
   so the in-process handlers see plain local paths;
2. runs the handler;
3. uploads each produced artifact (file or directory) to S3 under
   ``runs/{run_id}/{step}/artifacts/{name}`` and rewrites
   ``result.artifacts`` to the resulting ``s3://...`` URIs so downstream
   steps (running on fresh EC2 instances) can fetch them;
4. writes ``/work/result.json`` for the cloud driver to pick up.

In container mode neither env var is set, so steps 1+3 are no-ops and
the existing local-path contract is preserved.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any

from ophelian.core.nodes import Data, Deploy, Eval, Train, Tune
from ophelian.providers import standalone as _standalone

_NODE_TYPES: dict[str, type[Any]] = {
    "data": Data,
    "train": Train,
    "tune": Tune,
    "eval": Eval,
    "deploy": Deploy,
}


def _materialise_upstream_from_s3(
    artifacts: dict[str, dict[str, str]], work_dir: Path
) -> dict[str, dict[str, str]]:
    """Download every ``s3://...`` URI in *artifacts* into *work_dir* and
    return the same shape with local paths instead.

    Lazy boto3 import — keeps the module importable in container mode
    without the ``aws`` extra installed.
    """
    needs_s3 = any(
        isinstance(v, str) and v.startswith("s3://")
        for items in artifacts.values()
        for v in items.values()
    )
    if not needs_s3:
        return artifacts
    from ophelian.stores.s3 import S3ArtifactStore, parse_s3_uri

    out: dict[str, dict[str, str]] = {}
    cache: dict[str, S3ArtifactStore] = {}
    upstream_root = work_dir / "_upstream"
    upstream_root.mkdir(parents=True, exist_ok=True)
    for step_name, items in artifacts.items():
        local_items: dict[str, str] = {}
        for key, uri in items.items():
            if not (isinstance(uri, str) and uri.startswith("s3://")):
                local_items[key] = uri
                continue
            bucket, full_key = parse_s3_uri(uri)
            store = cache.get(bucket) or S3ArtifactStore(bucket=bucket)
            cache[bucket] = store
            local_dst = upstream_root / step_name / key
            local_dst.parent.mkdir(parents=True, exist_ok=True)
            local_items[key] = str(store.get(full_key, local_dst))
        out[step_name] = local_items
    return out


def _persist_artifacts_to_s3(
    artifacts: dict[str, str],
    run_id: str,
    step_name: str,
    bucket: str,
    prefix: str = "",
) -> dict[str, str]:
    """Upload local-path artifacts to S3 under
    ``{prefix}/runs/{run_id}/{step_name}/artifacts/{name}`` and return
    the rewritten ``{name: s3://...}`` mapping. ``prefix`` mirrors
    ``AWSConfig.artifact_prefix`` so all writes stay inside whatever
    sub-path the user pinned (important for buckets with policies that
    grant write access to a single prefix only). Already-S3 URIs are
    passed through unchanged.
    """
    from ophelian.stores.s3 import S3ArtifactStore

    store = S3ArtifactStore(bucket=bucket, prefix=prefix)
    rewritten: dict[str, str] = {}
    for name, value in artifacts.items():
        if not isinstance(value, str):
            rewritten[name] = value
            continue
        if value.startswith(("s3://", "http://", "https://")):
            rewritten[name] = value
            continue
        path = Path(value)
        if not path.exists():
            rewritten[name] = value
            continue
        base = f"runs/{run_id}/{step_name}/artifacts/{name}"
        if path.is_dir():
            rewritten[name] = store.put(base, path)
        else:
            rewritten[name] = store.put(f"{base}/{path.name}", path)
    return rewritten


def _upload_checkpoint_to_s3(
    checkpoint_dir: Path,
    *,
    bucket: str,
    prefix: str,
    run_id: str,
    step_name: str,
) -> str | None:
    """Upload ``checkpoint_dir`` to
    ``s3://{bucket}/{prefix}/runs/{run_id}/checkpoints/{step_name}`` so
    the next provider-driven retry can resume from it. Returns the
    rewritten URI, or ``None`` when the directory is missing/empty.
    """
    if not checkpoint_dir.exists():
        return None
    if not any(checkpoint_dir.iterdir()):
        return None
    from ophelian.stores.s3 import S3ArtifactStore

    store = S3ArtifactStore(bucket=bucket, prefix=prefix)
    sub = f"runs/{run_id}/checkpoints/{step_name}"
    return store.put(sub, checkpoint_dir)


def _install_sigterm_checkpoint_uploader(
    *,
    work_dir: Path,
    bucket: str | None,
    prefix: str,
    run_id: str | None,
    step_name: str | None,
) -> None:
    """Wire SIGTERM (sent by AWS ~2 minutes before spot reclamation) to
    a best-effort upload of ``work_dir/checkpoint`` to S3 so the
    provider's retry can pick the snapshot back up. We exit ``75``
    (``EX_TEMPFAIL``) so the cloud driver can distinguish "killed,
    please retry" from "step actually failed".

    No-op when running in container mode (no S3 env wired up) or when
    the necessary identifiers are missing — keeps the local Standalone
    contract unchanged.
    """
    if not (bucket and run_id and step_name):
        return

    def _handler(signum: int, _frame: FrameType | None) -> None:
        try:
            uri = _upload_checkpoint_to_s3(
                work_dir / "checkpoint",
                bucket=bucket,
                prefix=prefix,
                run_id=run_id,
                step_name=step_name,
            )
            sys.stderr.write(
                f"[step_runner] SIGTERM ({signum}) — checkpoint uploaded to {uri}\n"
            )
        except Exception as exc:  # pragma: no cover - best effort
            sys.stderr.write(
                f"[step_runner] SIGTERM checkpoint upload failed: {exc}\n"
            )
        sys.exit(75)

    signal.signal(signal.SIGTERM, _handler)


def _download_resume_checkpoint(uri: str, work_dir: Path) -> Path:
    """Materialise a checkpoint URI into ``/work/_resume`` and return it.

    Accepts either an ``s3://...`` URI (the production path used by the
    AWS provider when retrying an interrupted Train step) or a plain
    local path (used by tests and by the in-process executor).
    """
    if uri.startswith("s3://"):
        from ophelian.stores.s3 import S3ArtifactStore, parse_s3_uri

        bucket, full_key = parse_s3_uri(uri)
        dst = work_dir / "_resume"
        dst.mkdir(parents=True, exist_ok=True)
        store = S3ArtifactStore(bucket=bucket)
        return Path(store.get(full_key, dst))
    return Path(uri)


def run(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    kind = spec["kind"]
    node_payload = spec["node"]
    artifacts: dict[str, dict[str, str]] = spec.get("artifacts", {})
    work_dir = spec_path.parent

    node_cls = _NODE_TYPES[kind]
    node = node_cls(**node_payload)

    bucket = os.environ.get("OPHELIAN_ARTIFACT_BUCKET") or None
    run_id = os.environ.get("OPHELIAN_RUN_ID") or None
    prefix = os.environ.get("OPHELIAN_ARTIFACT_PREFIX", "")
    resume_uri = os.environ.get("OPHELIAN_RESUME_FROM", "") or spec.get("resume_from", "")

    # Best-effort spot-reclamation snapshot. Has to come BEFORE the
    # handler runs so that an early SIGTERM during artifact download
    # still triggers an upload — empty checkpoint dirs are filtered
    # out inside the helper.
    if kind == "train":
        _install_sigterm_checkpoint_uploader(
            work_dir=work_dir,
            bucket=bucket,
            prefix=prefix,
            run_id=run_id,
            step_name=spec.get("node", {}).get("name") or "train",
        )

    # On EC2 workers the upstream artifacts are S3 URIs — pull them down
    # so the in-process handlers see plain filesystem paths.
    if bucket and run_id:
        artifacts = _materialise_upstream_from_s3(artifacts, work_dir)

    # If the provider is retrying an interrupted Train step, fetch the
    # last checkpoint snapshot it left in S3 and hand it to the Train
    # handler so the adapter can pick up where it left off.
    resume_from: Path | None = None
    if resume_uri:
        resume_from = _download_resume_checkpoint(resume_uri, work_dir)

    # Reuse the in-process handlers; we instantiate a provider in
    # in-process mode rooted at the mounted workspace so paths line up.
    provider = _standalone.StandaloneProvider(
        local=True,
        workspace=work_dir,
        container=False,
    )
    if kind == "data":
        result = provider._handle_data(node, work_dir)
    elif kind == "train":
        result = provider._handle_train(node, work_dir, artifacts, resume_from=resume_from)
    elif kind == "tune":
        result = provider._handle_tune(node, work_dir, artifacts)
    elif kind == "eval":
        result = provider._handle_eval(node, work_dir, artifacts)
    elif kind == "deploy":
        # Build the FastAPI app + descriptor and emit a result.json
        # describing how to serve it. The caller (e.g. the EC2 user-data
        # script, or the local Standalone container entrypoint) is then
        # responsible for actually launching uvicorn against the
        # descriptor — this keeps step_runner side-effect-free and lets
        # the cloud driver pick the right host/port to advertise.
        result = provider._handle_deploy(node, work_dir, artifacts)
    else:
        raise ValueError(f"Unknown step kind: {kind!r}")

    # Push artifacts to S3 (if configured) so downstream steps running
    # on fresh EC2 instances can fetch them — the driver only reads
    # `result.json` and trusts the URIs in `result.artifacts`.
    out_artifacts = dict(result.artifacts)
    if bucket and run_id:
        out_artifacts = _persist_artifacts_to_s3(
            out_artifacts, run_id, result.name, bucket, prefix=prefix
        )

    payload = {
        "name": result.name,
        "kind": result.kind,
        "status": result.status,
        "metrics": result.metrics,
        "artifacts": out_artifacts,
        "info": result.info,
        "error": result.error,
    }
    (work_dir / "result.json").write_text(json.dumps(payload, default=str))
    return 0 if result.status == "success" else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print("usage: python -m ophelian.runtime.step_runner <step.json>", file=sys.stderr)
        return 2
    return run(Path(argv[0]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
