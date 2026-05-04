"""Entry point used inside Standalone-provider containers and cloud workers.

The Standalone provider (container mode), the AWS EC2 driver, the GCP
GCE driver and the Azure VM driver all write a small ``step.json``
describing the node, the kind, and any upstream artifact references
(local paths for containers, ``s3://``/``gs://``/``az://`` URIs for
cloud workers). The container/instance then runs::

    python -m ophelian.runtime.step_runner /work/step.json

The active artifact backend is selected at run-time:

* ``OPHELIAN_ARTIFACT_BACKEND=s3`` (or unset, with
  ``OPHELIAN_ARTIFACT_BUCKET`` set) — AWS S3 (legacy default).
* ``OPHELIAN_ARTIFACT_BACKEND=gcs`` — Google Cloud Storage.
* ``OPHELIAN_ARTIFACT_BACKEND=azure`` — Azure Blob Storage.

When a cloud backend is selected ``step_runner``:

1. downloads any cloud-URI upstream artifacts into ``/work/_upstream``
   so the in-process handlers see plain local paths;
2. runs the handler;
3. uploads each produced artifact (file or directory) under
   ``runs/{run_id}/{step}/artifacts/{name}`` and rewrites
   ``result.artifacts`` to the resulting cloud URIs so downstream steps
   (running on fresh workers) can fetch them;
4. writes ``/work/result.json`` for the cloud driver to pick up.

In container mode no backend env var is set, so steps 1+3 are no-ops
and the existing local-path contract is preserved.
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


def _detect_backend() -> str | None:
    """Return ``'s3'`` / ``'gcs'`` / ``'azure'`` / ``None`` based on env vars."""
    explicit = os.environ.get("OPHELIAN_ARTIFACT_BACKEND")
    if explicit:
        return explicit.lower()
    if os.environ.get("OPHELIAN_ARTIFACT_BUCKET"):
        return "s3"
    return None


def _build_runtime_store(backend: str) -> Any:
    """Build the artifact store the worker uses on this cloud."""
    prefix = os.environ.get("OPHELIAN_ARTIFACT_PREFIX", "")
    if backend == "s3":
        from ophelian.stores.s3 import S3ArtifactStore

        bucket = os.environ.get("OPHELIAN_ARTIFACT_BUCKET", "")
        return S3ArtifactStore(bucket=bucket, prefix=prefix)
    if backend == "gcs":
        from ophelian.stores.gcs import GCSArtifactStore

        bucket = os.environ.get("OPHELIAN_ARTIFACT_BUCKET", "")
        project = os.environ.get("OPHELIAN_GCP_PROJECT")
        return GCSArtifactStore(bucket=bucket, prefix=prefix, project=project, ensure_bucket=False)
    if backend == "azure":
        from ophelian.stores.azure_blob import AzureBlobArtifactStore

        account = os.environ.get("OPHELIAN_ARTIFACT_ACCOUNT", "")
        container = os.environ.get("OPHELIAN_ARTIFACT_CONTAINER", "")
        return AzureBlobArtifactStore(
            account=account,
            container=container,
            prefix=prefix,
            ensure_container=False,
        )
    raise ValueError(f"Unknown OPHELIAN_ARTIFACT_BACKEND={backend!r}")


def _backend_uri_prefix(backend: str) -> str:
    return {"s3": "s3://", "gcs": "gs://", "azure": "az://"}[backend]


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
            sys.stderr.write(f"[step_runner] SIGTERM ({signum}) — checkpoint uploaded to {uri}\n")
        except Exception as exc:  # pragma: no cover - best effort
            sys.stderr.write(f"[step_runner] SIGTERM checkpoint upload failed: {exc}\n")
        sys.exit(75)

    signal.signal(signal.SIGTERM, _handler)


def _download_resume_checkpoint(uri: str, work_dir: Path) -> Path:
    """Materialise a checkpoint URI into ``/work/_resume`` and return it.

    Accepts ``s3://``, ``gs://``, ``az://`` URIs (production paths
    used by the cloud providers when retrying an interrupted Train
    step) or a plain local path (tests + in-process executor).
    """
    dst = work_dir / "_resume"
    dst.mkdir(parents=True, exist_ok=True)
    if uri.startswith("s3://"):
        from ophelian.stores.s3 import S3ArtifactStore, parse_s3_uri

        bucket, full_key = parse_s3_uri(uri)
        return Path(S3ArtifactStore(bucket=bucket).get(full_key, dst))
    if uri.startswith("gs://"):
        from ophelian.stores.gcs import GCSArtifactStore, parse_gs_uri

        bucket, full_key = parse_gs_uri(uri)
        return Path(GCSArtifactStore(bucket=bucket, ensure_bucket=False).get(full_key, dst))
    if uri.startswith("az://"):
        from ophelian.stores.azure_blob import AzureBlobArtifactStore, parse_az_uri

        account, container, full_key = parse_az_uri(uri)
        store = AzureBlobArtifactStore(account=account, container=container, ensure_container=False)
        return Path(store.get(full_key, dst))
    return Path(uri)


def _materialise_upstream_multicloud(
    artifacts: dict[str, dict[str, str]], work_dir: Path
) -> dict[str, dict[str, str]]:
    """Backend-aware version of :func:`_materialise_upstream_from_s3`.

    Inspects each URI and routes to the appropriate store. Returns the
    same shape with cloud URIs replaced by local paths.
    """
    needs_remote = any(
        isinstance(v, str) and v.startswith(("s3://", "gs://", "az://"))
        for items in artifacts.values()
        for v in items.values()
    )
    if not needs_remote:
        return artifacts
    upstream_root = work_dir / "_upstream"
    upstream_root.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, str]] = {}
    for step_name, items in artifacts.items():
        local_items: dict[str, str] = {}
        for key, uri in items.items():
            if not isinstance(uri, str):
                local_items[key] = uri
                continue
            local_dst = upstream_root / step_name / key
            local_dst.parent.mkdir(parents=True, exist_ok=True)
            if uri.startswith("s3://"):
                from ophelian.stores.s3 import S3ArtifactStore, parse_s3_uri

                bucket, full_key = parse_s3_uri(uri)
                local_items[key] = str(S3ArtifactStore(bucket=bucket).get(full_key, local_dst))
            elif uri.startswith("gs://"):
                from ophelian.stores.gcs import GCSArtifactStore, parse_gs_uri

                bucket, full_key = parse_gs_uri(uri)
                local_items[key] = str(
                    GCSArtifactStore(bucket=bucket, ensure_bucket=False).get(full_key, local_dst)
                )
            elif uri.startswith("az://"):
                from ophelian.stores.azure_blob import (
                    AzureBlobArtifactStore,
                    parse_az_uri,
                )

                account, container, full_key = parse_az_uri(uri)
                store = AzureBlobArtifactStore(
                    account=account, container=container, ensure_container=False
                )
                local_items[key] = str(store.get(full_key, local_dst))
            else:
                local_items[key] = uri
        out[step_name] = local_items
    return out


def _persist_artifacts_to_store(
    artifacts: dict[str, str],
    run_id: str,
    step_name: str,
    store: Any,
) -> dict[str, str]:
    """Backend-agnostic version of :func:`_persist_artifacts_to_s3`."""
    rewritten: dict[str, str] = {}
    for name, value in artifacts.items():
        if not isinstance(value, str):
            rewritten[name] = value
            continue
        if value.startswith(("s3://", "gs://", "az://", "http://", "https://")):
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


def _upload_checkpoint_to_store(
    checkpoint_dir: Path,
    *,
    store: Any,
    run_id: str,
    step_name: str,
) -> str | None:
    if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
        return None
    sub = f"runs/{run_id}/checkpoints/{step_name}"
    return str(store.put(sub, checkpoint_dir))


def _install_sigterm_checkpoint_uploader_multicloud(
    *,
    work_dir: Path,
    backend: str | None,
    run_id: str | None,
    step_name: str | None,
) -> None:
    """Backend-aware version of :func:`_install_sigterm_checkpoint_uploader`.

    Uploads the in-flight checkpoint dir to whatever cloud store the
    worker was configured for (S3 / GCS / Azure Blob) and exits 75
    (``EX_TEMPFAIL``) so the cloud driver retries.
    """
    if not (backend and run_id and step_name):
        return

    def _handler(signum: int, _frame: FrameType | None) -> None:
        try:
            store = _build_runtime_store(backend)
            uri = _upload_checkpoint_to_store(
                work_dir / "checkpoint",
                store=store,
                run_id=run_id,
                step_name=step_name,
            )
            sys.stderr.write(f"[step_runner] SIGTERM ({signum}) — checkpoint uploaded to {uri}\n")
        except Exception as exc:  # pragma: no cover - best effort
            sys.stderr.write(f"[step_runner] SIGTERM checkpoint upload failed: {exc}\n")
        sys.exit(75)

    signal.signal(signal.SIGTERM, _handler)


def run(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    kind = spec["kind"]
    node_payload = spec["node"]
    artifacts: dict[str, dict[str, str]] = spec.get("artifacts", spec.get("upstream_artifacts", {}))
    work_dir = spec_path.parent

    node_cls = _NODE_TYPES[kind]
    node = node_cls(**node_payload)

    backend = _detect_backend()
    run_id = os.environ.get("OPHELIAN_RUN_ID") or None
    resume_uri = os.environ.get("OPHELIAN_RESUME_FROM", "") or spec.get("resume_from", "")

    # Bind run_id into the observability ContextVar so every log line
    # the worker emits carries the correlation id.
    if run_id:
        try:
            from ophelian.observability import set_run_id

            set_run_id(run_id)
        except Exception:  # pragma: no cover - defensive
            pass

    # Best-effort spot-reclamation snapshot. Has to come BEFORE the
    # handler runs so that an early SIGTERM during artifact download
    # still triggers an upload — empty checkpoint dirs are filtered
    # out inside the helper.
    if kind == "train":
        _install_sigterm_checkpoint_uploader_multicloud(
            work_dir=work_dir,
            backend=backend,
            run_id=run_id,
            step_name=spec.get("node", {}).get("name") or "train",
        )

    # On cloud workers the upstream artifacts are cloud URIs — pull
    # them down so the in-process handlers see plain filesystem paths.
    if backend and run_id:
        artifacts = _materialise_upstream_multicloud(artifacts, work_dir)

    # If the provider is retrying an interrupted Train step, fetch the
    # last checkpoint snapshot it left in cloud storage and hand it to
    # the Train handler so the adapter can pick up where it left off.
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
    # Wrap the handler call in an OTel step span so cloud workers
    # contribute to the same trace shape as the in-process executor.
    # We keep ``emit_metric=True`` (the default) here because this
    # process owns the authoritative ``ophelian.step.duration`` sample
    # for container/cloud-executed steps. Host-side container drivers
    # must NOT also record one — see the cardinality contract on
    # ``ophelian.observability.otel.step_span``.
    from ophelian.observability.events import (
        StepCompleted,
        StepFailed,
        StepStarted,
    )
    from ophelian.observability.events import (
        emit as emit_lifecycle,
    )
    from ophelian.observability.otel import step_span

    step_name = (spec.get("node", {}) or {}).get("name") or kind
    # ``spec["context"]`` is the opaque dict propagated from
    # ``Pipeline.context`` by the cloud/container drivers' step-spec
    # encoders. May be absent (older specs) or explicitly null.
    step_context = spec.get("context") if isinstance(spec.get("context"), dict) else None
    step_ctx = step_span(
        step_name=step_name,
        step_kind=kind,
        run_id=run_id,
        provider="standalone-runtime",
    )
    import time as _time

    _t0 = _time.monotonic()
    with step_ctx:
        emit_lifecycle(
            StepStarted(
                source=step_name,
                run_id=run_id,
                context=step_context,
                step_name=step_name,
                step_kind=kind,
                provider="standalone-runtime",
            )
        )
        try:
            result = _dispatch_handler(
                provider=provider,
                kind=kind,
                node=node,
                work_dir=work_dir,
                artifacts=artifacts,
                resume_from=resume_from,
            )
        except Exception as exc:
            emit_lifecycle(
                StepFailed(
                    source=step_name,
                    run_id=run_id,
                    context=step_context,
                    step_name=step_name,
                    step_kind=kind,
                    provider="standalone-runtime",
                    duration_seconds=_time.monotonic() - _t0,
                    error=str(exc),
                )
            )
            raise
        if result.status == "failed":
            emit_lifecycle(
                StepFailed(
                    source=step_name,
                    run_id=run_id,
                    context=step_context,
                    step_name=step_name,
                    step_kind=kind,
                    provider="standalone-runtime",
                    duration_seconds=_time.monotonic() - _t0,
                    error=result.error or "unknown",
                )
            )
        else:
            emit_lifecycle(
                StepCompleted(
                    source=step_name,
                    run_id=run_id,
                    context=step_context,
                    step_name=step_name,
                    step_kind=kind,
                    provider="standalone-runtime",
                    duration_seconds=_time.monotonic() - _t0,
                    status=result.status,
                )
            )

    # Push artifacts to whatever cloud store is configured so
    # downstream steps running on fresh workers can fetch them.
    out_artifacts = dict(result.artifacts)
    if backend and run_id:
        store = _build_runtime_store(backend)
        out_artifacts = _persist_artifacts_to_store(out_artifacts, run_id, result.name, store)

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


def _dispatch_handler(
    *,
    provider: Any,
    kind: str,
    node: Any,
    work_dir: Path,
    artifacts: dict[str, dict[str, str]],
    resume_from: Path | None,
) -> Any:
    """Route a step kind to the matching standalone in-process handler.

    The deploy path is not actually side-effectful here — it builds
    the FastAPI app + descriptor and emits a result that the caller
    (e.g. the EC2 user-data script or the local Standalone container
    entrypoint) is responsible for launching uvicorn against. Keeping
    step_runner side-effect-free lets the cloud driver pick the right
    host/port to advertise.
    """
    if kind == "data":
        return provider._handle_data(node, work_dir)
    if kind == "train":
        return provider._handle_train(node, work_dir, artifacts, resume_from=resume_from)
    if kind == "tune":
        return provider._handle_tune(node, work_dir, artifacts)
    if kind == "eval":
        return provider._handle_eval(node, work_dir, artifacts)
    if kind == "deploy":
        return provider._handle_deploy(node, work_dir, artifacts)
    raise ValueError(f"Unknown step kind: {kind!r}")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print("usage: python -m ophelian.runtime.step_runner <step.json>", file=sys.stderr)
        return 2
    return run(Path(argv[0]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
