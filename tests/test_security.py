"""Security tests.

These cover the small attack surface a single-process ML pipeline
framework actually has:

* **Path traversal in the local artifact store** — keys come from
  pipeline definitions, environment variables, and (in some configs)
  the network. A malicious ``key="../../etc/whatever"`` must NOT be
  able to write outside the configured root.
* **Symlink escape** — a pre-existing symlink under the root must not
  let writes leak out of the root.
* **Secret leakage in the run summary** — the user-visible summary
  emitted at the end of every run must never echo back arbitrary
  metric/artifact values that look like AWS access keys or bearer
  tokens. We don't promise full DLP; we promise "obvious shapes".

These are framework-level guarantees, not provider-level. The cloud
stores (S3/GCS/Azure) refuse path-traversal at the bucket layer, but
``LocalArtifactStore`` is what tests, the standalone provider, and the
docker runtime use as the workspace store — and a path-traversal there
means arbitrary local file writes/reads.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
from ophelian.core.nodes import StepResult
from ophelian.observability.summary import emit_run_summary
from ophelian.stores.local import LocalArtifactStore

# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------


def test_local_store_rejects_parent_dir_escape_in_put_bytes(tmp_path: Path) -> None:
    """``put_bytes("../escape", ...)`` must NOT write to the parent
    directory. If the framework lets a pipeline-supplied key escape the
    root, every artifact store on local disk becomes a write-anywhere
    primitive."""
    root = tmp_path / "store"
    sentinel_outside = tmp_path / "escape"
    store = LocalArtifactStore(root)

    with pytest.raises((ValueError, OSError, PermissionError)):
        store.put_bytes("../escape", b"pwned")

    assert not sentinel_outside.exists(), (
        "put_bytes('../escape', ...) wrote to the parent directory — "
        "LocalArtifactStore allows path traversal."
    )


def test_local_store_rejects_parent_dir_escape_in_put_file(tmp_path: Path) -> None:
    src = tmp_path / "payload"
    src.write_bytes(b"pwned")
    root = tmp_path / "store"
    store = LocalArtifactStore(root)
    sentinel_outside = tmp_path / "escape"

    with pytest.raises((ValueError, OSError, PermissionError)):
        store.put("../escape", src)

    assert not sentinel_outside.exists()


def test_local_store_rejects_absolute_key(tmp_path: Path) -> None:
    """An absolute key must be rejected outright. `Path(root) / "/etc/x"`
    in pure pathlib silently *replaces* the root with the absolute key
    on POSIX — a bug class we want zero tolerance for."""
    root = tmp_path / "store"
    store = LocalArtifactStore(root)

    with pytest.raises((ValueError, OSError, PermissionError)):
        store.put_bytes("/tmp/ophelian_traversal_marker", b"pwned")
    assert not Path("/tmp/ophelian_traversal_marker").exists()


def test_local_store_rejects_get_with_parent_dir_escape(tmp_path: Path) -> None:
    """Read-side path traversal is just as bad — it lets a pipeline
    read arbitrary files (SSH keys, /etc/passwd) by passing a crafted
    key to ``store.get(...)``."""
    secret = tmp_path / "secret"
    secret.write_text("very secret")
    root = tmp_path / "store"
    store = LocalArtifactStore(root)

    with pytest.raises((FileNotFoundError, ValueError, OSError, PermissionError)):
        store.get("../secret")


def test_local_store_rejects_symlink_escape(tmp_path: Path) -> None:
    """An attacker who pre-plants a symlink ``store/x -> /tmp/escape``
    (e.g. via a previous job on a shared workspace) must NOT be able
    to make a future ``put_bytes("x", ...)`` write through the link."""
    root = tmp_path / "store"
    root.mkdir()
    target = tmp_path / "escape_target"
    link = root / "link"
    link.symlink_to(target)
    store = LocalArtifactStore(root)

    with pytest.raises((ValueError, OSError, PermissionError)):
        store.put_bytes("link", b"pwned")
    assert not target.exists(), "symlink escape allowed write outside the root"


def test_local_store_rejects_read_through_symlink_escape(tmp_path: Path) -> None:
    """Symmetric to the write-side test: a pre-existing symlink under
    the root that points to an outside file must NOT let a caller
    *read* arbitrary host files through ``get_bytes``."""
    secret = tmp_path / "host_secret"
    secret.write_text("ssh-rsa AAAA... user@host")
    root = tmp_path / "store"
    root.mkdir()
    (root / "link").symlink_to(secret)
    store = LocalArtifactStore(root)

    with pytest.raises((ValueError, OSError, PermissionError, FileNotFoundError)):
        store.get_bytes("link")


def test_local_store_normal_keys_still_work(tmp_path: Path) -> None:
    """Sanity: locking down traversal must not break normal nested
    keys like ``runs/<id>/<step>/artifacts/model``."""
    store = LocalArtifactStore(tmp_path / "store")
    store.put_bytes("runs/r1/train/artifacts/model.bin", b"hello")
    assert store.exists("runs/r1/train/artifacts/model.bin")
    assert store.get_bytes("runs/r1/train/artifacts/model.bin") == b"hello"


# ---------------------------------------------------------------------------
# Secret leakage in run summary
# ---------------------------------------------------------------------------


_SECRET_PATTERNS = [
    "AKIA1234567890ABCDEF",  # AWS access key id shape
    "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # GitHub PAT shape
    "Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",  # JWT bearer token
]


@pytest.mark.parametrize("secret", _SECRET_PATTERNS)
def test_run_summary_does_not_leak_obvious_secrets_in_artifact_uri(
    secret: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real artifact URIs sometimes get presigned-with-token style
    suffixes; the summary table renders those URIs verbatim. If a
    user accidentally bakes credentials into a metric/artifact value,
    the summary line is the most-visible place they can leak (it
    lands in CI logs, Slack notifications, and stderr).

    We assert the summary REDACTS obvious secret shapes. Today's
    behaviour does not redact — this test will FAIL until either
    redaction is added or the test is downgraded to xfail with a
    documented rationale. Either outcome is preferable to silently
    leaking tokens."""
    monkeypatch.delenv("OPHELIAN_NO_SUMMARY", raising=False)
    monkeypatch.delenv("OPHELIAN_LOG_FORMAT", raising=False)
    buffer = io.StringIO()
    step = StepResult(
        name="trainer",
        kind="train",
        status="success",
        artifacts={"model": f"s3://bucket/model?token={secret}"},
        metrics={"accuracy": 0.9},
        duration_seconds=1.0,
    )
    emit_run_summary(
        provider="standalone",
        run_id="r1",
        pipeline="p",
        steps=[step],
        stream=buffer,
    )
    rendered = buffer.getvalue()
    assert secret not in rendered, (
        "Summary rendered a secret-shaped value verbatim. "
        "Add redaction (e.g. mask anything matching AKIA[0-9A-Z]{16}, "
        "ghp_[A-Za-z0-9]+, Bearer <jwt>) before printing artifact URIs."
    )


def test_run_summary_json_mode_does_not_leak_secrets_to_logger(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Same guarantee in JSON mode — structured logs are *more* likely
    to be aggregated and shared, so the redaction must apply there too."""
    monkeypatch.delenv("OPHELIAN_NO_SUMMARY", raising=False)
    monkeypatch.setenv("OPHELIAN_LOG_FORMAT", "json")
    secret = "AKIA1234567890ABCDEF"
    step = StepResult(
        name="trainer",
        kind="train",
        status="success",
        artifacts={"model": f"s3://bucket/path?aws_id={secret}"},
        duration_seconds=0.5,
    )
    with caplog.at_level(logging.INFO, logger="ophelian.observability.summary"):
        emit_run_summary(
            provider="standalone",
            run_id="r2",
            pipeline="p",
            steps=[step],
        )

    record_blob = ""
    for record in caplog.records:
        # Both the message and the structured extras must be searched.
        record_blob += record.getMessage()
        record_blob += json.dumps(getattr(record, "__dict__", {}), default=str)
    assert secret not in record_blob, (
        "JSON-mode summary leaked a secret-shaped artifact URI to the logger."
    )


def test_run_summary_skip_env_actually_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense in depth: an operator who sets
    ``OPHELIAN_NO_SUMMARY=1`` must get **zero** output, no exceptions,
    no half-rendered tables. Catches a refactor that conditionally
    bypasses the env check."""
    monkeypatch.setenv("OPHELIAN_NO_SUMMARY", "1")
    buffer = io.StringIO()
    emit_run_summary(
        provider="standalone",
        run_id="r3",
        pipeline="p",
        steps=[],
        stream=buffer,
    )
    assert buffer.getvalue() == "", (
        f"OPHELIAN_NO_SUMMARY=1 must produce no output; got {buffer.getvalue()!r}"
    )
