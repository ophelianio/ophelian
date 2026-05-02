"""Resilience tests for opt-in / best-effort code paths.

The pricing module is the canonical example: it advertises that
* live fetches NEVER raise — they return ``None`` or an empty list,
* the on-disk cache tolerates corruption,
* missing optional dependencies (boto3, requests) degrade gracefully.

These guarantees are easy to violate in a refactor — one stray
``raise`` inside ``_http_get_json`` and every ``Auto(...)`` call in a
user pipeline starts crashing the whole run. We pin the contract.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
from ophelian.pricing import lookup_cheapest
from ophelian.pricing._http import _http_get_json
from ophelian.pricing.cache import _load_cache, _save_cache
from ophelian.pricing.live import aws as live_aws

# ---------------------------------------------------------------------------
# Cache resilience
# ---------------------------------------------------------------------------


def test_load_cache_returns_empty_dict_when_file_missing(tmp_path: Path) -> None:
    """A fresh install has no cache file — must return ``{}``, never raise."""
    assert _load_cache(tmp_path / "nope.json") == {}


def test_load_cache_tolerates_corrupt_json(tmp_path: Path) -> None:
    """A truncated/corrupt cache (process killed mid-write, disk full)
    must NOT brick all subsequent pricing lookups. Return ``{}``."""
    p = tmp_path / "pricing.json"
    p.write_text("{not json at all,,,,")
    assert _load_cache(p) == {}


def test_load_cache_tolerates_non_dict_top_level(tmp_path: Path) -> None:
    """A cache whose top-level is a list/string (e.g. an old format)
    must be ignored, not coerced into a dict-shaped TypeError later."""
    p = tmp_path / "pricing.json"
    p.write_text(json.dumps(["not", "a", "dict"]))
    assert _load_cache(p) == {}


def test_save_cache_creates_missing_parent_directories(tmp_path: Path) -> None:
    """The default cache lives at ``~/.cache/ophelian/pricing.json`` —
    on a fresh machine the parent does not exist. Save must mkdir."""
    target = tmp_path / "deep" / "nested" / "pricing.json"
    _save_cache({"aws": {"fetched_at": 0, "quotes": []}}, target)
    assert target.exists()
    assert _load_cache(target) == {"aws": {"fetched_at": 0, "quotes": []}}


# ---------------------------------------------------------------------------
# HTTP helper resilience — every error class returns None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        URLError("dns failure"),
        TimeoutError("read timeout"),
        TimeoutError("socket timeout"),
        ConnectionResetError("peer reset"),
        OSError("generic os error"),
        ValueError("malformed url"),
        HTTPError("http://x", 500, "server error", hdrs=None, fp=None),  # type: ignore[arg-type]
    ],
)
def test_http_get_json_returns_none_on_every_error_class(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    """`_http_get_json` advertises 'returns None on any error'. If a
    new exception class slips through, an `Auto(...)` user gets a hard
    failure on the network instead of a silent fallback."""
    import urllib.request

    def boom(*_a: Any, **_kw: Any) -> Any:
        raise exc

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert _http_get_json("http://example.invalid/x", timeout=0.01) is None


def test_http_get_json_returns_none_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 response with non-JSON body must NOT raise — return None
    so the caller falls back to the static table."""
    import urllib.request

    class _FakeResp:
        def read(self) -> bytes:
            return b"<html>not json</html>"

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_a: Any) -> None:
            return None

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_kw: _FakeResp())
    assert _http_get_json("http://example.invalid/x") is None


def test_http_get_json_returns_none_on_non_utf8_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some misconfigured servers return latin-1 bytes with a JSON
    content-type. Must NOT raise UnicodeDecodeError to the caller."""
    import urllib.request

    class _FakeResp:
        def read(self) -> bytes:
            return b"\xff\xfe\xfd not valid utf-8"

        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_a: Any) -> None:
            return None

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_kw: _FakeResp())
    assert _http_get_json("http://example.invalid/x") is None


# ---------------------------------------------------------------------------
# Optional dependency: boto3
# ---------------------------------------------------------------------------


def test_live_aws_returns_empty_when_boto3_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`live_aws._fetch_aws_spot_quotes` must return ``[]`` when
    ``boto3`` is not installed. The cluster of users who don't use AWS
    should never need to install boto3 just to call ``lookup_cheapest``."""
    real_import = builtins.__import__

    def faux_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "boto3" or name.startswith("boto3.") or name == "botocore.exceptions":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", faux_import)
    quotes = live_aws._fetch_aws_spot_quotes("A100", regions=["us-east-1"])
    assert quotes == [], f"expected empty list with boto3 missing; got {quotes!r}"


def test_lookup_cheapest_works_with_live_disabled_and_no_optionals() -> None:
    """The default code path (`allow_live=False`) consults only the
    static table — must NEVER touch the network or import optional
    deps. Pin: a known GPU returns a quote with positive price."""
    quote = lookup_cheapest("A100", allow_live=False)
    assert quote is not None
    assert quote.hourly_usd > 0
    assert quote.gpu_family.lower() == "a100"


def test_lookup_cheapest_returns_none_for_unknown_gpu_family() -> None:
    """An unknown GPU family must return ``None`` rather than raising
    or returning some arbitrary 'closest' match."""
    assert lookup_cheapest("DEFINITELY-NOT-A-REAL-GPU-12345") is None


def test_lookup_cheapest_does_not_raise_when_live_fetch_explodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if `fetch_live` somehow raises (it shouldn't, but a future
    refactor might forget the contract), `lookup_cheapest` must
    swallow it and fall back to the static table — `Auto(...)` users
    should never see a pricing exception in their pipeline run."""
    import ophelian.pricing.lookup as lookup_mod

    def boom(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("live fetch blew up")

    monkeypatch.setattr(lookup_mod, "fetch_live", boom)
    quote = lookup_mod.lookup_cheapest("A100", allow_live=True)
    assert quote is not None, "static fallback failed when live fetch raised"
    assert quote.hourly_usd > 0
