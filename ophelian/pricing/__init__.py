"""GPU pricing tables + Auto-router cost lookups.

The :func:`lookup_cheapest` function is the small primitive on which
:func:`ophelian.envs.auto.Auto` builds: given a GPU family and a list
of regions, return the cheapest provider/region/instance triple that
satisfies the request.

We ship a hand-curated **static fallback** table covering AWS, GCP and
Azure for the most-common GPU families (T4, L4, V100, A10G, A100, H100).
The numbers are honest order-of-magnitude on-demand prices captured at
release time and recorded in :data:`STATIC_PRICES_LAST_REVIEW`. They
exist so the router works even when the user's environment has no
network access to live pricing endpoints — there's nothing more
embarrassing than an "ML infra tool" that refuses to start because a
pricing API rate-limited it.

When the optional :pypi:`requests` dependency is available we *also*
support a ``fetch_live`` path that hits the official pricing APIs and
caches results on disk for ``cache_ttl_seconds`` (default: 24 h). The
fetcher is opt-in — :func:`lookup_cheapest` only consults it when the
caller passes ``allow_live=True``.

For callers that need to know **where** each provider's prices came
from (live API, cache hit, static fallback, ...) use the ``*_with_meta``
variants — :func:`fetch_live_with_meta` and
:func:`lookup_cheapest_with_meta` — which return the same data plus a
provenance map. The legacy :func:`fetch_live` and :func:`lookup_cheapest`
functions are thin wrappers that drop the provenance, kept for
backward compatibility.

This module is a re-export shim. The implementation lives in:

* :mod:`ophelian.pricing.quotes` — :class:`PriceQuote`,
  :class:`RouterDecision`, :func:`explain`
* :mod:`ophelian.pricing.static` — :data:`STATIC_PRICES`,
  :data:`STATIC_PRICES_LAST_REVIEW`, :func:`static_quotes`
* :mod:`ophelian.pricing.cache` — on-disk JSON cache helpers
* :mod:`ophelian.pricing.live` — :func:`fetch_live_with_meta`
  coordinator with one submodule per cloud (``aws``, ``azure``,
  ``gcp``)
* :mod:`ophelian.pricing.lookup` — :func:`lookup_cheapest_with_meta`
"""

from __future__ import annotations

from ophelian.pricing.live import fetch_live, fetch_live_with_meta
from ophelian.pricing.lookup import lookup_cheapest, lookup_cheapest_with_meta
from ophelian.pricing.quotes import PriceQuote, RouterDecision, explain
from ophelian.pricing.static import (
    STATIC_PRICES,
    STATIC_PRICES_LAST_REVIEW,
    static_quotes,
)

__all__ = [
    "STATIC_PRICES",
    "STATIC_PRICES_LAST_REVIEW",
    "PriceQuote",
    "RouterDecision",
    "explain",
    "fetch_live",
    "fetch_live_with_meta",
    "lookup_cheapest",
    "lookup_cheapest_with_meta",
    "static_quotes",
]
