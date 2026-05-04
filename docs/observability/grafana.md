# Grafana dashboard for serve metrics

Ophelian ships a curated Grafana dashboard at
[`dashboards/grafana/serve.json`](https://github.com/ophelianio/ophelian/blob/main/dashboards/grafana/serve.json)
that visualizes every metric the FastAPI runtime exports under the
`ophelian.serve.*` namespace. Import it once, point it at a Prometheus
datasource scraping your model's `/metrics` endpoint, and you're done.

## What it shows

The dashboard has three rows:

**Traffic**

- **Request rate by status class** — stacked `2xx` / `4xx` / `5xx`
  rates so you can spot error spikes immediately.
- **In-flight + queue depth** — concurrent requests being served, plus
  the optional queue gauge for routes that wire one up.

**Latency**

- **Request latency heatmap** — full HTTP-in-to-HTTP-out distribution.
- **Inference duration heatmap** — model `.predict()` only. Subtract
  this from total latency to see your serialization / framework
  overhead.
- **Latency percentiles** — p50 / p95 / p99 total plus a p95 inference-
  only line so the gap between the two is obvious.

**Tokens & idle**

- **Token throughput** — input and output tokens per second. Only
  populated when the adapter exposes a token-usage shape (e.g.
  OpenAI-compatible `response.usage`).
- **Idle-time fraction** — fraction of wall-clock time the endpoint
  sat with zero in-flight requests. Drives autoscaler reclaim-on-idle
  decisions.

A `Route` template variable lets you slice every panel by HTTP route
(or pick `All` for an aggregated view).

## Prerequisites

1. **Install the `[otel]` extra** so the OTel SDK + Prometheus exporter
   are available:

    ```bash
    pip install 'ophelian[otel]'
    ```

2. **Enable the `/metrics` endpoint** when building your serve app:

    ```python
    from ophelian.runtime.fastapi_runtime import build_app

    app = build_app(
        framework="sklearn",
        model_path="/path/to/model.joblib",
        enable_prometheus=True,
    )
    ```

    The framework name is whatever your model adapter is registered
    under (`sklearn`, `xgboost`, `pytorch`, …). The `FastAPIRuntime`
    class accepts the same `enable_prometheus=True` flag if you prefer
    that entrypoint.

3. **Wire your Prometheus server to scrape it** (a minimal job):

    ```yaml
    scrape_configs:
      - job_name: ophelian-serve
        metrics_path: /metrics
        static_configs:
          - targets: ["your-serve-host:8000"]
    ```

## Importing the dashboard

In Grafana 10+:

1. **Dashboards → New → Import**.
2. Paste the contents of `dashboards/grafana/serve.json` (or upload the
   file directly).
3. When prompted, select the Prometheus datasource that scrapes your
   serve `/metrics` endpoint.
4. Click **Import**.

The dashboard's UID is `ophelian-serve`, so you can also provision it
declaratively under `/etc/grafana/provisioning/dashboards/`:

```yaml
apiVersion: 1
providers:
  - name: ophelian
    folder: Ophelian
    type: file
    options:
      path: /var/lib/grafana/dashboards/ophelian
```

…and drop `serve.json` into that path.

## Metric reference

| Panel | Prometheus metric | OTel name |
| --- | --- | --- |
| Request rate | `ophelian_serve_requests_total{http_status_class=...}` | `ophelian.serve.requests` |
| In-flight | `ophelian_serve_inflight{http_route=...}` | `ophelian.serve.inflight` |
| Queue depth | `ophelian_serve_queue_depth{http_route=...}` | `ophelian.serve.queue.depth` |
| Total latency | `ophelian_serve_latency_seconds_bucket` | `ophelian.serve.latency` |
| Inference duration | `ophelian_serve_inference_duration_seconds_bucket` | `ophelian.serve.inference.duration` |
| Tokens in / out | `ophelian_serve_tokens_in_total`, `ophelian_serve_tokens_out_total` | `ophelian.serve.tokens.in`, `ophelian.serve.tokens.out` |
| Idle fraction | `ophelian_serve_idle_seconds_total` (rate) | `ophelian.serve.idle.seconds` |

The Prometheus exporter applies the standard OTel-to-Prometheus name
translation: dots become underscores, histograms gain
`_bucket` / `_sum` / `_count` suffixes, counters gain `_total`, and
`http.*` attributes become `http_*` labels.

## Customizing

The JSON is meant to be edited. Common tweaks:

- **Change SLO thresholds** — edit the `thresholds` block on the
  `In-flight + queue depth` and `Idle-time fraction` panels.
- **Add a route filter on import** — the `route` template variable
  already supports multi-select; restrict it via the dashboard URL,
  e.g. `&var-route=/predict`.
- **Add per-tenant rows** — if your subscribers tag events with a
  `tenant_id` from `Pipeline.context` (see the lifecycle events bus),
  duplicate the existing panels and add `tenant_id` to the
  `sum by (...)` clauses.

## Troubleshooting

- **No data for any panel** — confirm `enable_prometheus=True` was
  passed to `build_app` and that `/metrics` returns text on
  `curl http://your-host:8000/metrics`.
- **Histograms are empty but counters work** — the OTel SDK only
  exposes histogram buckets when at least one sample has been
  recorded; send a few requests and re-check.
- **Token panels stay flat** — that's expected for adapters that
  don't return a token-usage shape (most non-LLM models). The
  framework only records token counters when
  `extract_token_usage(...)` returns a non-empty dict.
