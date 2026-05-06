"""Ophelian end-to-end demo.

Runs as a plain Python script (PyCharm / `python examples/demo.py`) and
also cell-by-cell from a Jupyter notebook (see ``examples/demo.ipynb``).

Requirements
------------
The demo needs Ophelian **1.1.0+** for the per-provider provenance
(``data_quality``) and ``require_live`` strict mode shown in sections
4 and 5. Until 1.1.0 is on PyPI, install the dev branch directly::

    pip install "git+https://github.com/ophelianio/ophelian.git@dev" scikit-learn

If you only have 1.0.1 from PyPI the first three sections still work;
sections 4 and 5 print a "needs 1.1.0+" notice and continue.

Sections
--------
    1. Define a pipeline with the declarative DSL.
    2. Train locally on the Standalone executor (no Docker, no cloud).
    3. Inspect step results, metrics, and the trained-model artifact.
    4. Cost-route the same pipeline across AWS / GCP / Azure with
       ``Auto(...)`` in dry-run mode and print the router's decision,
       per-provider data provenance, and the human-readable explain().
    5. Strict mode: ``Auto(..., require_live=[...])`` raises if a
       provider's price didn't come from a live API call.
"""

from __future__ import annotations

from ophelian import Data, Deploy, Eval, Pipeline, Standalone, Train, __version__
from ophelian.envs import Auto, AutoRouterError
from ophelian.pricing import explain
from ophelian.pricing.quotes import RouterDecision

_HAS_PROVENANCE = "data_quality" in RouterDecision.__dataclass_fields__


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# 1. Define the pipeline — the same object moves from laptop to any cloud.
# ---------------------------------------------------------------------------
banner("1. Declarative pipeline")

pipe = Pipeline(
    [
        Data(
            name="iris",
            source="synthetic://iris",
            format="synthetic",
            options={"name": "iris"},
        ),
        Train(
            name="trainer",
            framework="sklearn",
            model="sklearn.linear_model.LogisticRegression",
            data="iris",
            hyperparameters={"max_iter": 200},
        ),
        Eval(
            name="scorer",
            model="trainer",
            data="iris",
            metrics=["accuracy", "f1_macro"],
        ),
        Deploy(name="serve", model="trainer", port=8080),
    ],
    name="ophelian-demo",
)
print(f"Pipeline   : {pipe.name}")
print(f"Steps      : {[node.name for node in pipe.steps]}")


# ---------------------------------------------------------------------------
# 2. Run locally on Standalone — the laptop executor.
# ---------------------------------------------------------------------------
banner("2. Local training run (Standalone)")

result = pipe.run(env=Standalone(local=True))
print(f"Succeeded  : {result.succeeded}")


# ---------------------------------------------------------------------------
# 3. Inspect what the run produced.
# ---------------------------------------------------------------------------
banner("3. Step results")

for step in result.steps:
    print(f"  - {step.name:<8} {step.status}")

print(f"\nEval metrics: {result.step('scorer').metrics}")

serve_info = result.step("serve").info
print(f"Predict URL : {serve_info.get('predict')}")
print(f"Health URL  : {serve_info.get('health')}")


# ---------------------------------------------------------------------------
# 4. Cost-route the same pipeline across clouds (dry-run, no creds needed).
# ---------------------------------------------------------------------------
banner("4. Auto router — cheapest A100 across AWS / GCP / Azure")
print(f"(running ophelian {__version__})")

router = Auto(
    cheapest_gpu="A100",
    regions=["us-east-1", "us-central1", "eastus"],
    dry_run=True,
    require_credentials=False,
)
decision = router.decision  # type: ignore[attr-defined]
print(f"Picked     : {decision.quote.provider}/{decision.quote.region}")
print(f"Instance   : {decision.quote.instance}")
print(f"Price      : {decision.quote.hourly_usd} USD/h ({'spot' if decision.quote.spot else 'on-demand'})")
print(f"Considered : {len(decision.considered)} quotes")
if _HAS_PROVENANCE:
    print(f"Provenance : {decision.data_quality}")
else:
    print("Provenance : <needs ophelian >= 1.1.0 — see install instructions at top>")
print()
print(explain(decision))


# ---------------------------------------------------------------------------
# 5. Strict mode — fail loudly when a price isn't from a live API.
# ---------------------------------------------------------------------------
banner("5. Strict mode: require_live=[...]")

if not _HAS_PROVENANCE:
    print("Skipped: require_live needs ophelian >= 1.1.0.")
    print('Install with: pip install "git+https://github.com/ophelianio/ophelian.git@dev"')
else:
    try:
        Auto(
            cheapest_gpu="A100",
            regions=["us-east-1", "us-central1", "eastus"],
            dry_run=True,
            require_credentials=False,
            require_live=["aws", "gcp", "azure"],
            allow_live=False,  # force everything to 'static' so the check trips
        )
    except AutoRouterError as exc:
        print(f"AutoRouterError raised (as expected):\n  {exc}")


banner("Done")
print("Next: swap Standalone() for AWS(), GCP(), Azure(), or Auto() —")
print("the same Pipeline object runs on any of them.")
