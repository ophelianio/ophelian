"""LoRA-fine-tune Llama on whichever cloud is cheapest right now.

The demo is deliberately the smallest possible thing that actually
works: it picks the cheapest A100 across AWS / GCP / Azure with the
:func:`Auto` router, fine-tunes a 7-B instruct model with LoRA on a
JSONL instruction dataset, evaluates perplexity, and exposes a
``/predict`` endpoint behind FastAPI.

Run it::

    OPHELIAN_DATASET=s3://my-bucket/instruct/train.jsonl \
        OPHELIAN_EVAL_DATASET=s3://my-bucket/instruct/eval.jsonl \
        python examples/llama_finetune.py

Add ``OPHELIAN_DRY_RUN=1`` to print the routing decision without
spinning up infra. Override ``OPHELIAN_GPU=H100`` for full
fine-tuning, or ``OPHELIAN_PROVIDERS=aws,gcp`` to whitelist clouds.
"""

from __future__ import annotations

import os

from ophelian import Auto, Data, Deploy, Eval, Pipeline, Train

DATASET = os.environ.get("OPHELIAN_DATASET", "s3://ophelian-demo/instruct/train.jsonl")
EVAL_DATASET = os.environ.get(
    "OPHELIAN_EVAL_DATASET", "s3://ophelian-demo/instruct/eval.jsonl"
)
MODEL = os.environ.get("OPHELIAN_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
GPU = os.environ.get("OPHELIAN_GPU", "A100")


pipe = Pipeline(
    [
        Data(name="train_ds", source=DATASET, format="jsonl", split="train"),
        Data(name="eval_ds", source=EVAL_DATASET, format="jsonl", split="eval"),
        Train(
            name="finetune",
            framework="huggingface",
            model=MODEL,
            data="train_ds",
            hyperparameters={
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "learning_rate": 2e-4,
            },
            epochs=int(os.environ.get("OPHELIAN_EPOCHS", "3")),
            batch_size=int(os.environ.get("OPHELIAN_BATCH", "4")),
        ),
        Eval(name="ev", model="finetune", data="eval_ds", metrics=("perplexity",)),
        Deploy(name="serve", model="finetune", port=8080, replicas=2),
    ],
    name="llama-finetune",
)


def _csv_env(name: str) -> list[str] | None:
    """Parse a comma-separated env var, returning None when unset/blank."""
    raw = os.environ.get(name)
    if not raw:
        return None
    items = [piece.strip() for piece in raw.split(",") if piece.strip()]
    return items or None


def main() -> None:
    env = Auto(
        cheapest_gpu=GPU,
        providers=_csv_env("OPHELIAN_PROVIDERS"),
        regions=_csv_env("OPHELIAN_REGIONS"),
        spot=os.environ.get("OPHELIAN_SPOT", "1") == "1",
        dry_run=os.environ.get("OPHELIAN_DRY_RUN", "0") == "1",
        require_credentials=os.environ.get("OPHELIAN_REQUIRE_CREDS", "1") == "1",
    )
    result = pipe.run(env=env)
    if result.succeeded:
        info = result.step("serve").info
        print("Endpoint:", info.get("endpoint_url") or info)
    else:
        for step in result.steps:
            print(f"{step.name}: {step.status} {step.error or ''}")


if __name__ == "__main__":
    main()
