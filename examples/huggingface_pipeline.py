"""HuggingFace quickstart — fine-tune a tiny GPT-2 on an inline corpus.

This runs locally with the ``huggingface`` extra installed::

    pip install -e '.[huggingface]'
    python examples/huggingface_pipeline.py

The only network access is a one-time download of the ~5 MB
``sshleifer/tiny-gpt2`` model from the Hub; the training data is inline,
so no dataset download is needed. To train on a Hub dataset instead, use
``source="hf://<dataset-id>"`` (needs the ``datasets`` package) in place
of ``options={"texts": [...]}``.
"""

from __future__ import annotations

from ophelian import Data, Deploy, Pipeline, Standalone, Train

CORPUS = [
    "Ophelian runs the same pipeline locally and on every cloud.",
    "Declare a pipeline once, then pick an env to run it anywhere.",
    "The cost router picks the cheapest GPU across AWS, GCP, and Azure.",
    "Every deploy step serves /health and /predict over HTTP.",
    "Checkpointing makes spot and preemptible instances safe to use.",
    "Adapters cover sklearn, xgboost, pytorch, and huggingface.",
]

pipe = Pipeline(
    [
        Data(
            name="corpus",
            source="inline://",
            format="huggingface",
            options={"texts": CORPUS},
        ),
        Train(
            name="ft",
            framework="huggingface",
            model="sshleifer/tiny-gpt2",
            data="corpus",
            epochs=1,
            hyperparameters={"per_device_train_batch_size": 2, "max_length": 32},
        ),
        Deploy(name="serve", model="ft", port=8000),
    ],
    name="huggingface-quickstart",
)


if __name__ == "__main__":
    result = pipe.run(env=Standalone(local=True))
    for step in result.steps:
        print(f"{step.name}: {step.status} {step.error or ''}")
