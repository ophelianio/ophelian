"""Fine-tune a HuggingFace causal LM on AWS GPU instances.

Run with::

    OPHELIAN_AWS_REGION=us-east-1 \
        OPHELIAN_AWS_BUCKET=my-ophelian-bucket \
        python examples/aws/huggingface_llm.py

This is the heaviest of the three demos — defaults to ``g5.2xlarge`` (A10G
24 GB) which is enough for LoRA fine-tuning on a small Llama / Mistral
chat model. Bump to ``p4d.24xlarge`` for full fine-tuning of larger
models.

The pipeline:

* Streams an instruction-tuning dataset from S3
* Fine-tunes the base model with HuggingFace ``Trainer``
* Evaluates on a held-out split
* Deploys behind a FastAPI ``/predict`` endpoint with 3 replicas
"""

from __future__ import annotations

import os

from ophelian import AWS, Data, Deploy, Eval, Pipeline, Train

REGION = os.environ.get("OPHELIAN_AWS_REGION", "us-east-1")
BUCKET = os.environ["OPHELIAN_AWS_BUCKET"]
DATASET = os.environ.get(
    "OPHELIAN_AWS_DATASET",
    f"s3://{BUCKET}/datasets/instruct/train.jsonl",
)
EVAL_DATASET = os.environ.get(
    "OPHELIAN_AWS_EVAL_DATASET",
    f"s3://{BUCKET}/datasets/instruct/eval.jsonl",
)


pipe = Pipeline(
    [
        Data(name="train_ds", source=DATASET, format="jsonl", split="train"),
        Data(name="eval_ds", source=EVAL_DATASET, format="jsonl", split="eval"),
        Train(
            name="finetune",
            framework="huggingface",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            data="train_ds",
            hyperparameters={
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "learning_rate": 2e-4,
                "weight_decay": 0.0,
            },
            epochs=3,
            batch_size=4,
        ),
        Eval(name="ev", model="finetune", data="eval_ds", metrics=("perplexity",)),
        Deploy(name="serve", model="finetune", port=8080, replicas=3),
    ],
    name="hf-llm-aws",
)


def main() -> None:
    env = AWS(
        region=REGION,
        instance=os.environ.get("OPHELIAN_AWS_INSTANCE", "g5.2xlarge"),
        spot=os.environ.get("OPHELIAN_AWS_SPOT", "1") == "1",
        artifact_bucket=BUCKET,
        iam_role=os.environ.get("OPHELIAN_AWS_INSTANCE_PROFILE"),
        runtime_extras=("huggingface", "pytorch"),
        timeout_seconds=24 * 60 * 60,
        resume_run_id=os.environ.get("OPHELIAN_AWS_RESUME_RUN_ID") or None,
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
