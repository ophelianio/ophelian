"""Adapter for HuggingFace transformers (LLM/text fine-tuning)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from ophelian.models.base import ModelAdapter, register_adapter


@register_adapter
class HuggingFaceAdapter(ModelAdapter):
    framework: ClassVar[str] = "huggingface"

    def train(
        self,
        *,
        model: str,
        data: Any,
        hyperparameters: dict[str, Any],
        epochs: int | None,
        batch_size: int | None,
        resume_from: Path | None = None,
        checkpoint_dir: Path | None = None,
    ) -> Any:
        del batch_size
        # The HuggingFace ``Trainer`` already supports resume via
        # ``output_dir``; if the caller pinned a checkpoint_dir we route
        # ``output_dir`` there so intermediate state survives spot
        # interruption, and we forward ``resume_from`` to ``trainer.train``
        # when it points at an existing checkpoint folder.
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(model)
        net = AutoModelForCausalLM.from_pretrained(model, **hyperparameters)
        if data is None:
            return {"model": net, "tokenizer": tokenizer}
        out_dir = str(checkpoint_dir) if checkpoint_dir is not None else "/tmp/ophelian-hf"
        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs or 1,
            per_device_train_batch_size=hyperparameters.get("per_device_train_batch_size", 4),
            logging_steps=10,
            save_steps=hyperparameters.get("save_steps", 500),
        )
        trainer = Trainer(model=net, args=args, train_dataset=data, tokenizer=tokenizer)
        trainer.train(resume_from_checkpoint=str(resume_from) if resume_from is not None else None)
        return {"model": net, "tokenizer": tokenizer}

    def save(self, model: Any, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(model, dict):
            model["model"].save_pretrained(path)
            model["tokenizer"].save_pretrained(path)
        else:
            model.save_pretrained(path)
        return path

    def load(self, path: Path) -> Any:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        net = AutoModelForCausalLM.from_pretrained(path)
        return {"model": net, "tokenizer": tokenizer}

    def predict(self, model: Any, payload: Any) -> Any:
        net = model["model"] if isinstance(model, dict) else model
        tokenizer = model.get("tokenizer") if isinstance(model, dict) else None
        if tokenizer is None:
            raise RuntimeError(
                "HuggingFaceAdapter.predict requires a tokenizer alongside the model"
            )
        text = payload["text"] if isinstance(payload, dict) else str(payload)
        inputs = tokenizer(text, return_tensors="pt")
        outputs = net.generate(**inputs, max_new_tokens=64)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
