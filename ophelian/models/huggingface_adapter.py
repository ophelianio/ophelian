"""Adapter for HuggingFace transformers (LLM/text fine-tuning)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, ClassVar

from ophelian.models.base import ModelAdapter, register_adapter


@register_adapter
class HuggingFaceAdapter(ModelAdapter):
    framework: ClassVar[str] = "huggingface"

    # Hyperparameter keys that control training/tokenisation rather than
    # model construction, so they must be kept out of
    # ``AutoModelForCausalLM.from_pretrained(**model_kwargs)``.
    _CONTROL_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"per_device_train_batch_size", "save_steps", "max_length"}
    )

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

        # Revision pinning is the caller's responsibility: ``model`` is a
        # user-supplied repo ID (or local path) and Ophelian must not silently
        # override it. Document that users should pin via env vars on the Hub.
        tokenizer = AutoTokenizer.from_pretrained(model)  # nosec B615
        # ``hyperparameters`` is a mixed bag: some keys configure training
        # (batch size, checkpoint cadence) or tokenisation (max_length), the
        # rest are genuine model-construction kwargs (dtype, ...). Only the
        # latter may reach ``from_pretrained`` — forwarding a training key
        # like ``per_device_train_batch_size`` raises a TypeError there.
        model_kwargs = {k: v for k, v in hyperparameters.items() if k not in self._CONTROL_KEYS}
        net = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)  # nosec B615
        if data is None:
            return {"model": net, "tokenizer": tokenizer}
        out_dir = (
            str(checkpoint_dir)
            if checkpoint_dir is not None
            else str(Path(tempfile.gettempdir()) / "ophelian-hf")
        )
        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs or 1,
            per_device_train_batch_size=hyperparameters.get("per_device_train_batch_size", 4),
            logging_steps=10,
            save_steps=hyperparameters.get("save_steps", 500),
        )
        train_dataset = self._as_train_dataset(data, tokenizer, hyperparameters)
        # ``Trainer``'s tokenizer kwarg was renamed ``processing_class`` in
        # transformers 4.46 and the old ``tokenizer`` name was removed in
        # 5.x. Pass whichever the installed version accepts so the adapter
        # works across the supported transformers range.
        import inspect

        trainer_kwargs: dict[str, Any] = {
            "model": net,
            "args": args,
            "train_dataset": train_dataset,
        }
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:  # pragma: no cover - exercised only on transformers < 4.46
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = Trainer(**trainer_kwargs)
        trainer.train(resume_from_checkpoint=str(resume_from) if resume_from is not None else None)
        return {"model": net, "tokenizer": tokenizer}

    @staticmethod
    def _as_train_dataset(data: Any, tokenizer: Any, hyperparameters: dict[str, Any]) -> Any:
        """Turn the loader payload into something ``Trainer`` accepts.

        The standalone provider round-trips datasets through JSON, so the
        ``huggingface`` loader hands us a plain
        ``{"texts": [str, ...]}`` dict rather than a live
        ``datasets.Dataset``. We tokenise those strings here into a small
        in-memory causal-LM dataset (labels == input_ids). Any other
        ``data`` (e.g. a Dataset built by a cloud driver) is passed
        through untouched.
        """
        if not (isinstance(data, dict) and "texts" in data):
            return data

        texts = [str(t) for t in data["texts"]]
        # GPT-2-style tokenizers ship without a pad token; reuse EOS so
        # padding works without resizing embeddings.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_length = int(hyperparameters.get("max_length", 64))
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        import torch

        class _TextDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
            def __len__(self) -> int:
                return len(texts)

            def __getitem__(self, index: int) -> dict[str, Any]:
                input_ids = torch.tensor(encoded["input_ids"][index])
                return {
                    "input_ids": input_ids,
                    "attention_mask": torch.tensor(encoded["attention_mask"][index]),
                    "labels": input_ids.clone(),
                }

        return _TextDataset()

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

        # ``path`` is a local directory written by ``save`` above; no
        # network download happens here, so revision pinning does not apply.
        tokenizer = AutoTokenizer.from_pretrained(path)  # nosec B615
        net = AutoModelForCausalLM.from_pretrained(path)  # nosec B615
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
