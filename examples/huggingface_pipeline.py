"""HuggingFace quickstart (advanced — requires network + transformers).

Unlike the other examples, this one is **not** runnable on a fresh checkout
without internet access and the ``transformers`` extra installed::

    pip install -e '.[huggingface]'

It is included as documentation of the intended pipeline shape; CI does not
exercise it end-to-end.
"""

from __future__ import annotations

from ophelian import Data, Deploy, Pipeline, Standalone, Train

pipe = Pipeline(
    [
        Data(name="alpaca", source="hf://yahma/alpaca-cleaned", format="huggingface"),
        Train(
            name="ft",
            framework="huggingface",
            model="sshleifer/tiny-gpt2",
            data="alpaca",
            epochs=1,
        ),
        Deploy(name="serve", model="ft", port=8000),
    ],
    name="huggingface-quickstart",
)


if __name__ == "__main__":
    pipe.run(env=Standalone(local=True))
