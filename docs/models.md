# Models

Ophelian doesn't reimplement ML libraries — it adapts the ones you
already use. A `Train` / `Tune` / `Eval` node accepts a `model=`
string of the form `"<library>.<class.path>"` and Ophelian looks up
the right adapter at runtime through the `ophelian.adapters`
entry-point group.

## Built-in adapters

| Adapter | Triggered by `model=` prefix | Extras |
|---|---|---|
| `pytorch` | `torch.*` | `pip install 'ophelian[pytorch]'` |
| `huggingface` | `meta-llama/...`, `mistralai/...`, etc. (any HF repo id) | `pip install 'ophelian[huggingface]'` |
| `sklearn` | `sklearn.*` | `pip install 'ophelian[sklearn]'` |
| `xgboost` | `xgboost.*` | `pip install 'ophelian[xgboost]'` |

## Examples

```python
from ophelian import Pipeline, Train

# scikit-learn
Pipeline([Train(
    model="sklearn.ensemble.RandomForestClassifier",
    data="iris", target="species",
    hyperparams={"n_estimators": 200, "max_depth": 8},
)])

# XGBoost
Pipeline([Train(
    model="xgboost.XGBClassifier",
    data="titanic", target="survived",
    hyperparams={"n_estimators": 400, "tree_method": "hist"},
)])

# Hugging Face fine-tune
Pipeline([Train(
    model="meta-llama/Llama-3.2-1B",
    data="s3://bucket/dataset.jsonl",
    epochs=3, lr=2e-5,
)])
```

## Writing a custom adapter

Adapters implement
`ophelian.models.base.ModelAdapter` and register through the
`ophelian.adapters` entry-point group. From your distribution's
`pyproject.toml`:

```toml
[project.entry-points."ophelian.adapters"]
mylib = "my_pkg.adapters:MyLibAdapter"
```

Once installed, `Train(model="mylib.MyClass", ...)` will route to your
adapter without any change to the framework.
