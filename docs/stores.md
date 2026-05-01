# Stores

An **`ArtifactStore`** is the durable surface where step outputs and
checkpoints live. Ophelian ships four implementations and treats them
all interchangeably through the
`ophelian.stores.base.ArtifactStore` Protocol.

| Store | Backend | Extras |
|---|---|---|
| `LocalArtifactStore` | filesystem | core |
| `S3ArtifactStore` | AWS S3 | `pip install 'ophelian[aws]'` |
| `GCSArtifactStore` | Google Cloud Storage | `pip install 'ophelian[gcp]'` |
| `AzureBlobArtifactStore` | Azure Blob Storage | `pip install 'ophelian[azure]'` |

## Protocol

```python
class ArtifactStore(Protocol):
    def put(self, key: str, src: Path) -> str: ...
    def get(self, key: str, dst: Path) -> Path: ...
    def put_bytes(self, key: str, data: bytes) -> str: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str = "") -> Iterable[str]: ...
    def uri(self, key: str) -> str: ...
```

That's it. Everything in the framework — artifact persistence,
upstream materialisation, checkpoint upload, resume — goes through
exactly these eight methods.

## Picking a store

You almost never instantiate a store directly. The provider chooses
the right one based on the env:

| Env | Store |
|---|---|
| `Standalone` | `LocalArtifactStore` |
| `AWS` | `S3ArtifactStore` |
| `GCP` | `GCSArtifactStore` |
| `Azure` | `AzureBlobArtifactStore` |
| `Auto` | the one for whichever provider won |

You only need to touch the store directly if you're writing a custom
provider or reading artifacts from another tool.

## Reading from another cloud's URI

`Data(source=...)` recognises `s3://`, `gs://`, and
`azure://<account>/<container>/<path>` URIs and routes through the
matching store. You do not need to match the store to the env — a
pipeline running on AWS can read a `gs://` dataset, as long as
`google-cloud-storage` is installed and credentials are set up.

## URI conventions

- `s3://<bucket>/<prefix>/<run_id>/<step>/<file>`
- `gs://<bucket>/<prefix>/<run_id>/<step>/<file>`
- `azure://<account>/<container>/<prefix>/<run_id>/<step>/<file>`

Every store's `uri(key)` returns a string in the matching scheme so
the summary table stays copy-pasteable.
