import os

from modelscope import snapshot_download

_cache_root = os.environ.get(
    "MS_CACHE_HOME",
    os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub"),
)
os.makedirs(_cache_root, exist_ok=True)

model_dir = snapshot_download("Qwen/Qwen3-VL-4B-Instruct", cache_dir=_cache_root)
print(model_dir)
