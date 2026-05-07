"""Download Qwen3-30B-A3B-Base config files (no weights)."""
import os, shutil
from huggingface_hub import hf_hub_download

target_dir = "/data/home/scyb091/run/models/Qwen3-30B-A3B-Base"
os.makedirs(target_dir, exist_ok=True)

files = ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"]
for f in files:
    try:
        path = hf_hub_download(
            repo_id="Qwen/Qwen3-30B-A3B-Base",
            filename=f,
            cache_dir="/data/home/scyb091/run/.hf_cache",
        )
        shutil.copy(path, target_dir + "/" + f)
        print(f"OK: {f}")
    except Exception as e:
        print(f"SKIP: {f} - {e}")

print("DONE:", os.listdir(target_dir))
with open(os.path.join(target_dir, "config.json")) as cf:
    import json
    c = json.load(cf)
    print("hidden_size:", c.get("hidden_size"), "num_layers:", c.get("num_hidden_layers"), "num_experts:", c.get("num_experts"))
