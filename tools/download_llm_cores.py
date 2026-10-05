"""Download only the three approved model checkpoints, never dataset files."""
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, get_token

TOKEN = get_token()  # Preserve existing user authorization when relocating caches.

ROOT = Path(__file__).resolve().parents[1]/"checkpoints"
os.environ.setdefault("HF_HOME", str(ROOT/".hf_home"))
os.environ.setdefault("HF_XET_CACHE", str(ROOT/".xet"))

MODELS = {
    "google/gemma-3-1b-it": "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}


def main():
    completed = []
    for repo, revision in MODELS.items():
        print(f"Downloading model {repo} at {revision}", flush=True)
        info = HfApi(token=TOKEN).model_info(repo, revision=revision, files_metadata=True)
        folder = ROOT/repo.split("/")[-1]
        snapshot_download(repo, revision=revision, local_dir=folder, max_workers=4, token=TOKEN,
                          allow_patterns=["*.json", "*.model", "*.safetensors", "*.jinja", "merges.txt", "vocab.txt", "LICENSE*", "README.md"])
        files = []
        for remote in info.siblings:
            local = folder/remote.rfilename
            if local.is_file():
                if remote.size is not None and local.stat().st_size != remote.size:
                    raise ValueError(f"Downloaded size mismatch: {repo}/{remote.rfilename}")
                files.append({"file": remote.rfilename, "bytes": local.stat().st_size,
                              "sha256": remote.lfs.sha256 if remote.lfs else None})
        record = {"repo": repo, "revision": revision, "path": str(folder), "files": files}
        (folder/"download_provenance.json").write_text(json.dumps(record, indent=2))
        completed.append(record)
        (ROOT/"download_manifest.json").write_text(json.dumps(completed, indent=2))
        print(f"Ready: {repo} ({sum(f['bytes'] for f in files):,} bytes)", flush=True)


if __name__ == "__main__":
    main()
