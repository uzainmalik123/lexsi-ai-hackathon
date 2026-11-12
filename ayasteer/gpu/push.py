"""Upload the results folder to a public HF dataset repo (idempotent)."""
from __future__ import annotations

from pathlib import Path


def push_results(repo_id: str, path: Path, token: str | None = None, private: bool = False) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    info = api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(path),
                             path_in_repo=Path(path).name, commit_message=f"upload {Path(path).name}")
    return str(info)
