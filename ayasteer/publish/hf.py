"""Hugging Face publishing helpers (all repos public, idempotent)."""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

USER = os.environ.get("AYASTEER_HF_USER", "uzainahmed")
REPOS = {
    "benchmark": f"{USER}/tiny-aya-multilingual-safety-bench",
    "runs": f"{USER}/tiny-aya-mlsteer-runs",
    "direction_en": f"{USER}/tiny-aya-global-refusal-direction-en",
    "direction_multi": f"{USER}/tiny-aya-global-refusal-direction-multi",
    "space": f"{USER}/tiny-aya-multilingual-steering",
}


def _api(token: str | None = None):
    from huggingface_hub import HfApi
    return HfApi(token=token or os.environ.get("HF_TOKEN"))


def _render_card(card_path: Path, **vars) -> str:
    text = Path(card_path).read_text(encoding="utf-8")
    for k, v in vars.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def publish_dataset(repo_id: str, local_dir: Path, card_path: Path, token: str | None = None, **card_vars) -> str:
    api = _api(token)
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(local_dir), commit_message="upload data",
                      ignore_patterns=["*.pyc", "__pycache__/*", "curatorkit/.checkpoints/*"])
    api.upload_file(path_or_fileobj=_render_card(card_path, **card_vars).encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset", commit_message="dataset card")
    return f"https://huggingface.co/datasets/{repo_id}"


def publish_redacted_runs(repo_id: str, public_dir: Path, card_path: Path, token: str | None = None, **card_vars) -> str:
    """Replace ``results/`` in the (private) runs repo with the redacted copy, squash the history so the
    raw outputs Kaggle pushed are not reachable from any commit, and only then make the repo public."""
    api = _api(token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(public_dir), path_in_repo="results",
                      delete_patterns=["*"], commit_message="redacted public copy of results")
    api.upload_file(path_or_fileobj=_render_card(card_path, **card_vars).encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset", commit_message="dataset card")
    api.super_squash_history(repo_id=repo_id, repo_type="dataset", commit_message="redacted results")
    api.update_repo_settings(repo_id, repo_type="dataset", private=False)
    return f"https://huggingface.co/datasets/{repo_id}"


def publish_model(repo_id: str, direction_dir: Path, card_path: Path, token: str | None = None, **card_vars) -> str:
    api = _api(token)
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(direction_dir), commit_message="steering direction")
    api.upload_file(path_or_fileobj=_render_card(card_path, **card_vars).encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model", commit_message="model card")
    return f"https://huggingface.co/{repo_id}"


def publish_space(repo_id: str, space_dir: Path, token: str | None = None) -> str:
    api = _api(token)
    api.create_repo(repo_id, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(space_dir), commit_message="static replay")
    return f"https://huggingface.co/spaces/{repo_id}"


def create_collection(title: str, items: list[tuple[str, str]], description: str = "", token: str | None = None) -> str:
    """items: [(repo_type, repo_id)]. Returns the collection URL. Reuses an existing collection with the same title."""
    api = _api(token)
    owner = api.whoami()["name"]
    existing = {c.title: c for c in api.list_collections(owner=owner)}
    coll = existing.get(title) or api.create_collection(title=title, description=description, private=False)
    coll = api.get_collection(coll.slug)
    have = {(it.item_type, it.item_id) for it in coll.items}
    for repo_type, repo_id in items:
        if (repo_type, repo_id) not in have:
            api.add_collection_item(coll.slug, item_id=repo_id, item_type=repo_type, exists_ok=True)
    return f"https://huggingface.co/collections/{coll.slug}"
