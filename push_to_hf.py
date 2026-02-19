"""
push_to_hf.py: Publish dataset + Gradio Space to HuggingFace.

Reads HF_TOKEN and HF_ORG from .env, substitutes {{HF_ORG}} placeholders
in README.md / app.py at upload time (source files stay clean).

Usage:
    python push_to_hf.py
"""

import os
import pathlib
import tempfile
import shutil

# ── Load .env ──────────────────────────────────────────────────────────────────

def load_env(path=None):
    if path is None:
        # Walk up to find .env at project root (two levels above hf_dataset/)
        path = pathlib.Path(__file__).parent.parent.parent / ".env"
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    except FileNotFoundError:
        raise SystemExit(f"No .env file found at {path}. Copy .env.example and fill it in.")
    return env

env = load_env()
HF_TOKEN = env.get("HF_TOKEN", "")
HF_ORG   = env.get("HF_ORG", "")

if not HF_TOKEN or HF_TOKEN == "hf_your_token_here":
    raise SystemExit("Set HF_TOKEN in .env")
if not HF_ORG or HF_ORG == "your-org-slug":
    raise SystemExit("Set HF_ORG in .env")

# ── Auth ───────────────────────────────────────────────────────────────────────

from huggingface_hub import HfApi, login  # noqa: E402

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()
print(f"Logged in as: {api.whoami()['name']}")

# ── Helpers ────────────────────────────────────────────────────────────────────

ROOT  = pathlib.Path(__file__).parent          # nyc_clinic_infra/
SPACE = ROOT / "explorer"                      # nyc_clinic_infra/explorer/

DATASET_REPO = f"{HF_ORG}/nyc-clinic-ai-infrastructure"
SPACE_REPO   = f"{HF_ORG}/nyc-clinic-ai-infra-map"


def substitute(text: str) -> str:
    """Replace {{HF_ORG}} placeholders with the real org slug."""
    return text.replace("{{HF_ORG}}", HF_ORG)


def make_substituted_copy(src_dir: pathlib.Path, files_to_patch: list[str]) -> pathlib.Path:
    """
    Copy src_dir to a temp directory, substituting placeholders in listed files.
    Returns the temp dir path (caller must clean up).
    """
    tmp = pathlib.Path(tempfile.mkdtemp())
    shutil.copytree(src_dir, tmp / "upload", dirs_exist_ok=True)
    for rel_path in files_to_patch:
        f = tmp / "upload" / rel_path
        if f.exists():
            f.write_text(substitute(f.read_text()))
    return tmp / "upload"

# ── Dataset repo ───────────────────────────────────────────────────────────────

print(f"\n── Dataset: {DATASET_REPO}")
api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True, private=False)

tmp_dataset = make_substituted_copy(ROOT / "pipeline", ["README.md"])
try:
    api.upload_folder(
        folder_path=str(tmp_dataset),
        repo_id=DATASET_REPO,
        repo_type="dataset",
        ignore_patterns=[
            "requested/**", "cache/**", "**/__pycache__/**", "**/*.pyc",
            ".env", ".env.example", ".env.local",
        ],
        commit_message="Update dataset scripts and README",
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "pipeline" / "outputs" / "nyc_clinic_infrastructure.csv"),
        path_in_repo="nyc_clinic_infrastructure.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="Update dataset CSV",
    )
finally:
    shutil.rmtree(tmp_dataset.parent)

print(f"  ✓ https://huggingface.co/datasets/{DATASET_REPO}")

# ── Gradio Space ───────────────────────────────────────────────────────────────

print(f"\n── Space: {SPACE_REPO}")
api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)

tmp_space = make_substituted_copy(SPACE, ["README.md", "app.py"])
try:
    api.upload_folder(
        folder_path=str(tmp_space),
        repo_id=SPACE_REPO,
        repo_type="space",
        ignore_patterns=[".env", ".env.example", ".env.local"],
        commit_message="Update Gradio Space",
    )
finally:
    shutil.rmtree(tmp_space.parent)

print(f"  ✓ https://huggingface.co/spaces/{SPACE_REPO}")

# ── Done ───────────────────────────────────────────────────────────────────────

print("\nAll done.")
