from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    dest = Path(args.dest).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Downloaded {args.repo_id} to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
