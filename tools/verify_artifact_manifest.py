#!/usr/bin/env python3
"""Verify published artifact sizes, SHA-256 digests and the exact allowlist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    paths = [row["path"] for row in manifest["files"]]
    if len(paths) != len(set(paths)) or len(paths) != manifest["file_count"]:
        raise SystemExit("Manifest file count or duplicate path error")
    total = 0
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise SystemExit("Unsafe manifest path: " + row["path"])
        data = path.read_bytes()
        if len(data) != row["size_bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise SystemExit("Artifact failed verification: " + row["path"])
        total += len(data)
    present = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if present != set(paths) | {"manifest.json"}:
        raise SystemExit("Published files differ from the manifest allowlist")
    if total != manifest["total_bytes"]:
        raise SystemExit("Manifest byte total error")
    print(json.dumps({"status": "ok", "verified_files": len(paths), "verified_bytes": total}))


if __name__ == "__main__":
    main()
