"""Fetch immutable upstream revisions and apply the recorded runtime patches."""

import argparse
import json
import subprocess
from pathlib import Path


def fetch(root, names=None, ssh=False):
    catalog = Path(__file__).resolve().parents[1] / "environments/sources.json"
    sources = json.loads(catalog.read_text())
    root.mkdir(parents=True, exist_ok=True)
    for name in names or sources:
        spec = sources[name]
        url = spec["url"]
        if ssh and url.startswith("https://github.com/"):
            url = "git@github.com:" + url[len("https://github.com/") :]
        destination = root / name
        patch = catalog.parent / spec["patch"] if spec.get("patch") else None
        if destination.exists():
            head = subprocess.check_output(
                ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
            ).strip()
            changed = subprocess.check_output(
                ["git", "-C", str(destination), "diff", "--binary", "HEAD"]
            )
            if head != spec["revision"] or changed != (patch.read_bytes() if patch else b""):
                raise RuntimeError(
                    f"Existing source differs: {destination}. Use a fresh directory."
                )
            continue
        subprocess.run(["git", "init", str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", url], check=True)
        subprocess.run(
            ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", spec["revision"]],
            check=True,
            timeout=180,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True
        )
        if patch:
            subprocess.run(["git", "-C", str(destination), "apply", str(patch)], check=True)
        subprocess.run(
            ["git", "-C", str(destination), "submodule", "update", "--init", "--recursive"],
            check=True,
            timeout=180,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("third_party"))
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--ssh", action="store_true", help="Use configured GitHub SSH access")
    args = parser.parse_args()
    fetch(args.out.resolve(), args.sources, args.ssh)
