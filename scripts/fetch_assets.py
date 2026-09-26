"""Download explicit HTTP(S) asset URLs and verify their SHA-256 before publishing."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def fetch(catalog, output):
    for asset in json.loads(Path(catalog).read_text())["assets"]:
        relative = Path(asset["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Asset destinations must be relative to the output root")
        if not asset["url"].startswith(("https://", "http://")):
            raise ValueError("Expected an explicit HTTP(S) download URL")
        destination = Path(output) / relative
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"Existing asset differs: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        digest = hashlib.sha256()
        with (
            urllib.request.urlopen(asset["url"], timeout=60) as response,
            temporary.open("wb") as stream,
        ):
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
                digest.update(block)
        if digest.hexdigest() != asset["sha256"]:
            raise ValueError(f"Download checksum mismatch: {temporary}")
        temporary.replace(destination)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("weights"))
    a = p.parse_args()
    fetch(a.catalog, a.out)
