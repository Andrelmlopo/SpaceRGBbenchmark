"""Atomic artifacts, deterministic identities, and configuration paths."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def read_config(path):
    path = Path(path)
    if path.suffix == ".json":
        value = read_json(path)
    else:
        import yaml

        value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a configuration mapping: {path}")
    return value


def absolute(path, base):
    # Do not resolve interpreter symlinks: that discards virtual environments.
    return Path(os.path.abspath(Path(base) / os.path.expandvars(os.path.expanduser(str(path)))))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_npz(path, **arrays):
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def safe_name(value):
    import re

    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid artifact name: {value!r}")
    return value
