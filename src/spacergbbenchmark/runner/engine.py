from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .. import __version__
from ..datasets import Dataset
from ..evaluation.score import score
from ..io import absolute, atomic_json, identity, read_config, read_json, safe_name, sha256
from ..models import model_class
from ..schema import read_predictions, valid_pose, write_predictions
from .resources import lease, output_lease

PATH_KEYS = {
    "python",
    "pico_python",
    "mega_python",
    "giga_python",
    "droid_python",
    "rgbtrack_python",
    "pico_source",
    "pico_checkpoint",
    "mega_source",
    "mega_data",
    "droid_source",
    "droid_checkpoint",
    "giga_source",
    "giga_checkpoint",
    "dinov2_source",
    "giga_root",
    "rgbtrack_source",
    "templates",
    "srt3d_executable",
    "srt3d_source",
    "srt3d_model_cache",
    "object_dir",
}


def resolve_settings(settings, base):
    result = dict(settings)
    for key, value in list(result.items()):
        if key in PATH_KEYS and isinstance(value, str):
            result[key] = str(absolute(value, base))
        elif key in {"asset_paths", "package_paths"}:
            result[key] = [str(absolute(v, base)) for v in value]
        elif isinstance(value, dict) and key == "objects":
            result[key] = {k: resolve_settings(v, base) for k, v in value.items()}
    return result


def source_receipt():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}


def fingerprint(sequence, settings, sources):
    paths = {sequence.file(sequence.config["mesh"])}
    for frame in sequence.frames:
        paths.add(sequence.file(frame["image"]))
        for key in ("mask", "slam_mask"):
            if frame.get(key):
                paths.add(sequence.file(frame[key]))
    for key in PATH_KEYS - {
        "python",
        "pico_python",
        "mega_python",
        "giga_python",
        "droid_python",
        "rgbtrack_python",
        "srt3d_model_cache",
    }:
        value = settings.get(key)
        if value:
            item = Path(value)
            if key == "mega_data":
                for model in ("coarse-rgb-906902141", "refiner-rgb-653307694"):
                    paths.update(
                        item / "megapose-models" / model / filename
                        for filename in ("config.yaml", "checkpoint.pth.tar")
                    )
                continue
            if item.is_file():
                paths.add(item)
            elif item.is_dir():
                if (item / ".git").exists():
                    # Include tracked source content, including local modifications.
                    tracked = (
                        subprocess.check_output(
                            ["git", "-C", str(item), "ls-files", "--recurse-submodules", "-z"]
                        )
                        .decode()
                        .split("\0")
                    )
                    paths.update(
                        item / name for name in tracked if name and (item / name).is_file()
                    )
                else:
                    paths.update(
                        p for p in item.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                    )
            else:
                raise FileNotFoundError(value)
    for value in settings.get("asset_paths", []):
        item = Path(value)
        paths.update(
            p
            for p in item.rglob("*")
            if p.is_file()
            and not {"__pycache__", ".git"}.intersection(p.parts)
            and p.suffix not in {".pyc", ".pyo"}
        ) if item.is_dir() else paths.add(item)
    for value in settings.get("package_paths", []):
        paths.update(
            p for p in Path(value).rglob("*") if p.is_file() and p.suffix in {".py", ".json"}
        )
    assets = {str(p): sha256(p) for p in sorted(paths)}
    environments = {}
    for key in sorted(PATH_KEYS):
        if (key == "python" or key.endswith("_python")) and settings.get(key):
            interpreter = settings[key]
            if interpreter not in environments:
                environments[interpreter] = subprocess.check_output(
                    [
                        interpreter,
                        "-c",
                        "import sys,json,importlib.metadata as m;print(json.dumps([sys.version,sorted((d.metadata.get('Name',''),d.version) for d in m.distributions())]))",
                    ],
                    text=True,
                ).strip()
    return dict(
        sequence_sha256=sha256(sequence.path),
        settings=settings,
        source=sources,
        assets=assets,
        environments=environments,
    )


def hardware():
    result = dict(
        python=sys.version,
        platform=platform.platform(),
        cpu=platform.processor(),
        cpu_count=os.cpu_count(),
        package_version=__version__,
        command=sys.argv,
    )
    result["dependencies"] = {
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "Pillow", "PyYAML")
    }
    if shutil.which("nvidia-smi"):
        result["gpu"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
    return result


def sequence_job(
    dataset,
    stream,
    model_name,
    settings,
    output,
    gpu,
    *,
    warmup=0,
    timing=False,
    max_frames=None,
    stop_event=None,
):
    with output_lease(output):
        return _sequence_job(
            dataset,
            stream,
            model_name,
            settings,
            output,
            gpu,
            warmup=warmup,
            timing=timing,
            max_frames=max_frames,
            stop_event=stop_event,
        )


def _sequence_job(
    dataset,
    stream,
    model_name,
    settings,
    output,
    gpu,
    *,
    warmup=0,
    timing=False,
    max_frames=None,
    stop_event=None,
):
    sequence = dataset.sequence(stream)
    cls = model_class(model_name)
    if cls.temporal_mode not in {"framewise", "causal", "offline"}:
        raise ValueError("Model must declare framewise, causal or offline behavior")
    unavailable = set(cls.required_modalities) - {"rgb", "intrinsics", "cad", "localization"}
    if unavailable:
        raise ValueError(f"Model requires unavailable modalities: {sorted(unavailable)}")
    if cls.measurement_kind == "fixture" and dataset.config["scope"] == "full":
        raise ValueError("The fixture adapter is restricted to fixture manifests")
    selected_settings = dict(settings)
    objects = selected_settings.pop("objects", {})
    selected_settings.update(objects.get(f"{dataset.name}:{sequence.object_id}", {}))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(sequence, selected_settings, source_receipt())
    if hasattr(cls, "tracker_package"):
        package = importlib.import_module(cls.tracker_package)
        root = Path(package.__file__).parent
        signature["tracker_source"] = {
            str(p.relative_to(root)): sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".json"}
        }
    signature.update(
        mode=cls.temporal_mode,
        model=model_name,
        max_frames=max_frames,
        warmup=warmup,
        timing=timing,
        adapter_source_sha256=sha256(inspect.getfile(cls)),
    )
    digest = identity(signature)
    state_path = output / "state.json"
    if state_path.exists():
        old = read_json(state_path)
        if old["identity"] != digest:
            raise ValueError("Inputs/configuration changed; create a new run")
        if (
            old["status"] == "complete"
            and (output / "predictions.npz").is_file()
            and sha256(output / "predictions.npz") == old["prediction_sha256"]
        ):
            read_predictions(output / "predictions.npz")
            return old
    attempts = len(list(output.glob("attempt-*")))
    attempt = output / f"attempt-{attempts + 1:03d}"
    attempt.mkdir()
    atomic_json(output / "identity.json", signature)
    state = dict(
        status="running",
        identity=digest,
        model=model_name,
        stream=sequence.name,
        attempt=attempt.name,
        pid=os.getpid(),
        gpu=gpu,
        started=time.time(),
        completed_frames=0,
    )
    atomic_json(state_path, state)
    indices = [
        i
        for i, frame in enumerate(sequence.frames)
        if cls.temporal_mode != "framewise" or frame["target"]
    ]
    if max_frames is not None:
        indices = indices[:max_frames]
    model = None
    predictions, latencies = [], []
    started = None
    try:
        if timing and (
            cls.temporal_mode == "offline"
            or len(indices) <= warmup
            or cls.measurement_kind != "fresh_inference"
        ):
            raise ValueError("Sustained timing needs fresh streaming inference beyond warm-up")
        with lease(gpu if cls.needs_gpu else None, timing=timing):
            setup_started = time.perf_counter()
            model = cls(selected_settings, sequence, attempt, gpu=gpu)
            model.start()
            setup_seconds = time.perf_counter() - setup_started
            if cls.temporal_mode == "offline":
                predictions = model.run_sequence(indices)
            else:
                for position, index in enumerate(indices):
                    if stop_event is not None and stop_event.is_set():
                        raise InterruptedError("Suite interrupted; restart this sequence on resume")
                    if position == warmup:
                        if (
                            timing
                            and cls.temporal_mode == "causal"
                            and not any(valid_pose(p.pose) for p in predictions)
                        ):
                            raise ValueError("Tracker has not initialized; increase --warmup")
                        model.synchronize()
                        started = time.perf_counter()
                    before = time.perf_counter()
                    predictions.append(model.step(index))
                    latencies.append(time.perf_counter() - before)
                    if not timing and position % 30 == 0:
                        state["completed_frames"] = position + 1
                        atomic_json(state_path, state)
            model.synchronize()
            elapsed = None if started is None else time.perf_counter() - started
            model.close()
            model = None
        frame_ids = [sequence.frames[i]["id"] for i in indices]
        write_predictions(output / "predictions.npz", frame_ids, predictions)
        read_predictions(output / "predictions.npz", frame_ids)
        measurement = dict(
            kind=cls.measurement_kind,
            measured_sustained=bool(timing),
            seconds=elapsed if timing else None,
            frames=len(indices) - warmup if timing else None,
            fps=(len(indices) - warmup) / elapsed if timing else None,
            warmup_inputs=warmup,
            processed_inputs=len(indices),
            timed_valid_outputs=sum(valid_pose(p.pose) for p in predictions[warmup:])
            if timing
            else None,
            denominator="scored target inputs"
            if cls.temporal_mode == "framewise"
            else "dense video inputs",
            scope="resident pose pipeline; excludes loading, warm-up, external detection and scoring",
            latency_median_seconds=float(np.median(latencies[warmup:]))
            if latencies[warmup:]
            else None,
            initialization_included_in_warmup=True,
            setup_seconds=setup_seconds,
            warmup_seconds=float(sum(latencies[:warmup])),
            cpu_load_1min=os.getloadavg()[0],
            precision=selected_settings.get(
                "precision", "upstream default; see environment receipt"
            ),
            input_resolution=[sequence.width, sequence.height],
            cpu_threads=selected_settings.get("cpu_threads", 1),
        )
        atomic_json(output / "timing.json", measurement)
        state.update(
            status="complete",
            finished=time.time(),
            completed_frames=len(indices),
            prediction_sha256=sha256(output / "predictions.npz"),
            fresh_inference=cls.measurement_kind == "fresh_inference",
        )
        atomic_json(state_path, state)
        return state
    except BaseException as error:
        state.update(
            status="interrupted"
            if isinstance(error, (KeyboardInterrupt, InterruptedError))
            else "failed",
            error=f"{type(error).__name__}: {error}",
            finished=time.time(),
        )
        atomic_json(state_path, state)
        (attempt / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        if model is not None:
            model.close()


def load_suite(path):
    path = Path(path).absolute()
    config = read_config(path)
    result = dict(config)
    result["datasets"] = {
        name: str(absolute(value, path.parent)) for name, value in config["datasets"].items()
    }
    result["models"] = {}
    for name, value in config["models"].items():
        safe_name(name)
        if isinstance(value, str):
            model_path = absolute(value, path.parent)
            value = read_config(model_path)
            base = model_path.parent
        else:
            base = path.parent
        result["models"][name] = dict(
            adapter=value.get("adapter", name),
            settings=resolve_settings(value.get("settings", {}), base),
        )
    return result


def run_suite(
    config, output, *, resume=False, jobs=1, gpus=None, max_frames=None, streams=None, no_adds=False
):
    with output_lease(output):
        return _run_suite(
            config,
            output,
            resume=resume,
            jobs=jobs,
            gpus=gpus,
            max_frames=max_frames,
            streams=streams,
            no_adds=no_adds,
        )


def _run_suite(
    config, output, *, resume=False, jobs=1, gpus=None, max_frames=None, streams=None, no_adds=False
):
    output = Path(output).absolute()
    if output.exists() and not resume:
        raise FileExistsError(f"Preserving run: {output}; use resume")
    output.mkdir(parents=True, exist_ok=True)
    if resume and read_json(output / "suite.json") != config:
        raise ValueError("Resolved suite changed; create a new run")
    dataset_hashes = {name: sha256(path) for name, path in config["datasets"].items()}
    if resume and read_json(output / "datasets.json") != dataset_hashes:
        raise ValueError("Dataset or scoring protocol changed; create a new run")
    atomic_json(output / "datasets.json", dataset_hashes)
    atomic_json(output / "suite.json", config)
    hardware_path = output / (
        f"hardware-resume-{time.time_ns()}.json" if resume else "hardware.json"
    )
    atomic_json(hardware_path, hardware())
    options = dict(jobs=jobs, gpus=gpus, max_frames=max_frames, streams=streams, no_adds=no_adds)
    if resume and read_json(output / "options.json") != options:
        raise ValueError("Run options changed; create a new run")
    atomic_json(output / "options.json", options)
    pending = []
    for name, path in config["datasets"].items():
        dataset = Dataset(path)
        if dataset.name != name:
            raise ValueError("Dataset key and manifest differ")
        dataset.validate()
        for method, model in config["models"].items():
            for stream in dataset.streams:
                if streams and stream["id"] not in streams:
                    continue
                gpu = str(gpus[len(pending) % len(gpus)]) if gpus else None
                if model_class(model["adapter"]).needs_gpu and gpu is None:
                    raise ValueError(f"{method} needs --gpus with available device indices")
                pending.append((dataset, stream, method, model, gpu))
    if not pending:
        raise ValueError("No jobs selected")

    stop_event = threading.Event()

    def execute(job):
        dataset, stream, method, model, gpu = job
        destination = output / "jobs" / method / dataset.name / stream["id"]
        retries = int(config.get("retries", 0))
        for retry in range(retries + 1):
            if stop_event.is_set():
                return dict(
                    method=method, dataset=dataset.name, stream=stream["id"], status="interrupted"
                )
            try:
                result = sequence_job(
                    dataset,
                    stream,
                    model["adapter"],
                    model["settings"],
                    destination,
                    gpu,
                    max_frames=max_frames,
                    stop_event=stop_event,
                )
                predicted = output / "predictions" / method / dataset.name
                predicted.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(destination / "predictions.npz", predicted / f"{stream['id']}.npz")
                return dict(method=method, dataset=dataset.name, **result)
            except Exception as error:
                if retry == retries:
                    return dict(
                        method=method,
                        dataset=dataset.name,
                        stream=stream["id"],
                        status="failed",
                        error=str(error),
                    )

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(execute, job) for job in pending]
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                atomic_json(
                    output / "progress.json",
                    dict(total=len(pending), finished=len(results), results=results),
                )
                print(
                    f"{len(results)}/{len(pending)} {result['method']}/{result['dataset']}/{result['stream']}: {result['status']}",
                    flush=True,
                )
        except KeyboardInterrupt:
            stop_event.set()
            for future in futures:
                future.cancel()
            raise
    summaries = []
    for name, path in config["datasets"].items():
        for method in config["models"]:
            destination = output / "metrics" / method / name
            summary = score(
                path, output / "predictions" / method / name, destination, method, adds=not no_adds
            )
            matching = [r for r in results if r["method"] == method and r["dataset"] == name]
            summary["execution_complete"] = bool(matching) and all(
                r["status"] == "complete" for r in matching
            )
            summary["development_run"] = bool(max_frames is not None or streams)
            atomic_json(destination / "metrics.json", summary)
            summaries.append(summary)
    status = "complete" if all(r["status"] == "complete" for r in results) else "failed"
    atomic_json(
        output / "summary.json", dict(status=status, methods=summaries, finished=time.time())
    )
    from ..evaluation.report import report

    report(output, plots=False)
    return dict(status=status, output=str(output), jobs=len(results))


def new_run_id():
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
