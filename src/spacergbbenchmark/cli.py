"""Single command-line interface for data, inference, scoring and reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RGB pose benchmarking on four complete evaluation partitions"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect the coordinator and installed model adapters")
    assets = sub.add_parser("prepare-model", help="Render per-object templates once")
    assets.add_argument("--suite", type=Path, required=True)
    assets.add_argument("--dataset", required=True)
    assets.add_argument("--model", required=True)
    assets.add_argument("--objects", nargs="+", type=int)
    assets.add_argument("--gpu", type=int, default=0)
    data = sub.add_parser("data")
    commands = data.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--out", type=Path, default=Path("results/data"))
    prepare.add_argument("--datasets", nargs="+", default=["all"])
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path("results/data"))
    validate.add_argument("--datasets", nargs="+", default=["all"])
    validate.add_argument("--deep", action="store_true")
    for name in ("run", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path)
        if name == "run":
            p.add_argument("--suite", type=Path, required=True)
            p.add_argument("--model")
            p.add_argument("--dataset")
            p.add_argument("--jobs", type=int, default=1)
            p.add_argument("--gpus", nargs="+", type=int)
            p.add_argument("--max-frames", type=int)
            p.add_argument("--streams", nargs="+")
            p.add_argument("--no-adds", action="store_true")
    scoring = sub.add_parser("score")
    scoring.add_argument("--dataset", type=Path, required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--out", type=Path, required=True)
    scoring.add_argument("--method", required=True)
    scoring.add_argument("--no-adds", action="store_true")
    importing = sub.add_parser("import")
    importing.add_argument("--dataset", type=Path, required=True)
    importing.add_argument("--source", type=Path, required=True)
    importing.add_argument("--out", type=Path, required=True)
    importing.add_argument("--format", choices=["npz", "bop"], default="npz")
    importing.add_argument("--pattern", default="{stream}.npz")
    importing.add_argument("--pose-key", default="poses")
    importing.add_argument("--frame-key", default="frames")
    importing.add_argument("--schedule", choices=["inputs", "targets"])
    importing.add_argument("--units", choices=["m", "mm"], required=True)
    importing.add_argument("--candidate-scores")
    importing.add_argument("--source-to-object", type=Path)
    reporting = sub.add_parser("report")
    reporting.add_argument("--suite-run", type=Path, required=True)
    reporting.add_argument("--out", type=Path)
    reporting.add_argument("--no-plots", action="store_true")
    demo = sub.add_parser("demo", help="Run a tiny redistributable CPU example")
    demo.add_argument("--out", type=Path, default=Path("results/demo"))
    benchmark = sub.add_parser(
        "benchmark", help="Three fresh isolated resident-model timing repeats"
    )
    benchmark.add_argument("--suite", type=Path, required=True)
    benchmark.add_argument("--dataset", required=True)
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--stream", required=True)
    benchmark.add_argument("--gpu", type=int, required=True)
    benchmark.add_argument("--warmup", type=int, default=30)
    benchmark.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if isinstance(result, dict) and result.get("status") == "failed" else 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"spacergbbenchmark: {error}", file=sys.stderr)
        return 1


def dispatch(a):
    from .io import read_json
    from .runner.engine import hardware, load_suite, new_run_id, run_suite

    if a.command == "doctor":
        from importlib.metadata import entry_points

        from .models import BUILTINS, model_class

        names = sorted(
            set(BUILTINS) | {p.name for p in entry_points(group="spacergbbenchmark.models")}
        )

        return dict(
            hardware(),
            adapters={
                name: dict(
                    temporal_mode=model_class(name).temporal_mode,
                    needs_gpu=model_class(name).needs_gpu,
                )
                for name in names
            },
            note="GPU model environments and assets are checked when each adapter starts",
        )
    if a.command == "prepare-model":
        from .assets import prepare_templates

        return prepare_templates(load_suite(a.suite), a.dataset, a.model, a.objects, a.gpu)
    if a.command == "data":
        names = ["spark", "ycbv", "swisscube", "shirt"] if a.datasets == ["all"] else a.datasets
        if a.action == "prepare":
            from .datasets.prepare import prepare

            return prepare(a.config, a.out, names)
        from .datasets import Dataset

        return [Dataset(a.root / name / "dataset.json").validate(a.deep) for name in names]
    if a.command == "run":
        config = load_suite(a.suite)
        if a.model:
            config["models"] = {a.model: config["models"][a.model]}
        if a.dataset:
            config["datasets"] = {a.dataset: config["datasets"][a.dataset]}
        if a.jobs < 1 or a.max_frames is not None and a.max_frames < 1:
            raise ValueError("jobs and max-frames must be positive")
        return run_suite(
            config,
            a.run or Path("results/runs") / new_run_id(),
            jobs=a.jobs,
            gpus=a.gpus,
            max_frames=a.max_frames,
            streams=a.streams,
            no_adds=a.no_adds,
        )
    if a.command == "resume":
        if a.run is None:
            raise ValueError("resume requires --run")
        return run_suite(
            read_json(a.run / "suite.json"), a.run, resume=True, **read_json(a.run / "options.json")
        )
    if a.command == "score":
        from .evaluation.score import score

        return score(a.dataset, a.predictions, a.out, a.method, adds=not a.no_adds)
    if a.command == "import":
        from .evaluation.importer import import_predictions

        return import_predictions(
            a.dataset,
            a.source,
            a.out,
            format=a.format,
            pattern=a.pattern,
            pose_key=a.pose_key,
            frame_key=a.frame_key,
            schedule=a.schedule,
            units=a.units,
            candidate_scores=a.candidate_scores,
            source_to_object=read_json(a.source_to_object) if a.source_to_object else None,
        )
    if a.command == "report":
        from .evaluation.report import report

        return report(a.suite_run, a.out, plots=not a.no_plots)
    if a.command == "demo":
        from .demo import demo

        return demo(a.out)
    if a.command == "benchmark":
        from .runner.timing import benchmark

        return benchmark(load_suite(a.suite), a.dataset, a.model, a.stream, a.gpu, a.warmup, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
