from __future__ import annotations

import csv
from pathlib import Path

from ..io import atomic_json, read_json


def report(run, output=None, plots=True):
    run = Path(run)
    output = Path(output) if output else run / "report"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((run / "metrics").glob("*/*/metrics.json"))
    if not paths:
        raise ValueError("No scored method/dataset results in this run")
    summaries = [read_json(path) for path in paths]
    for ds in {s["dataset"] for s in summaries}:
        identities = {s["dataset_sha256"] for s in summaries if s["dataset"] == ds}
        if len(identities) != 1:
            raise ValueError(f"Cannot compare different dataset manifests under one name: {ds}")
    rows = [
        {
            key: s.get(key)
            for key in (
                "dataset",
                "method",
                "scope",
                "development_run",
                "execution_complete",
                "E_p",
                "E_p_SD",
                "E_t_m",
                "E_q_deg",
                "ADD_S_m",
                "valid",
                "targets",
                "coverage",
                "localization_policy",
            )
        }
        for s in summaries
    ]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# Benchmark report",
        "",
        "Errors are pooled over valid targets. Coverage and failed jobs are reported separately.",
        "",
        "| Dataset | Method | E_p | SD | E_t (m) | E_q (deg) | ADD-S (m) | Valid/targets | Scope |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(value):
        return "—" if value is None else f"{value:.6f}"

    for s in summaries:
        scope = "development" if s.get("development_run") else s["scope"]
        if s.get("execution_complete") is False:
            scope += ", failed jobs"
        text.append(
            f"| {s['dataset']} | {s['method']} | "
            + " | ".join(fmt(s.get(k)) for k in ("E_p", "E_p_SD", "E_t_m", "E_q_deg", "ADD_S_m"))
            + f" | {s['valid']}/{s['targets']} | {scope} |"
        )
    text.extend(["", "Input policies:", ""])
    for ds in sorted({s["dataset"] for s in summaries}):
        policy = next(s["localization_policy"] for s in summaries if s["dataset"] == ds)
        text.append(f"- {ds}: {policy}")
    matched = []
    for ds in sorted({s["dataset"] for s in summaries}):
        methods = {}
        for path, summary in zip(paths, summaries):
            if summary["dataset"] != ds:
                continue
            with (path.parent / "targets.csv").open() as stream:
                methods[summary["method"]] = {
                    (r["stream"], r["frame"], r["object_id"]): float(r["E_p"])
                    for r in csv.DictReader(stream)
                    if r["valid"] in {"True", "1"}
                }
        if len(methods) > 1:
            common = set.intersection(*(set(m) for m in methods.values()))
            matched.append(
                dict(
                    dataset=ds,
                    common_targets=len(common),
                    E_p={
                        name: sum(values[k] for k in common) / len(common) if common else None
                        for name, values in methods.items()
                    },
                )
            )
    atomic_json(output / "matched_targets.json", matched)
    text.extend(["", "Matched valid targets:", ""])
    for item in matched:
        text.append(
            f"- {item['dataset']}: {item['common_targets']} common targets; "
            + ", ".join(f"{k} E_p={fmt(v)}" for k, v in item["E_p"].items())
        )
    timings = []
    for path in sorted((run / "timings").rglob("timing.json")):
        item = read_json(path)
        if "repetitions" not in item:
            continue
        if len(item["repetitions"]) != 3 or not all(
            r.get("measured_sustained") and r.get("kind") == "fresh_inference"
            for r in item["repetitions"]
        ):
            raise ValueError(f"Unverified fresh timing receipt: {path}")
        timings.append(item)
    atomic_json(output / "timings.json", timings)
    text.extend(["", "Sustained pose-stage timing:", ""])
    if not timings:
        text.append(
            "No fresh three-repeat timings attached. Accuracy runs and cached imports do not establish FPS."
        )
    for item in timings:
        text.append(
            f"- {item['method']}/{item['dataset']}/{item['stream']}: "
            f"{item['fps_mean']:.4f} ± {item['fps_sd']:.4f} FPS (three repeats). " + item["scope"]
        )
    atomic_json(output / "comparison.json", summaries)
    (output / "REPORT.md").write_text("\n".join(text) + "\n")
    if plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return dict(report=str(output / "REPORT.md"), plots="install the plots extra")
        datasets = sorted({s["dataset"] for s in summaries})
        fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 7), squeeze=False)
        for col, ds in enumerate(datasets):
            data = [s for s in summaries if s["dataset"] == ds and s["E_p"] is not None]
            labels = [s["method"] for s in data]
            axes[0, col].bar(
                labels, [s["E_p"] for s in data], yerr=[s["E_p_SD"] or 0 for s in data], capsize=3
            )
            axes[1, col].bar(labels, [100 * s["coverage"] for s in data])
            axes[0, col].set_title(ds)
            axes[0, col].set_ylabel("Pose error (mean ± sample SD)")
            axes[1, col].set_ylabel("Coverage (%)")
            axes[1, col].set_ylim(0, 105)
            for row in range(2):
                axes[row, col].tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(output / "accuracy_coverage.png", dpi=160)
        fig.savefig(output / "accuracy_coverage.pdf")
        plt.close(fig)
        if timings:
            fig, ax = plt.subplots(figsize=(max(7, len(timings) * 1.1), 4))
            labels = [f"{t['method']}\n{t['dataset']}\n{t['stream']}" for t in timings]
            ax.bar(
                labels,
                [t["fps_mean"] for t in timings],
                yerr=[t["fps_sd"] for t in timings],
                capsize=3,
            )
            ax.set_ylabel("Resident pose-stage FPS (3 fresh repeats)")
            ax.tick_params(axis="x", labelrotation=30)
            fig.tight_layout()
            fig.savefig(output / "fps.png", dpi=160)
            fig.savefig(output / "fps.pdf")
            plt.close(fig)
    return dict(report=str(output / "REPORT.md"))
