# Import and score saved predictions

The canonical NPZ contains `schema`, integer `frames`, metric `poses` shaped
N×4×4, boolean `valid`, string `status`, string `failure` and optional confidence.
`schema` is `spacergbbenchmark.predictions.v1`. One archive represents one object
stream, named exactly as the stream ID. Dataset and object identity are supplied
by the frozen manifest. Invalid predictions never count as zero error.

```bash
.venv/bin/spacergbbenchmark import \
  --dataset results/data/spark/dataset.json --source /path/to/saved \
  --out results/imported/spark --format npz --units m \
  --pattern '{stream}/poses.npz' --pose-key poses --frame-key frames
.venv/bin/spacergbbenchmark score \
  --dataset results/data/spark/dataset.json --predictions results/imported/spark \
  --out results/saved/metrics/method/spark --method method
.venv/bin/spacergbbenchmark report --suite-run results/saved
```

When an archive has no frame IDs, require `--schedule inputs` or `--schedule targets`.
Unknown timestamps or implicit length-based matching are rejected. Select a
candidate-score array explicitly using `--candidate-scores scores` for top-K input.
For another CAD frame, provide the 4×4 source-to-object transform as a JSON file
with `--source-to-object`. Translation units are always explicit. `--format bop`
reads BOP CSV rotations and translations with the declared input units.

Imports keep a source checksum receipt and are labeled saved-output import.
They never establish fresh inference FPS. `targets.csv` has every target,
including missing ones. `metrics.json` contains pooled means, sample SD and
coverage. `sequences.json` keeps separate sequence results. ADD-S requires frozen
mesh samples, or `--no-adds` explicitly omits that metric.

The full-data scorer regression is recorded in
[scoring_validation.json](scoring_validation.json). All 51,079 declared targets
were checked, with missing YCB predictions retained in the denominator. These
are verification results for historical inputs, not a new model comparison.
