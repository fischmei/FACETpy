"""Reproducible synthetic benchmark, not an optimization entry point.

Run with: python scripts/benchmark_flex_optimization.py /tmp/flex-benchmark
Each method runs in a fresh process. Import time is excluded from wall time;
RSS includes the interpreter/scientific stack. Filesystem block counters may
be zero for cached I/O; logical bytes written are reported separately.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def run_method(method):
    import mne
    import numpy as np
    from loguru import logger
    from threadpoolctl import threadpool_limits

    from facet import DownSample, Flex, Loader, TriggerDetector, UpSample
    from facet.correction.flex.decisions import AveragingMatrixBuilder, MatrixDecisions
    from facet.optimization import metrics, search
    from facet.optimization.cache import EvaluationCache, file_identity
    from facet.optimization.prepared import prepare_dataset

    logger.remove()
    mne.set_log_level("ERROR")
    with tempfile.TemporaryDirectory(prefix="facet_benchmark_") as temporary, threadpool_limits(limits=1):
        root = Path(temporary)
        sfreq, channels = 200, 3
        times = np.arange(8000) / sfreq
        rng = np.random.default_rng(42)
        physiological = (
            np.sin(2 * np.pi * 6.7 * times) + np.sin(2 * np.pi * 10.7 * times) + rng.normal(0, 0.2, len(times))
        ) * 1e-6
        data = np.tile(physiological + 20e-6 * np.sin(2 * np.pi * 20 * times), (channels, 1))
        raw = mne.io.RawArray(data, mne.create_info(["Cz", "Pz", "Fz"], sfreq, "eeg"))
        triggers = np.arange(100, 7901, 100)
        raw.set_annotations(mne.Annotations(triggers / sfreq, np.zeros(len(triggers)), ["R128"] * len(triggers)))
        path = root / "sample_raw.fif"
        raw.save(path, fmt="double", overwrite=True)
        settings = search.RunSettings("^R128$", 2, 5, 5, None, 2, 0.2, 3, 13, 80, 6, 0.5, 0.05, 10, chunk_seconds=120)
        caps = search.SearchCapabilities(False, False, False, False)
        import optuna

        recipes = [
            search.suggest_matrix_decisions(optuna.trial.FixedTrial(params), caps, settings)
            for _, params in search.anchor_parameter_sets(caps)
        ]
        counters = {"original_psds": 0, "trigger_detections": 0, "upsampled_channels": 0, "corrected_channels": 0}
        written = 0
        before = resource.getrusage(resource.RUSAGE_SELF)
        started = time.perf_counter()
        if method in {"dense_matrix", "sparse_matrix"}:
            epochs = rng.normal(size=(800, 256))
            builder = AveragingMatrixBuilder(MatrixDecisions())
            matrix = builder.build(epochs) if method == "dense_matrix" else builder.build_sparse(epochs)
            template = matrix @ epochs
            written = 0
            counters = {
                "epochs": len(epochs),
                "epoch_samples": 256,
                "matrix_bytes": matrix.nbytes
                if method == "dense_matrix"
                else matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
                "template_checksum": float(template.sum()),
            }
        elif method == "original_lifecycle":
            for repeat in range(3):
                for number, recipe in enumerate(recipes):
                    context = Loader(str(path)).execute(None)
                    context = TriggerDetector("^R128$").execute(context)
                    counters["trigger_detections"] += 1
                    up = UpSample(2).execute(context)
                    counters["upsampled_channels"] += channels
                    corrected = Flex(
                        matrix_decisions=recipe,
                        realign_after_averaging=True,
                        interpolate_volume_gaps=True,
                        apply_epoch_alpha_scaling=True,
                        track_estimated_noise=False,
                    ).execute(up)
                    corrected = DownSample(2).execute(corrected).get_raw()
                    counters["corrected_channels"] += channels
                    for channel in range(channels):
                        f, psd, mask = metrics.original_spectrum(raw._data[channel, 100:], sfreq, triggers, settings)
                        metrics.spectral_metrics(corrected._data[channel, 100:], sfreq, f, psd, mask)
                        counters["original_psds"] += 1
                    output = root / f"{repeat}_{number}.edf"
                    corrected.export(output, fmt="edf", overwrite=True, verbose=False)
                    written += output.stat().st_size
        else:
            dataset = search.DatasetSpec("sample", path, triggers, None, input_identity=file_identity(path))
            cache = EvaluationCache(root / "cache.sqlite3", signature="benchmark")
            prepare_dataset(dataset, settings, root / "prepared")
            for _ in range(3):
                for recipe in recipes:
                    row = search.evaluate_configuration(recipe, dataset, settings, cache)
                    if not row["success"]:
                        raise RuntimeError(row["error"])
            counters = {
                "original_psds": channels,
                "trigger_detections": 0,
                "upsampled_channels": channels,
                "corrected_channels": channels * len(recipes),
                "dataset_metric_cache_hits": cache.hits,
                "unique_configurations": len(recipes),
                "note": "Triggers supplied from the common synthetic fixture; preflight adds one detection per session.",
            }
            written = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and p != path)
            cache.close()
        elapsed = time.perf_counter() - started
        after = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "method": method,
            "wall_seconds": elapsed,
            "peak_rss_mb": after.ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
            "block_inputs": after.ru_inblock - before.ru_inblock,
            "block_outputs": after.ru_oublock - before.ru_oublock,
            "logical_output_bytes": written,
            "calculations": counters,
            "synthetic_input": {"channels": channels, "sfreq": sfreq, "seconds": 40, "repeated_folds": 3},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method")
    args = parser.parse_args()
    if args.method:
        args.output.write_text(json.dumps(run_method(args.method), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for method in ("dense_matrix", "sparse_matrix", "original_lifecycle", "prepared_metrics"):
        destination = args.output / f"{method}.json"
        subprocess.run([sys.executable, __file__, str(destination), "--method", method], check=True)
        results.append(json.loads(destination.read_text()))
    (args.output / "benchmark.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
