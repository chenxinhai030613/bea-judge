"""Unit tests for src/calibration_methods.py (P0-3 5-way comparison)."""

import sys
import json
import unittest
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibration_methods import (  # noqa: E402
    CalibrationResult,
    SUPPORTED_METHODS,
    METHOD_DISPATCH,
    _pav_isotonic,
    _platt_one_vs_rest,
    compute_calibration_metrics,
    fit_conformal_split,
    fit_isotonic,
    fit_platt,
    fit_temperature,
    fit_vector_scaling,
    reliability_bins,
    run_calibration_comparison,
)


def _make_synthetic_data(
    n_dev: int = 240,
    n_test: int = 200,
    n_classes: int = 3,
    overconfident: bool = True,
    seed: int = 20260520,
) -> tuple:
    rng = np.random.default_rng(seed)
    y_dev = rng.integers(low=0, high=n_classes, size=n_dev)
    y_test = rng.integers(low=0, high=n_classes, size=n_test)

    def _build(y: np.ndarray) -> np.ndarray:
        n = len(y)
        logits = rng.normal(loc=0.0, scale=0.3, size=(n, n_classes))
        true_boost = 1.0 if overconfident else 0.5
        for i, label in enumerate(y):
            if rng.random() < 0.7:
                logits[i, label] += true_boost
        if overconfident:
            logits *= 4.0
        m = logits.max(axis=1, keepdims=True)
        exp = np.exp(logits - m)
        return exp / exp.sum(axis=1, keepdims=True)

    p_dev = _build(y_dev)
    p_test = _build(y_test)
    return p_dev, y_dev, p_test, y_test


class CalibrationMethodsTest(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(20260520)
        self.p_dev, self.y_dev, self.p_test, self.y_test = _make_synthetic_data()

    def test_temperature_returns_valid_distribution(self) -> None:
        result = fit_temperature(self.p_dev, self.y_dev, self.p_test, self.y_test)

        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.method, "temperature")
        for arr in (result.probs_calibrated_dev, result.probs_calibrated_test):
            self.assertEqual(arr.shape[1], self.p_dev.shape[1])
            np.testing.assert_allclose(arr.sum(axis=1), np.ones(arr.shape[0]), atol=1e-6)
            self.assertTrue(np.all(arr > 0))
        self.assertIn("T", result.params)

    def test_platt_monotonic_in_predicted_class(self) -> None:
        scores = np.linspace(-3.0, 3.0, 60)
        targets = (scores >= 0.5).astype(float)

        a, b, _ = _platt_one_vs_rest(scores, targets)

        z_sorted = np.sort(scores)
        from calibration_methods import _sigmoid

        p = _sigmoid(a * z_sorted + b)
        diffs = np.diff(p)
        self.assertTrue(np.all(diffs >= -1e-6), f"non-monotonic: min diff={diffs.min()}")

    def test_isotonic_pav_monotonic_non_decreasing(self) -> None:
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=float)
        y = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=float)

        knot_x, knot_y = _pav_isotonic(x, y)

        diffs = np.diff(knot_y)
        self.assertTrue(np.all(diffs >= -1e-9), f"PAV not non-decreasing: {knot_y}")
        self.assertEqual(len(knot_x), len(knot_y))

    def test_isotonic_row_normalize_sums_to_one(self) -> None:
        result = fit_isotonic(self.p_dev, self.y_dev, self.p_test, self.y_test)

        for arr in (result.probs_calibrated_dev, result.probs_calibrated_test):
            np.testing.assert_allclose(arr.sum(axis=1), np.ones(arr.shape[0]), atol=1e-6)
            self.assertTrue(np.all(arr >= 0))

    def test_vector_scaling_recovers_identity_when_well_calibrated(self) -> None:
        rng = np.random.default_rng(20260520)
        n, k = 1200, 3
        logits = rng.normal(scale=0.8, size=(n, k))
        m = logits.max(axis=1, keepdims=True)
        probs = np.exp(logits - m) / np.exp(logits - m).sum(axis=1, keepdims=True)
        y = np.array([rng.choice(k, p=probs[i]) for i in range(n)])

        result = fit_vector_scaling(probs, y, probs, y, lr=0.02, n_steps=400, patience=80)

        W = np.array(result.params["W"], dtype=float)
        b = np.array(result.params["b"], dtype=float)
        self.assertTrue(np.all(np.abs(W - 1.0) < 0.5), f"W far from identity: {W}")
        self.assertTrue(np.all(np.abs(b) < 0.5), f"b far from zero: {b}")

    def test_conformal_coverage_at_least_1_minus_alpha(self) -> None:
        alpha = 0.1
        result = fit_conformal_split(
            self.p_dev, self.y_dev, self.p_dev, self.y_dev, alpha=alpha
        )

        coverage = result.extras["coverage_dev"]
        self.assertGreaterEqual(
            coverage,
            (1.0 - alpha) - 0.05,
            f"coverage {coverage} below target {1.0 - alpha}",
        )
        self.assertLessEqual(coverage, 1.0)
        self.assertEqual(result.extras["target_coverage"], round(1.0 - alpha, 4))

    def test_compute_calibration_metrics_handles_perfect_predictions(self) -> None:
        n, k = 50, 3
        y = np.array([i % k for i in range(n)])
        probs = np.zeros((n, k), dtype=float)
        probs[np.arange(n), y] = 1.0

        metrics = compute_calibration_metrics(y, probs)

        self.assertAlmostEqual(metrics["accuracy"], 1.0, places=4)
        self.assertAlmostEqual(metrics["brier"], 0.0, places=4)
        self.assertLess(metrics["nll"], 1e-3)
        self.assertLess(metrics["ece"], 1e-3)

    def test_reliability_bins_count_sums_to_total(self) -> None:
        bins = reliability_bins(self.y_test, self.p_test, n_bins=10)

        self.assertEqual(len(bins), 10)
        total_count = sum(int(row["count"]) for row in bins)
        self.assertEqual(total_count, len(self.y_test))
        for row in bins:
            self.assertGreaterEqual(row["bin_upper"], row["bin_lower"])

    def test_run_calibration_comparison_writes_three_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "calibration_comparison"
            summary = run_calibration_comparison(
                self.p_dev, self.y_dev, self.p_test, self.y_test,
                methods=SUPPORTED_METHODS, out_dir=out_dir, head="pairwise",
            )

            csv_path = out_dir / "calibration_methods_comparison.csv"
            json_path = out_dir / "calibration_per_method.json"
            pdf_path = out_dir / "reliability_diagram.pdf"
            png_path = out_dir / "reliability_diagram.png"

            self.assertTrue(csv_path.exists(), f"missing: {csv_path}")
            self.assertTrue(json_path.exists(), f"missing: {json_path}")
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                self.assertIsNone(summary["artifacts"]["diagram_pdf"])
                self.assertIsNone(summary["artifacts"]["diagram_png"])
            else:
                self.assertTrue(
                    pdf_path.exists() or png_path.exists(),
                    "expected reliability_diagram.pdf or .png",
                )

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["head"], "pairwise")
            for method in SUPPORTED_METHODS:
                self.assertIn(method, payload["results"])

            csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(csv_lines), 1 + len(SUPPORTED_METHODS) * 2)
            self.assertEqual(set(summary["methods"]), set(SUPPORTED_METHODS))

    def test_all_methods_preserve_argmax_on_extreme_logits(self) -> None:
        n_classes = 3
        probs = np.zeros((20, n_classes), dtype=float)
        argmax_targets = np.array([i % n_classes for i in range(20)])
        for i, k in enumerate(argmax_targets):
            other = (np.arange(n_classes) != k)
            probs[i, k] = 0.98
            probs[i, other] = 0.01
        y = argmax_targets.copy()

        for method in SUPPORTED_METHODS:
            with self.subTest(method=method):
                result = METHOD_DISPATCH[method](probs, y, probs, y)
                pred = result.probs_calibrated_test.argmax(axis=1)
                np.testing.assert_array_equal(
                    pred, argmax_targets, err_msg=f"argmax altered by {method}"
                )


if __name__ == "__main__":
    unittest.main()
