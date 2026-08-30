"""Calibration methods comparison for BEA-Judge pairwise predictions.

Implements 5 calibration methods with a unified interface:
- Temperature Scaling (baseline, wraps existing calibrate_temperature)
- Platt Scaling (one-vs-rest sigmoid logistic regression, Newton-Raphson)
- Isotonic Regression (one-vs-rest PAV + row-normalize)
- Vector Scaling (parametric W * logits + b, gradient descent on dev NLL)
- Split Conformal Prediction (nonconformity = 1 - p(y_true))

All implementations are pure numpy; no sklearn/scipy dependency.
Reuses temperature_scale, calibrate_temperature, brier_score, ece_score,
negative_log_likelihood, and confidence_for_predictions from bea_judge_train.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bea_judge_train import (
    brier_score,
    calibrate_temperature,
    confidence_for_predictions,
    ece_score,
    negative_log_likelihood,
    temperature_scale,
)


SUPPORTED_METHODS: Tuple[str, ...] = (
    "temperature",
    "platt",
    "isotonic",
    "vector_scaling",
    "conformal",
)

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "temperature": "Temperature Scaling",
    "platt": "Platt Scaling",
    "isotonic": "Isotonic Regression",
    "vector_scaling": "Vector Scaling",
    "conformal": "Conformal Prediction",
}


@dataclasses.dataclass
class CalibrationResult:
    method: str
    params: Dict[str, Any]
    probs_calibrated_dev: np.ndarray
    probs_calibrated_test: np.ndarray
    metrics_dev: Dict[str, float]
    metrics_test: Dict[str, float]
    reliability_bins_test: List[Dict[str, float]]
    extras: Dict[str, Any] = dataclasses.field(default_factory=dict)


def _logits_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(probs, eps, 1.0))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _row_normalize(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    safe = np.clip(probs, eps, None)
    return safe / safe.sum(axis=1, keepdims=True)


def reliability_bins(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Equal-width binning consistent with ece_score_from_predictions."""
    if len(y_true) == 0:
        return []
    pred = probs.argmax(axis=1)
    conf = confidence_for_predictions(probs, pred)
    bins: List[Dict[str, float]] = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin_lower": round(float(lo), 4),
                    "bin_upper": round(float(hi), 4),
                    "count": 0,
                    "mean_confidence": 0.0,
                    "accuracy": 0.0,
                }
            )
            continue
        bins.append(
            {
                "bin_lower": round(float(lo), 4),
                "bin_upper": round(float(hi), 4),
                "count": count,
                "mean_confidence": round(float(conf[mask].mean()), 6),
                "accuracy": round(float((pred[mask] == y_true[mask]).mean()), 6),
            }
        )
    return bins


def compute_calibration_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute ECE/MCE/Brier/NLL/Accuracy on a (probs, y_true) pair."""
    if len(y_true) == 0:
        return {"ece": 0.0, "mce": 0.0, "brier": 0.0, "nll": 0.0, "accuracy": 0.0}
    pred = probs.argmax(axis=1)
    conf = confidence_for_predictions(probs, pred)
    ece = ece_score(y_true, probs, bins=n_bins)
    brier = brier_score(y_true, probs, probs.shape[1])
    nll = negative_log_likelihood(y_true, probs)
    acc = float((pred == y_true).mean())
    mce = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b < n_bins - 1:
            mask = (conf >= lo) & (conf < hi)
        else:
            mask = (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        gap = abs(float((pred[mask] == y_true[mask]).mean()) - float(conf[mask].mean()))
        if gap > mce:
            mce = gap
    return {
        "ece": round(float(ece), 6),
        "mce": round(float(mce), 6),
        "brier": round(float(brier), 6),
        "nll": round(float(nll), 6),
        "accuracy": round(float(acc), 6),
    }


def fit_temperature(
    p_dev: np.ndarray,
    y_dev: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
) -> CalibrationResult:
    """Wrap calibrate_temperature; uses NLL+ECE+0.25*Brier objective on dev."""
    info = calibrate_temperature(p_dev, y_dev)
    T = float(info["temperature"])
    p_dev_cal = temperature_scale(p_dev, T)
    p_test_cal = temperature_scale(p_test, T)
    return CalibrationResult(
        method="temperature",
        params={"T": T, "selection_metric": info["selection_metric"]},
        probs_calibrated_dev=p_dev_cal,
        probs_calibrated_test=p_test_cal,
        metrics_dev=compute_calibration_metrics(y_dev, p_dev_cal),
        metrics_test=compute_calibration_metrics(y_test, p_test_cal),
        reliability_bins_test=reliability_bins(y_test, p_test_cal),
        extras={},
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _platt_one_vs_rest(
    scores: np.ndarray,
    targets: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[float, float, bool]:
    """Fit Platt sigmoid p = sigmoid(a*score + b) via Newton-Raphson.

    Returns (a, b, converged). Falls back to gradient descent on failure.
    """
    a, b = 1.0, 0.0
    n = len(scores)
    for _ in range(max_iter):
        z = a * scores + b
        p = _sigmoid(z)
        diff = p - targets
        ga = float((diff * scores).sum())
        gb = float(diff.sum())
        w = p * (1.0 - p)
        haa = float((w * scores * scores).sum()) + 1e-6
        hab = float((w * scores).sum())
        hbb = float(w.sum()) + 1e-6
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a_new = a - da
        b_new = b - db
        if abs(da) < tol and abs(db) < tol:
            a, b = a_new, b_new
            return a, b, True
        a, b = a_new, b_new
    if np.isfinite(a) and np.isfinite(b):
        return a, b, True
    a, b, lr = 1.0, 0.0, 0.05
    for _ in range(200):
        z = a * scores + b
        p = _sigmoid(z)
        diff = p - targets
        a -= lr * float((diff * scores).sum()) / max(1, n)
        b -= lr * float(diff.sum()) / max(1, n)
    return a, b, False


def fit_platt(
    p_dev: np.ndarray,
    y_dev: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
) -> CalibrationResult:
    """One-vs-rest Platt scaling on log-probabilities, row-normalized."""
    n_classes = p_dev.shape[1]
    logits_dev = _logits_from_probs(p_dev)
    logits_test = _logits_from_probs(p_test)
    params: Dict[str, Any] = {"a": [], "b": [], "fallback": []}
    out_dev = np.zeros_like(p_dev)
    out_test = np.zeros_like(p_test)
    for k in range(n_classes):
        targets = (y_dev == k).astype(float)
        a, b, ok = _platt_one_vs_rest(logits_dev[:, k], targets)
        params["a"].append(round(float(a), 6))
        params["b"].append(round(float(b), 6))
        params["fallback"].append(not ok)
        out_dev[:, k] = _sigmoid(a * logits_dev[:, k] + b)
        out_test[:, k] = _sigmoid(a * logits_test[:, k] + b)
    out_dev = _row_normalize(out_dev)
    out_test = _row_normalize(out_test)
    return CalibrationResult(
        method="platt",
        params=params,
        probs_calibrated_dev=out_dev,
        probs_calibrated_test=out_test,
        metrics_dev=compute_calibration_metrics(y_dev, out_dev),
        metrics_test=compute_calibration_metrics(y_test, out_test),
        reliability_bins_test=reliability_bins(y_test, out_test),
        extras={},
    )


def _pav_isotonic(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pool Adjacent Violators on (x, y).

    Returns (knot_x, knot_y) defining a piecewise-constant non-decreasing
    function with x sorted ascending. Ties on x are merged by averaging y.
    """
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ys = y[order].astype(float)
    weights = np.ones_like(ys)
    if len(xs) > 1:
        unique_x: List[float] = [float(xs[0])]
        unique_y: List[float] = [float(ys[0])]
        unique_w: List[float] = [1.0]
        for xi, yi in zip(xs[1:], ys[1:]):
            if xi == unique_x[-1]:
                w_new = unique_w[-1] + 1.0
                unique_y[-1] = (unique_y[-1] * unique_w[-1] + float(yi)) / w_new
                unique_w[-1] = w_new
            else:
                unique_x.append(float(xi))
                unique_y.append(float(yi))
                unique_w.append(1.0)
        xs = np.asarray(unique_x, dtype=float)
        ys = np.asarray(unique_y, dtype=float)
        weights = np.asarray(unique_w, dtype=float)
    stack_y: List[float] = []
    stack_w: List[float] = []
    stack_count: List[int] = []
    for yi, wi in zip(ys, weights):
        stack_y.append(float(yi))
        stack_w.append(float(wi))
        stack_count.append(1)
        while len(stack_y) >= 2 and stack_y[-2] > stack_y[-1]:
            wi2 = stack_w.pop()
            yi2 = stack_y.pop()
            ci2 = stack_count.pop()
            wi1 = stack_w.pop()
            yi1 = stack_y.pop()
            ci1 = stack_count.pop()
            w = wi1 + wi2
            stack_y.append((yi1 * wi1 + yi2 * wi2) / w)
            stack_w.append(w)
            stack_count.append(ci1 + ci2)
    fitted: List[float] = []
    for yi, ci in zip(stack_y, stack_count):
        fitted.extend([yi] * ci)
    fitted_arr = np.asarray(fitted[: len(xs)], dtype=float)
    return xs, fitted_arr


def _isotonic_predict(knot_x: np.ndarray, knot_y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(knot_x) == 0:
        return np.zeros_like(x, dtype=float)
    return np.interp(x, knot_x, knot_y, left=float(knot_y[0]), right=float(knot_y[-1]))


def fit_isotonic(
    p_dev: np.ndarray,
    y_dev: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
) -> CalibrationResult:
    """One-vs-rest isotonic regression via PAV, then row-normalize."""
    n_classes = p_dev.shape[1]
    out_dev = np.zeros_like(p_dev)
    out_test = np.zeros_like(p_test)
    knots_per_class: List[Dict[str, List[float]]] = []
    for k in range(n_classes):
        targets = (y_dev == k).astype(float)
        knot_x, knot_y = _pav_isotonic(p_dev[:, k], targets)
        out_dev[:, k] = _isotonic_predict(knot_x, knot_y, p_dev[:, k])
        out_test[:, k] = _isotonic_predict(knot_x, knot_y, p_test[:, k])
        knots_per_class.append(
            {
                "knot_x": [round(float(v), 6) for v in knot_x.tolist()],
                "knot_y": [round(float(v), 6) for v in knot_y.tolist()],
            }
        )
    out_dev = _row_normalize(out_dev)
    out_test = _row_normalize(out_test)
    return CalibrationResult(
        method="isotonic",
        params={"knots_per_class": knots_per_class},
        probs_calibrated_dev=out_dev,
        probs_calibrated_test=out_test,
        metrics_dev=compute_calibration_metrics(y_dev, out_dev),
        metrics_test=compute_calibration_metrics(y_test, out_test),
        reliability_bins_test=reliability_bins(y_test, out_test),
        extras={},
    )


def fit_vector_scaling(
    p_dev: np.ndarray,
    y_dev: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    lr: float = 0.05,
    n_steps: int = 200,
    patience: int = 20,
) -> CalibrationResult:
    """Parametric logits' = W * logits + b (W diagonal). Gradient descent on dev NLL.

    Uses a held-out 10% slice of dev for early stopping.
    """
    n_classes = p_dev.shape[1]
    logits_dev = _logits_from_probs(p_dev)
    logits_test = _logits_from_probs(p_test)
    rng = np.random.default_rng(20260520)
    n_dev = len(y_dev)
    idx = rng.permutation(n_dev)
    cut = max(1, int(round(n_dev * 0.1)))
    val_idx = idx[:cut]
    train_idx = idx[cut:]
    if len(train_idx) == 0:
        train_idx = idx
        val_idx = idx
    W = np.ones(n_classes, dtype=float)
    b = np.zeros(n_classes, dtype=float)
    best = (float("inf"), W.copy(), b.copy())
    no_improve = 0
    for _ in range(n_steps):
        scaled = logits_dev[train_idx] * W + b
        p = _softmax(scaled)
        target = np.eye(n_classes)[y_dev[train_idx]]
        diff = p - target
        gW = (diff * logits_dev[train_idx]).mean(axis=0)
        gb = diff.mean(axis=0)
        W -= lr * gW
        b -= lr * gb
        val_scaled = logits_dev[val_idx] * W + b
        val_p = _softmax(val_scaled)
        val_nll = negative_log_likelihood(y_dev[val_idx], val_p)
        if val_nll < best[0] - 1e-6:
            best = (val_nll, W.copy(), b.copy())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    _, W_best, b_best = best
    out_dev = _softmax(logits_dev * W_best + b_best)
    out_test = _softmax(logits_test * W_best + b_best)
    return CalibrationResult(
        method="vector_scaling",
        params={"W": [round(float(v), 6) for v in W_best.tolist()],
                "b": [round(float(v), 6) for v in b_best.tolist()]},
        probs_calibrated_dev=out_dev,
        probs_calibrated_test=out_test,
        metrics_dev=compute_calibration_metrics(y_dev, out_dev),
        metrics_test=compute_calibration_metrics(y_test, out_test),
        reliability_bins_test=reliability_bins(y_test, out_test),
        extras={"early_stop_val_nll": round(float(best[0]), 6)},
    )


def fit_conformal_split(
    p_dev: np.ndarray,
    y_dev: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 0.1,
) -> CalibrationResult:
    """Split conformal prediction with nonconformity = 1 - p(y_true).

    Probabilities are not modified (test_probs == p_test); the calibration
    yields a threshold q_hat used to derive prediction sets and report
    coverage and average set size.
    """
    n = len(y_dev)
    nonconformity = 1.0 - p_dev[np.arange(n), y_dev]
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = max(1, min(n, rank))
    q_hat = float(np.sort(nonconformity)[rank - 1])
    test_sets = (p_test >= (1.0 - q_hat))
    test_set_sizes = test_sets.sum(axis=1)
    in_set = test_sets[np.arange(len(y_test)), y_test]
    coverage = float(in_set.mean()) if len(y_test) else 0.0
    set_size_avg = float(test_set_sizes.mean()) if len(y_test) else 0.0
    dev_sets = (p_dev >= (1.0 - q_hat))
    dev_in_set = dev_sets[np.arange(n), y_dev]
    dev_coverage = float(dev_in_set.mean()) if n else 0.0
    return CalibrationResult(
        method="conformal",
        params={"q_hat": round(q_hat, 6), "alpha": round(alpha, 4), "rank": rank},
        probs_calibrated_dev=p_dev.copy(),
        probs_calibrated_test=p_test.copy(),
        metrics_dev=compute_calibration_metrics(y_dev, p_dev),
        metrics_test=compute_calibration_metrics(y_test, p_test),
        reliability_bins_test=reliability_bins(y_test, p_test),
        extras={
            "coverage_dev": round(dev_coverage, 6),
            "coverage_test": round(coverage, 6),
            "set_size_avg_test": round(set_size_avg, 6),
            "target_coverage": round(1.0 - alpha, 4),
        },
    )


METHOD_DISPATCH = {
    "temperature": fit_temperature,
    "platt": fit_platt,
    "isotonic": fit_isotonic,
    "vector_scaling": fit_vector_scaling,
    "conformal": fit_conformal_split,
}


def plot_reliability_diagram(
    results: Sequence[CalibrationResult],
    out_path: Path,
    n_bins: int = 10,
) -> Optional[Tuple[Path, Path]]:
    """Plot a 2x3 grid of reliability diagrams. Returns (pdf_path, png_path).

    Returns None if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0))
    axes = axes.flatten()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(results):
            ax.axis("off")
            continue
        result = results[ax_idx]
        bins_data = result.reliability_bins_test
        accs = np.array([row.get("accuracy", 0.0) for row in bins_data], dtype=float)
        confs = np.array([row.get("mean_confidence", 0.0) for row in bins_data], dtype=float)
        counts = np.array([row.get("count", 0) for row in bins_data], dtype=float)
        total = max(1.0, counts.sum())
        weights = counts / total
        ax.bar(
            centers,
            accs,
            width=1.0 / n_bins * 0.9,
            color="#3a7bd5",
            alpha=0.85,
            edgecolor="white",
            label="Accuracy",
        )
        ax.bar(
            centers,
            np.maximum(confs - accs, 0.0),
            width=1.0 / n_bins * 0.9,
            bottom=accs,
            color="#d54a3a",
            alpha=0.55,
            edgecolor="white",
            label="Gap",
        )
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.0)
        ax2 = ax.twinx()
        ax2.bar(centers, weights, width=1.0 / n_bins * 0.9, color="grey", alpha=0.18)
        ax2.set_ylim(0.0, max(0.5, float(weights.max()) * 1.2))
        ax2.set_yticks([])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted Confidence")
        ax.set_ylabel("Accuracy in Bin")
        ece = result.metrics_test.get("ece", 0.0)
        brier = result.metrics_test.get("brier", 0.0)
        ax.set_title(
            f"{METHOD_DISPLAY_NAMES.get(result.method, result.method)}\n"
            f"ECE={ece:.4f} | Brier={brier:.4f}"
        )
        if ax_idx == 0:
            ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Reliability Diagrams (test split, BEA-Judge pairwise head)", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    png_path = out_path.with_suffix(".png")
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def _result_to_summary_dict(result: CalibrationResult) -> Dict[str, Any]:
    return {
        "method": result.method,
        "params": result.params,
        "metrics_dev": result.metrics_dev,
        "metrics_test": result.metrics_test,
        "reliability_bins_test": result.reliability_bins_test,
        "extras": result.extras,
    }


def _write_csv(results: Sequence[CalibrationResult], path: Path) -> None:
    fieldnames = [
        "method", "split", "ece", "mce", "brier", "nll", "accuracy",
        "coverage", "set_size_avg",
    ]
    rows: List[Dict[str, Any]] = []
    for result in results:
        for split, metrics in (("dev", result.metrics_dev), ("test", result.metrics_test)):
            row: Dict[str, Any] = {
                "method": result.method,
                "split": split,
                "ece": metrics.get("ece"),
                "mce": metrics.get("mce"),
                "brier": metrics.get("brier"),
                "nll": metrics.get("nll"),
                "accuracy": metrics.get("accuracy"),
                "coverage": "",
                "set_size_avg": "",
            }
            if result.method == "conformal":
                if split == "dev":
                    row["coverage"] = result.extras.get("coverage_dev", "")
                else:
                    row["coverage"] = result.extras.get("coverage_test", "")
                    row["set_size_avg"] = result.extras.get("set_size_avg_test", "")
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_calibration_comparison(
    p_dev: np.ndarray,
    y_dev: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    methods: Sequence[str] = SUPPORTED_METHODS,
    out_dir: Optional[Path] = None,
    head: str = "pairwise",
) -> Dict[str, Any]:
    """Run all requested methods, write CSV/JSON/PDF artifacts, return summary."""
    if p_dev.shape[1] != p_test.shape[1]:
        raise ValueError("p_dev and p_test must share class dimension")
    results: List[CalibrationResult] = []
    for method in methods:
        if method not in METHOD_DISPATCH:
            raise ValueError(
                f"unsupported calibration method: {method}; "
                f"supported = {sorted(METHOD_DISPATCH)}"
            )
        result = METHOD_DISPATCH[method](p_dev, y_dev, p_test, y_test)
        results.append(result)
    summary: Dict[str, Any] = {
        "head": head,
        "methods": list(methods),
        "results": {result.method: _result_to_summary_dict(result) for result in results},
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "calibration_methods_comparison.csv"
        json_path = out_dir / "calibration_per_method.json"
        diagram_stem = out_dir / "reliability_diagram"
        _write_csv(results, csv_path)
        json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        diagram_paths = plot_reliability_diagram(results, diagram_stem)
        summary["artifacts"] = {
            "csv": str(csv_path),
            "json": str(json_path),
            "diagram_pdf": str(diagram_paths[0]) if diagram_paths else None,
            "diagram_png": str(diagram_paths[1]) if diagram_paths else None,
        }
    return summary




