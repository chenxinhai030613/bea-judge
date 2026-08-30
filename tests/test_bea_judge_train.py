import sys
import json
import unittest
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bea_judge_train import (
    JudgeOutputFeatures,
    apply_factuality_threshold_policy,
    apply_dataset_temperature_policy,
    apply_pairwise_tie_policy,
    base_pairwise_features,
    group_by_split,
    calibrate_temperature,
    evaluate_head_on_split,
    factuality_weight_candidates,
    factuality_feature_dict,
    load_judge_output_features,
    make_calibrated_rows,
    pairwise_feature_dict,
    path_relative_to_root,
    select_pairwise_tie_policy,
    select_dataset_overlay_tie_policy,
    select_factuality_threshold_policy,
    select_dataset_temperature_policy,
    sample_weight_vector,
    limit_samples,
    SoftmaxClassifier,
    HyperParams,
    validate_judge_output_coverage,
    validate_calibrated_rows_schema,
    risk_scores,
    select_review_threshold,
    select_samples,
    train_one_head,
    update_experiment_config,
    temperature_scale,
    PAIRWISE_LABELS,
)


class BeaJudgeCalibrationTest(unittest.TestCase):
    def test_limit_samples_preserves_order_and_applies_cap(self) -> None:
        samples = [{"id": 1}, {"id": 2}, {"id": 3}]

        self.assertEqual(limit_samples(samples, None), samples)
        self.assertEqual(limit_samples(samples, 2), samples[:2])

    def test_temperature_scale_preserves_probability_shape(self) -> None:
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=float)

        scaled = temperature_scale(probs, temperature=1.5)

        self.assertEqual(scaled.shape, probs.shape)
        np.testing.assert_allclose(scaled.sum(axis=1), np.ones(2), atol=1e-8)
        self.assertTrue(np.all(scaled > 0))

    def test_load_judge_output_features_accepts_qlora_backend(self) -> None:
        rows = [
            {
                "id": "qlora-1",
                "judge_backend": "m_prometheus_qlora",
                "parse_status": "ok",
                "pred_label": "A>B",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "base_scores.json"
            path.write_text(json.dumps(rows), encoding="utf-8")

            features = load_judge_output_features(path)

        self.assertEqual(features.source_counts["m_prometheus_qlora"], 1)
        self.assertIn("qlora-1", features.rows)

    def test_sample_weight_vector_combines_class_and_source_weights(self) -> None:
        samples = [
            {"human_label": "supported", "dataset": "ares_nq"},
            {"human_label": "unsupported", "dataset": "ragtruth"},
        ]

        weights = sample_weight_vector(
            samples,
            class_weights={"unsupported": 2.0},
            source_weights={"ragtruth": 1.5},
        )

        np.testing.assert_allclose(weights, np.array([1.0, 3.0]))

    def test_factuality_weight_candidates_use_v2_search_grid(self) -> None:
        candidates = factuality_weight_candidates()
        unsupported = {row["class_weights"]["unsupported"] for row in candidates}
        ragtruth = {row["source_weights"]["ragtruth"] for row in candidates}

        self.assertEqual(unsupported, {1.5, 2.0, 2.25, 2.5})
        self.assertEqual(ragtruth, {1.25, 1.5, 1.75})

    def test_softmax_unit_weights_match_unweighted_training(self) -> None:
        x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
        y = np.array([0, 1, 0], dtype=int)
        params = HyperParams(learning_rate=0.01, batch_size=2, l2=0.0, epochs=3)
        unweighted = SoftmaxClassifier(2, 2, seed=7)
        weighted = SoftmaxClassifier(2, 2, seed=7)

        unweighted.fit(x, y, params, seed=11)
        weighted.fit(x, y, params, seed=11, sample_weight=np.ones(len(y)))

        np.testing.assert_allclose(unweighted.weights, weighted.weights)
        np.testing.assert_allclose(unweighted.bias, weighted.bias)

    def test_factuality_threshold_policy_only_changes_ragtruth_rows(self) -> None:
        labels = ["supported", "unsupported"]
        probs = np.array([[0.62, 0.38], [0.62, 0.38]], dtype=float)
        datasets = ["ragtruth", "ares_nq"]
        policy = {"enabled": True, "dataset": "ragtruth", "unsupported_threshold": 0.35}

        pred = apply_factuality_threshold_policy(probs, labels, datasets, policy)

        self.assertEqual(pred.tolist(), [1, 0])

    def test_factuality_feature_dict_keeps_anchor_features_audit_only(self) -> None:
        sample = {
            "id": "fact-anchor-audit",
            "dataset": "ragtruth",
            "task_type": "factuality_rag",
            "human_label": "unsupported",
            "context": "The report says the mission launched from Florida in 2021.",
            "reference": "",
            "answer_a": "The mission launched from Florida in 2021. It later landed in Oslo with Dr. Vera Lang.",
            "answer_b": None,
            "metadata": {"factuality_task_form": "single_answer"},
        }

        features = factuality_feature_dict(sample)

        self.assertNotIn("evidence_low_support_anchor_sentence_ratio_a", features)
        self.assertNotIn("evidence_max_low_support_anchor_gap_a", features)
        self.assertNotIn("evidence_anchored_hallucination_severity_a", features)
        self.assertIn("evidence_local_hallucination_risk_a", features)

    def test_select_factuality_threshold_policy_can_improve_ragtruth_unsupported(self) -> None:
        labels = ["supported", "unsupported"]
        probs = np.array(
            [
                [0.62, 0.38],
                [0.64, 0.36],
                [0.90, 0.10],
                [0.85, 0.15],
                [0.62, 0.38],
                [0.64, 0.36],
                [0.90, 0.10],
                [0.85, 0.15],
                [0.62, 0.38],
                [0.64, 0.36],
            ],
            dtype=float,
        )
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
        datasets = ["ragtruth"] * len(y_true)

        policy = select_factuality_threshold_policy(probs, y_true, labels, datasets)
        pred = apply_factuality_threshold_policy(probs, labels, datasets, policy)

        self.assertTrue(policy["enabled"])
        self.assertGreaterEqual((pred == y_true).mean(), 0.9)

    def test_calibrate_temperature_returns_candidate_with_lowest_objective(self) -> None:
        probs = np.array([[0.75, 0.2, 0.05], [0.2, 0.65, 0.15], [0.2, 0.25, 0.55]], dtype=float)
        y_true = np.array([0, 1, 2], dtype=int)

        result = calibrate_temperature(probs, y_true, candidates=[0.5, 1.0, 2.0])

        self.assertIn(result["temperature"], {0.5, 1.0, 2.0})
        self.assertEqual(len(result["candidates"]), 3)
        best = min(result["candidates"], key=lambda row: row["objective"])
        self.assertEqual(result["temperature"], best["temperature"])

    def test_select_review_threshold_uses_dev_errors(self) -> None:
        confidence = np.array([0.95, 0.72, 0.44, 0.31], dtype=float)
        correct = np.array([True, False, False, True])

        risk = risk_scores(confidence)
        result = select_review_threshold(risk, correct, target_error_recall=1.0)

        self.assertGreaterEqual(result["error_recall"], 1.0)
        self.assertGreater(result["review_rate"], 0.0)
        self.assertGreaterEqual(result["threshold"], 0.28)
        self.assertEqual(result["risk_signal"], "1 - confidence")

    def test_make_calibrated_rows_emits_required_schema(self) -> None:
        samples = [
            {"id": "s1", "dataset": "unit", "task_type": "open_qa", "human_label": "A>B"},
            {"id": "s2", "dataset": "unit", "task_type": "open_qa", "human_label": "B>A"},
        ]
        probs = np.array([[0.8, 0.1, 0.1], [0.3, 0.45, 0.25]], dtype=float)
        labels = ["A>B", "B>A", "Tie"]

        rows = make_calibrated_rows(samples, labels, probs, review_threshold=0.5, head_name="pairwise")

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIn("final_score", row)
            self.assertIn("pairwise_label", row)
            self.assertIn("confidence", row)
            self.assertIn("risk_score", row)
            self.assertIn("review_flag", row)
            self.assertIn("label_probabilities", row)
        self.assertFalse(rows[0]["review_flag"])
        self.assertTrue(rows[1]["review_flag"])
        validate_calibrated_rows_schema(rows)

    def test_make_calibrated_rows_uses_policy_prediction_confidence(self) -> None:
        samples = [
            {"id": "s1", "dataset": "unit", "task_type": "open_qa", "human_label": "Tie"},
        ]
        probs = np.array([[0.43, 0.42, 0.15]], dtype=float)
        labels = ["A>B", "B>A", "Tie"]
        pred_indices = np.array([2], dtype=int)

        rows = make_calibrated_rows(
            samples,
            labels,
            probs,
            review_threshold=0.5,
            head_name="pairwise",
            pred_indices=pred_indices,
        )

        self.assertEqual(rows[0]["predicted_label"], "Tie")
        self.assertEqual(rows[0]["confidence"], 0.15)
        self.assertEqual(rows[0]["final_score"], 0.5)
        self.assertTrue(rows[0]["review_flag"])

    def test_apply_pairwise_tie_policy_overrides_close_ab_margin(self) -> None:
        probs = np.array(
            [
                [0.43, 0.42, 0.15],
                [0.70, 0.10, 0.20],
            ],
            dtype=float,
        )
        policy = {"enabled": True, "min_tie_probability": 0.15, "max_ab_margin": 0.02}

        pred = apply_pairwise_tie_policy(probs, PAIRWISE_LABELS, policy)

        self.assertEqual(pred.tolist(), [2, 0])

    def test_apply_pairwise_tie_policy_can_use_base_margin(self) -> None:
        probs = np.array(
            [
                [0.43, 0.42, 0.15],
                [0.43, 0.42, 0.15],
            ],
            dtype=float,
        )
        policy = {
            "enabled": True,
            "min_tie_probability": 0.15,
            "max_ab_margin": 0.02,
            "max_base_margin": 0.25,
        }

        pred = apply_pairwise_tie_policy(
            probs,
            PAIRWISE_LABELS,
            policy,
            base_margins=np.array([0.1, 0.9], dtype=float),
        )

        self.assertEqual(pred.tolist(), [2, 0])

    def test_apply_pairwise_tie_policy_can_use_dataset_specific_policy(self) -> None:
        probs = np.array(
            [
                [0.43, 0.42, 0.15],
                [0.43, 0.42, 0.15],
            ],
            dtype=float,
        )
        policy = {
            "enabled": True,
            "min_tie_probability": None,
            "max_ab_margin": None,
            "dataset_policies": {
                "hard": {"min_tie_probability": 0.15, "max_ab_margin": 0.02, "max_base_margin": None}
            },
        }

        pred = apply_pairwise_tie_policy(probs, PAIRWISE_LABELS, policy, datasets=["hard", "easy"])

        self.assertEqual(pred.tolist(), [2, 0])

    def test_select_pairwise_tie_policy_can_improve_tie_recall(self) -> None:
        probs = np.array(
            [
                [0.43, 0.42, 0.15],
                [0.44, 0.43, 0.13],
                [0.70, 0.10, 0.20],
            ],
            dtype=float,
        )
        y_true = np.array([2, 2, 0], dtype=int)

        policy = select_pairwise_tie_policy(probs, y_true, PAIRWISE_LABELS)
        pred = apply_pairwise_tie_policy(probs, PAIRWISE_LABELS, policy)

        self.assertTrue(policy["enabled"])
        self.assertGreaterEqual(policy["selected_metrics"]["tie_recall"], 0.5)
        self.assertEqual(pred[0], 2)

    def test_dataset_overlay_tie_policy_expands_selected_dataset_only(self) -> None:
        probs = np.array(
            [[0.56, 0.04, 0.40]] * 10
            + [[0.82, 0.04, 0.14]] * 30
            + [[0.70, 0.20, 0.10]] * 10,
            dtype=float,
        )
        y_true = np.array([2] * 10 + [0] * 30 + [0] * 10, dtype=int)
        datasets = ["helpsteer2"] * 40 + ["other"] * 10
        base_policy = {
            "enabled": True,
            "min_tie_probability": 0.08,
            "max_ab_margin": 0.40,
            "max_base_margin": None,
            "dataset_policies": {},
            "metrics": {},
        }

        policy = select_dataset_overlay_tie_policy(
            probs,
            y_true,
            PAIRWISE_LABELS,
            datasets,
            base_policy=base_policy,
            base_margins=None,
            macro_floor=0.0,
            accuracy_floor=0.0,
            ece_ceiling=1.0,
            baseline_tie=0.0,
        )
        self.assertIsNotNone(policy)

        pred = apply_pairwise_tie_policy(probs, PAIRWISE_LABELS, policy, datasets=datasets)

        self.assertEqual(policy["overlay_dataset"], "helpsteer2")
        self.assertEqual(pred[:10].tolist(), [2] * 10)
        self.assertEqual(pred[40:].tolist(), [0] * 10)

    def test_dataset_temperature_policy_is_dev_selected(self) -> None:
        probs = np.array(
            [
                [0.70, 0.20, 0.10],
                [0.70, 0.20, 0.10],
                [0.20, 0.70, 0.10],
                [0.20, 0.70, 0.10],
                [0.34, 0.33, 0.33],
            ],
            dtype=float,
        )
        y_true = np.array([0, 0, 1, 1, 2], dtype=int)
        datasets = ["unit"] * 5

        policy = select_dataset_temperature_policy(probs, y_true, PAIRWISE_LABELS, datasets)
        adjusted = apply_dataset_temperature_policy(probs, datasets, policy)

        self.assertIn("enabled", policy)
        self.assertEqual(adjusted.shape, probs.shape)
        np.testing.assert_allclose(adjusted.sum(axis=1), np.ones(5), atol=1e-8)

    def test_factuality_head_keeps_dataset_temperature_disabled(self) -> None:
        samples = []
        labels = ["supported", "unsupported"]
        for split, count in (("train", 6), ("dev", 6)):
            for i in range(count):
                label = labels[i % 2]
                samples.append(
                    {
                        "id": f"{split}-{i}",
                        "dataset": "unit",
                        "task_type": "factuality_rag",
                        "split": split,
                        "human_label": label,
                        "prompt": "Where does water boil?",
                        "context": "Water boils at 100 degrees Celsius.",
                        "reference": "Water boils at 100 degrees Celsius.",
                        "answer_a": "Water boils at 100 degrees Celsius.",
                        "answer_b": None,
                        "metadata": {"factuality_task_form": "single_answer"},
                    }
                )
        splits = group_by_split(samples)

        head = train_one_head("factuality", splits["train"], splits["dev"], labels, factuality_feature_dict)

        self.assertFalse(head["dataset_temperature_policy"]["enabled"])
        self.assertEqual(head["dataset_temperature_policy"]["method"], "disabled_for_stable_factuality_head")

    def test_validate_calibrated_rows_schema_rejects_missing_fields(self) -> None:
        with self.assertRaises(ValueError):
            validate_calibrated_rows_schema([{"id": "s1"}])

    def test_update_experiment_config_rewrites_latest_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "experiment.json"
            run_dir = Path(tmpdir) / "model_outputs" / "bea_judge_unit"
            config_path.write_text(
                '{"latest_outputs":{"validation_report":"old","calibrated_results":"old","run_directory":"old"}}',
                encoding="utf-8",
            )
            update_experiment_config(config_path, run_dir)
            payload = config_path.read_text(encoding="utf-8")

        self.assertIn("bea_judge_unit", payload)
        self.assertIn("latest_validation_report.json", payload)
        self.assertIn("calibrated_results.json", payload)

    def test_path_relative_to_root_handles_relative_paths(self) -> None:
        result = path_relative_to_root(Path("datasets") / "processed" / "unit.json")

        self.assertEqual(result, str(Path("datasets") / "processed" / "unit.json"))

    def test_load_judge_output_features_uses_real_prometheus_rows(self) -> None:
        rows = [
            {
                "id": "s1",
                "judge_backend": "m_prometheus",
                "pred_label": "A>B",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
            },
            {
                "id": "s1-legacy",
                "judge_backend": "prometheus2",
                "pred_label": "A>B",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
            },
            {
                "id": "s2",
                "judge_backend": "heuristic_fallback",
                "pred_label": "B>A",
                "parsed_scores": {"score_a": 0.0, "score_b": 1.0},
            },
            {
                "id": "s3",
                "judge_backend": "prometheus2",
                "parse_status": "backend_error",
                "parsed_scores": {"score_a": None, "score_b": None},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "base_scores.json"
            path.write_text(json.dumps(rows), encoding="utf-8")

            features = load_judge_output_features(path)

        self.assertEqual(features.source_counts["m_prometheus"], 1)
        self.assertEqual(features.source_counts["prometheus2"], 1)
        self.assertEqual(features.source_counts["heuristic_fallback"], 1)
        self.assertEqual(features.source_counts["backend_error"], 1)
        self.assertIn("s1", features.rows)
        self.assertIn("s1-legacy", features.rows)
        self.assertNotIn("s2", features.rows)
        self.assertNotIn("s3", features.rows)

    def test_load_judge_output_features_rejects_non_list_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "latest_summary.json"
            path.write_text('{"sample_count": 0}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "base_scores.json list"):
                load_judge_output_features(path)

    def test_base_pairwise_features_requires_real_judge_output(self) -> None:
        features = JudgeOutputFeatures(
            rows={
                "s1": {
                    "pred_label": "B>A",
                    "score_a": 0.0,
                    "score_b": 1.0,
                    "judge_backend": "prometheus2",
                }
            },
            source_counts={"prometheus2": 1, "heuristic_fallback": 0, "backend_error": 0},
            path="unit",
        )

        row = base_pairwise_features({"id": "s1"}, features)

        self.assertEqual(row["base_score_a"], 0.0)
        self.assertEqual(row["base_score_b"], 1.0)
        self.assertEqual(row["base_score_diff"], -1.0)
        self.assertEqual(row["base_margin"], 1.0)
        self.assertEqual(row["base_pred_b"], 1.0)
        self.assertEqual(row["swap_available"], 0.0)

        with self.assertRaises(KeyError):
            base_pairwise_features({"id": "missing"}, features)

    def test_validate_judge_output_coverage_reports_missing_pairwise_ids(self) -> None:
        samples = [
            {"id": "s1", "human_label": "A>B"},
            {"id": "s2", "human_label": "B>A"},
            {"id": "fact-1", "human_label": "supported"},
        ]
        outputs = JudgeOutputFeatures(
            rows={
                "s1": {
                    "pred_label": "A>B",
                    "score_a": 1.0,
                    "score_b": 0.0,
                    "judge_backend": "m_prometheus",
                }
            },
            source_counts={"m_prometheus": 1, "prometheus2": 0, "heuristic_fallback": 0, "backend_error": 0},
            path="unit",
        )

        with self.assertRaisesRegex(ValueError, "missing 1 of 2 required pairwise base judge rows"):
            validate_judge_output_coverage(samples, outputs)

    def test_validate_judge_output_coverage_passes_when_pairwise_ids_are_covered(self) -> None:
        samples = [
            {"id": "s1", "human_label": "A>B"},
            {"id": "fact-1", "human_label": "supported"},
        ]
        outputs = JudgeOutputFeatures(
            rows={
                "s1": {
                    "pred_label": "A>B",
                    "score_a": 1.0,
                    "score_b": 0.0,
                    "judge_backend": "m_prometheus",
                }
            },
            source_counts={"m_prometheus": 1, "prometheus2": 0, "heuristic_fallback": 0, "backend_error": 0},
            path="unit",
        )

        report = validate_judge_output_coverage(samples, outputs)

        self.assertEqual(report["required_pairwise_rows"], 1)
        self.assertEqual(report["covered_pairwise_rows"], 1)
        self.assertEqual(report["missing_pairwise_rows"], 0)

    def test_pairwise_training_can_use_real_judge_output_features(self) -> None:
        samples = []
        rows = []
        labels = ["A>B", "B>A", "Tie"]
        for split, count in (("train", 9), ("dev", 6), ("test", 6)):
            for i in range(count):
                label = labels[i % len(labels)]
                sample_id = f"{split}-{i}"
                samples.append(
                    {
                        "id": sample_id,
                        "dataset": "unit",
                        "task_type": "open_qa",
                        "split": split,
                        "human_label": label,
                        "prompt": f"Prompt {i}",
                        "context": "",
                        "reference": "",
                        "answer_a": "A detailed answer.",
                        "answer_b": "A short answer.",
                        "metadata": {"scoring_system": "pairwise_preference"},
                    }
                )
                score_a, score_b = {
                    "A>B": (1.0, 0.0),
                    "B>A": (0.0, 1.0),
                    "Tie": (0.5, 0.5),
                }[label]
                rows.append(
                    {
                        "id": sample_id,
                        "judge_backend": "m_prometheus",
                        "parse_status": "ok",
                        "pred_label": label,
                        "parsed_scores": {"score_a": score_a, "score_b": score_b},
                    }
                )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "base_scores.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            outputs = load_judge_output_features(path)

        feature_fn = lambda sample: pairwise_feature_dict(sample, outputs)
        splits = group_by_split(samples)
        head = train_one_head(
            "pairwise",
            select_samples(splits["train"], PAIRWISE_LABELS),
            select_samples(splits["dev"], PAIRWISE_LABELS),
            PAIRWISE_LABELS,
            feature_fn,
        )
        test_report = evaluate_head_on_split(head, splits["test"], PAIRWISE_LABELS, feature_fn, "test")

        self.assertEqual(outputs.source_counts["m_prometheus"], len(samples))
        self.assertGreaterEqual(head["calibrated_dev_metrics"]["accuracy"], 0.0)
        self.assertEqual(test_report["count"], 6)

    def test_pairwise_feature_dict_includes_bias_risk_features(self) -> None:
        sample = {
            "id": "bias-feature",
            "dataset": "synthetic_perturbed",
            "task_type": "pairwise_bias",
            "split": "train",
            "human_label": "A>B",
            "prompt": "Explain calibration.",
            "context": "",
            "reference": "",
            "answer_a": "Short but correct.",
            "answer_b": "A much longer answer with many extra words that should not be preferred.",
            "metadata": {"bias_type": "length", "perturbation_applied": "length"},
        }
        outputs = JudgeOutputFeatures(
            rows={
                "bias-feature": {
                    "pred_label": "B>A",
                    "score_a": 0.0,
                    "score_b": 1.0,
                    "judge_backend": "m_prometheus",
                }
            },
            source_counts={"m_prometheus": 1, "prometheus2": 0, "heuristic_fallback": 0, "backend_error": 0},
            path="unit",
        )

        features = pairwise_feature_dict(sample, outputs)

        self.assertIn("bias_overall_risk", features)
        self.assertIn("bias_length_risk", features)
        self.assertGreater(features["bias_overall_risk"], 0.0)

    def test_factuality_feature_dict_includes_evidence_features(self) -> None:
        sample = {
            "id": "fact-feature",
            "dataset": "unit",
            "task_type": "factuality_rag",
            "split": "train",
            "human_label": "supported",
            "prompt": "Where does water boil at 100 degrees Celsius?",
            "context": "Water boils at 100 degrees Celsius at sea level.",
            "reference": "At sea level, water boils at 100 degrees Celsius.",
            "answer_a": "Water boils at 100 degrees Celsius at sea level.",
            "answer_b": None,
            "metadata": {"scoring_system": "single_answer_factuality", "factuality_task_form": "single_answer"},
        }

        features = factuality_feature_dict(sample)

        self.assertIn("evidence_context_support_a", features)
        self.assertIn("evidence_risk", features)
        self.assertGreater(features["evidence_context_support_a"], 0.0)
        for value in features.values():
            self.assertIsInstance(value, (int, float))


if __name__ == "__main__":
    unittest.main()
