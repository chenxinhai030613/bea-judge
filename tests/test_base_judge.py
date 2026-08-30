import json
import sys
import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from base_judge import (
    DEFAULT_M_PROMETHEUS_MODEL_PATH,
    JudgeConfig,
    JudgeBackend,
    MPrometheus3BBackend,
    MockPrometheusBackend,
    Prometheus2Backend,
    base_score_rows_for_disk,
    build_prometheus_direct_prompt,
    build_prometheus_pairwise_prompt,
    evaluate_samples,
    extract_prometheus_direct_scores,
    extract_prometheus_pairwise_label,
    is_valid_pairwise_score_row,
    load_judge_config,
    pairwise_samples,
    run_resumable_evaluation,
    split_base_score_rows,
)


class BaseJudgeOutputTest(unittest.TestCase):
    def test_load_judge_config_matches_repository_default(self) -> None:
        config = load_judge_config(ROOT / "configs" / "judge.json")

        self.assertEqual(config.name, "m_prometheus_3b_base")
        self.assertEqual(config.version, DEFAULT_M_PROMETHEUS_MODEL_PATH)
        self.assertEqual(config.backend, "m_prometheus")
        self.assertEqual(DEFAULT_M_PROMETHEUS_MODEL_PATH, "Unbabel/M-Prometheus-3B")
        self.assertEqual(config.model_path, "models/M-Prometheus-3B")
        self.assertEqual(config.prompt_template, "m_prometheus_pairwise_v1")
        self.assertFalse(config.allow_fallback)

    def test_evaluate_samples_returns_summary_and_detail_rows(self) -> None:
        samples = [
            {
                "id": "unit-1",
                "dataset": "unit",
                "task_type": "open_qa",
                "human_label": "A>B",
                "prompt": "Explain benchmark calibration.",
                "context": "",
                "answer_a": "Calibration estimates whether confidence tracks observed accuracy.",
                "answer_b": "OK.",
            }
        ]

        summary, rows = evaluate_samples(samples, JudgeConfig(backend="heuristic"))

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["parse_failure_count"], 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "unit-1")
        self.assertIn(rows[0]["pred_label"], {"A>B", "B>A", "Tie"})

    def test_base_score_rows_for_disk_preserves_backend_errors(self) -> None:
        rows = [
            {
                "id": "ok-1",
                "dataset": "unit",
                "task_type": "open_qa",
                "pred_label": "A>B",
                "raw_output": "[RESULT] A",
                "judge_backend": "m_prometheus",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
                "parse_status": "ok",
            }
        ]
        parse_failures = [
            {
                "id": "bad-1",
                "dataset": "unit",
                "task_type": "open_qa",
                "raw_output": "",
                "parse_status": "backend_error",
                "backend_status": {"backend": "m_prometheus"},
                "error": "model unavailable",
            }
        ]

        output_rows = base_score_rows_for_disk(rows, parse_failures)

        self.assertEqual(len(output_rows), 2)
        self.assertEqual(output_rows[1]["id"], "bad-1")
        self.assertEqual(output_rows[1]["judge_backend"], "m_prometheus")
        self.assertIsNone(output_rows[1]["pred_label"])
        self.assertEqual(output_rows[1]["parsed_scores"], {"score_a": None, "score_b": None})
        self.assertEqual(output_rows[1]["parse_status"], "backend_error")

    def test_split_base_score_rows_separates_valid_rows_from_failures(self) -> None:
        rows = [
            {
                "id": "ok-1",
                "dataset": "unit",
                "task_type": "open_qa",
                "pred_label": "A>B",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
                "parse_status": "ok",
            },
            {
                "id": "bad-1",
                "dataset": "unit",
                "task_type": "open_qa",
                "pred_label": None,
                "parsed_scores": {"score_a": None, "score_b": None},
                "parse_status": "failed",
            },
        ]

        valid_rows, failures = split_base_score_rows(rows)

        self.assertTrue(is_valid_pairwise_score_row(rows[0]))
        self.assertFalse(is_valid_pairwise_score_row(rows[1]))
        self.assertEqual([row["id"] for row in valid_rows], ["ok-1"])
        self.assertEqual([row["id"] for row in failures], ["bad-1"])

    def test_run_resumable_evaluation_skips_existing_ids_and_writes_partial(self) -> None:
        samples = [
            {
                "id": "done",
                "dataset": "unit",
                "task_type": "open_qa",
                "human_label": "A>B",
                "prompt": "Prompt",
                "answer_a": "Good answer.",
                "answer_b": "Bad answer.",
            },
            {
                "id": "new",
                "dataset": "unit",
                "task_type": "open_qa",
                "human_label": "A>B",
                "prompt": "Prompt",
                "answer_a": "Good answer.",
                "answer_b": "Bad answer.",
            },
        ]
        existing = [
            {
                "id": "done",
                "dataset": "unit",
                "task_type": "open_qa",
                "gold_label": "A>B",
                "pred_label": "A>B",
                "gold_score": 1.0,
                "pred_score": 1.0,
                "raw_output": "[RESULT] A",
                "judge_backend": "m_prometheus",
                "parsed_scores": {"score_a": 1.0, "score_b": 0.0},
                "parse_status": "ok",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "base_scores.partial.json").write_text(json.dumps(existing), encoding="utf-8")

            summary, rows = run_resumable_evaluation(
                samples,
                JudgeConfig(name="unit_resume", backend="m_prometheus", model_path="mock"),
                run_dir,
                checkpoint_interval=1,
                backend=MockPrometheusBackend("[RESULT] B"),
            )
            partial_exists = (run_dir / "base_scores.partial.json").exists()

        self.assertTrue(partial_exists)
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["checkpoint"]["skipped_existing_count"], 1)
        self.assertEqual(summary["checkpoint"]["evaluated_new_count"], 1)
        self.assertEqual(rows[0]["pred_label"], "A>B")
        self.assertEqual(rows[1]["pred_label"], "B>A")

    def test_pairwise_samples_filters_single_answer_factuality_rows(self) -> None:
        samples = [
            {"id": "pair", "human_label": "A>B", "answer_b": "B"},
            {"id": "single", "human_label": "supported", "answer_b": None},
            {"id": "bad-pair", "human_label": "B>A", "answer_b": ""},
        ]

        scoped = pairwise_samples(samples)

        self.assertEqual([row["id"] for row in scoped], ["pair"])

class Prometheus2IntegrationTest(unittest.TestCase):
    def test_build_prometheus_pairwise_prompt_contains_fixed_fields(self) -> None:
        sample = {
            "prompt": "Summarize the method.",
            "context": "Evidence context.",
            "answer_a": "A detailed and grounded answer.",
            "answer_b": "A vague answer.",
            "reference": "Reference answer.",
        }

        prompt = build_prometheus_pairwise_prompt(sample)

        self.assertIn("###Task Description:", prompt)
        self.assertIn("###The instruction to evaluate:", prompt)
        self.assertIn("###Response A:", prompt)
        self.assertIn("###Response B:", prompt)
        self.assertIn("###Reference Answer:", prompt)
        self.assertIn("[RESULT]", prompt)

    def test_build_prometheus_pairwise_prompt_defaults_to_m_prometheus_identity(self) -> None:
        prompt = build_prometheus_pairwise_prompt(
            {
                "prompt": "Summarize the method.",
                "answer_a": "A detailed answer.",
                "answer_b": "A short answer.",
            }
        )

        self.assertIn("You are M-Prometheus-3B", prompt)

    def test_extract_prometheus_pairwise_label_parses_result_marker(self) -> None:
        label, parsed = extract_prometheus_pairwise_label("Reasoning text\n[RESULT] B")

        self.assertEqual(label, "B>A")
        self.assertEqual(parsed["result_token"], "B")

    def test_extract_prometheus_pairwise_label_parses_clear_natural_language(self) -> None:
        cases = [
            ("Response A is better because it directly answers the instruction.", "A>B"),
            ("Response A provides a more accurate answer and aligns better with the instruction.", "A>B"),
            ("Response A is more comprehensive in its approach to the lesson plan.", "A>B"),
            ("Response A correctly identifies the efficient algorithm, while Response B incorrectly uses sorting.", "A>B"),
            ("Therefore, based on the rubric, Response A is better than Response B. [RESULT", "A>B"),
            ("Response B is the better response because it is more complete.", "B>A"),
            ("Response B is more factually correct and more helpful.", "B>A"),
            ("Response B provides a more detailed explanation of the evidence.", "B>A"),
            ("Response A fails to answer the question. In contrast, Response B directly answers the question.", "B>A"),
            ("Response A contains several errors and misconceptions.", "B>A"),
            ("Neither response is complete, but Response B at least attempts to prove the statement.", "B>A"),
            ("Therefore, based on the rubric, Response B is better than Response A. [RESULT", "B>A"),
            ("Both responses are equivalent and this should be treated as a tie.", "Tie"),
            ("Both responses are identical in content and structure, providing a complete and accurate description.", "Tie"),
            ("Both responses correctly update the original review with the release dates.", "Tie"),
        ]

        for text, expected in cases:
            with self.subTest(text=text):
                label, parsed = extract_prometheus_pairwise_label(text)

                self.assertEqual(label, expected)
                self.assertEqual(parsed["parse_warning"], "natural_language_result_marker_missing")

    def test_extract_prometheus_pairwise_label_rejects_ambiguous_natural_language(self) -> None:
        label, parsed = extract_prometheus_pairwise_label(
            "Response A is detailed, while Response B is concise. The choice depends on priorities."
        )

        self.assertIsNone(label)
        self.assertNotIn("parse_warning", parsed)

    def test_extract_prometheus_pairwise_label_rejects_conflicting_negative_language(self) -> None:
        label, parsed = extract_prometheus_pairwise_label(
            "Response A fails to show empathy. Response B also lacks empathy. The choice depends on priorities."
        )

        self.assertIsNone(label)
        self.assertNotIn("parse_warning", parsed)

    def test_extract_prometheus_pairwise_label_does_not_parse_unrelated_numbers(self) -> None:
        label, _parsed = extract_prometheus_pairwise_label(
            "Response A mentions 2024, while Response B mentions 2025. More context is needed."
        )

        self.assertIsNone(label)

    def test_mock_prometheus_backend_flows_through_evaluate_samples(self) -> None:
        samples = [
            {
                "id": "unit-prometheus",
                "dataset": "unit",
                "task_type": "open_qa",
                "human_label": "A>B",
                "prompt": "Explain calibration.",
                "context": "",
                "answer_a": "A complete calibrated answer.",
                "answer_b": "Short.",
                "reference": "",
            }
        ]
        config = JudgeConfig(name="prometheus_2", backend="prometheus2", model_path="mock")

        summary, rows = evaluate_samples(samples, config, backend=MockPrometheusBackend("[RESULT] A"))

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(rows[0]["pred_label"], "A>B")
        self.assertEqual(rows[0]["judge_backend"], "prometheus2")
        self.assertEqual(rows[0]["prompt_template"], "prometheus2_pairwise_v1")

    def test_prometheus_backend_reports_missing_dependencies(self) -> None:
        backend = Prometheus2Backend(model_path="mock-model")
        status = backend.status()

        self.assertIn("available", status)
        if not status["available"]:
            self.assertIn("missing_dependencies", status)

    def test_prometheus_unavailable_backend_falls_back_with_prometheus_metadata_offline(self) -> None:
        class UnavailableBackend(JudgeBackend):
            name = "prometheus2"

            def status(self):
                return {
                    "available": False,
                    "backend": "prometheus2",
                    "model_path": "offline-missing-model",
                    "missing_dependencies": [],
                }

            def score_pairwise(self, sample, config):
                raise AssertionError("fallback should avoid model inference")

            def score_direct(self, sample, config):
                raise AssertionError("fallback should avoid model inference")

        samples = [
            {
                "id": "unit-fallback",
                "dataset": "unit",
                "task_type": "open_qa",
                "human_label": "A>B",
                "prompt": "Explain calibration.",
                "context": "",
                "answer_a": "A complete calibrated answer.",
                "answer_b": "Short.",
            }
        ]
        config = JudgeConfig(
            name="prometheus_2",
            backend="prometheus2",
            model_path="offline-missing-model",
            allow_fallback=True,
        )

        _summary, rows = evaluate_samples(samples, config, backend=UnavailableBackend())

        self.assertEqual(rows[0]["judge_backend"], "heuristic_fallback")
        self.assertEqual(rows[0]["prompt_template"], "prometheus2_pairwise_v1")

    def test_prometheus_unavailable_backend_without_fallback_records_error(self) -> None:
        class UnavailableBackend(JudgeBackend):
            name = "prometheus2"

            def status(self):
                return {
                    "available": False,
                    "backend": "prometheus2",
                    "model_path": "offline-missing-model",
                    "missing_dependencies": [],
                }

            def score_pairwise(self, sample, config):
                raise AssertionError("unavailable backend should fail before inference")

            def score_direct(self, sample, config):
                raise AssertionError("unavailable backend should fail before inference")

        samples = [
            {
                "id": "unit-no-fallback",
                "dataset": "unit",
                "task_type": "open_qa",
                "human_label": "A>B",
                "prompt": "Explain calibration.",
                "context": "",
                "answer_a": "A complete calibrated answer.",
                "answer_b": "Short.",
            }
        ]
        config = JudgeConfig(
            name="prometheus_2",
            backend="prometheus2",
            model_path="offline-missing-model",
            allow_fallback=False,
        )

        summary, rows = evaluate_samples(samples, config, backend=UnavailableBackend())

        self.assertEqual(rows, [])
        self.assertEqual(summary["parse_failure_count"], 1)
        self.assertEqual(summary["parse_failures"][0]["parse_status"], "backend_error")

    def test_prometheus_inference_error_falls_back_when_allowed(self) -> None:
        class FailingBackend(JudgeBackend):
            name = "prometheus2"

            def __init__(self) -> None:
                self.pairwise_calls = 0

            def status(self):
                return {
                    "available": True,
                    "backend": "prometheus2",
                    "model_path": "offline-present-but-failing-model",
                    "missing_dependencies": [],
                }

            def score_pairwise(self, sample, config):
                self.pairwise_calls += 1
                raise RuntimeError("model load failed")

            def score_direct(self, sample, config):
                raise RuntimeError("model load failed")

        samples = []
        for sample_id in ("unit-inference-fallback-1", "unit-inference-fallback-2"):
            samples.append(
                {
                    "id": sample_id,
                    "dataset": "unit",
                    "task_type": "open_qa",
                    "human_label": "A>B",
                    "prompt": "Explain calibration.",
                    "context": "",
                    "answer_a": "A complete calibrated answer.",
                    "answer_b": "Short.",
                }
            )
        config = JudgeConfig(
            name="prometheus_2",
            backend="prometheus2",
            model_path="offline-present-but-failing-model",
            allow_fallback=True,
        )
        backend = FailingBackend()

        summary, rows = evaluate_samples(samples, config, backend=backend)

        self.assertEqual(summary["parse_failure_count"], 0)
        self.assertEqual(rows[0]["judge_backend"], "heuristic_fallback")
        self.assertEqual(rows[0]["parse_status"], "fallback_ok")
        self.assertEqual(rows[0]["fallback_from"], "prometheus2")
        self.assertEqual(rows[1]["judge_backend"], "heuristic_fallback")
        self.assertEqual(backend.pairwise_calls, 1)

    def test_build_prometheus_direct_prompt_contains_json_schema(self) -> None:
        sample = {
            "prompt": "Summarize the method.",
            "answer_a": "A detailed and grounded answer.",
            "reference": "Reference answer.",
        }

        prompt = build_prometheus_direct_prompt(sample)

        self.assertIn("###Response:", prompt)
        self.assertIn("###Output JSON Schema:", prompt)
        self.assertIn("overall_score", prompt)

    def test_extract_prometheus_direct_scores_parses_json(self) -> None:
        score, parsed = extract_prometheus_direct_scores(
            '{"relevance":5,"completeness":4,"factuality":5,'
            '"instruction_following":4,"clarity":5,"safety":5,"overall_score":4.7}'
        )

        self.assertEqual(score, 4.7)
        self.assertEqual(parsed["dimensions"]["relevance"], 5.0)

    def test_extract_prometheus_direct_scores_parses_result_marker_score(self) -> None:
        score, parsed = extract_prometheus_direct_scores(
            "Feedback: The answer is mostly complete and factual. [RESULT] 4"
        )

        self.assertEqual(score, 4.0)
        self.assertEqual(parsed["overall_score"], 4.0)
        self.assertEqual(parsed["parse_warning"], "result_marker_only_dimensions_missing")

    def test_mock_prometheus_backend_direct_mode(self) -> None:
        samples = [
            {
                "id": "unit-direct",
                "dataset": "unit",
                "task_type": "open_qa",
                "human_label": "A>B",
                "prompt": "Explain calibration.",
                "answer_a": "A complete calibrated answer.",
            }
        ]
        config = JudgeConfig(name="prometheus_2", backend="prometheus2", mode="direct", model_path="mock")

        summary, rows = evaluate_samples(
            samples,
            config,
            backend=MockPrometheusBackend('{"overall_score":4,"relevance":4,"completeness":4}'),
        )

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(rows[0]["pred_score"], 4.0)
        self.assertEqual(rows[0]["prompt_template"], "prometheus2_direct_v1")


class MPrometheus3BBaseTest(unittest.TestCase):
    def test_m_prometheus_backend_status_identifies_backbone(self) -> None:
        backend = MPrometheus3BBackend(model_path=DEFAULT_M_PROMETHEUS_MODEL_PATH)
        status = backend.status()

        self.assertEqual(status["backend"], "m_prometheus")
        self.assertEqual(status["model_path"], DEFAULT_M_PROMETHEUS_MODEL_PATH)
        self.assertEqual(status["backbone"], "M-Prometheus-3B")

    def test_m_prometheus_pairwise_output_metadata_uses_base_template(self) -> None:
        samples = [
            {
                "id": "unit-mprometheus",
                "dataset": "unit",
                "task_type": "open_qa",
                "human_label": "A>B",
                "prompt": "Explain calibration.",
                "context": "",
                "answer_a": "A complete calibrated answer.",
                "answer_b": "Short.",
                "reference": "",
            }
        ]
        config = JudgeConfig(
            name="m_prometheus_3b_base",
            version=DEFAULT_M_PROMETHEUS_MODEL_PATH,
            backend="m_prometheus",
            model_path=DEFAULT_M_PROMETHEUS_MODEL_PATH,
        )

        summary, rows = evaluate_samples(samples, config, backend=MockPrometheusBackend("[RESULT] A"))

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["judge"]["backend"], "m_prometheus")
        self.assertEqual(rows[0]["judge_backend"], "m_prometheus")
        self.assertEqual(rows[0]["judge_version"], DEFAULT_M_PROMETHEUS_MODEL_PATH)
        self.assertEqual(rows[0]["prompt_template"], "m_prometheus_pairwise_v1")
        self.assertEqual(rows[0]["pred_label"], "A>B")

    def test_m_prometheus_direct_mode_emits_initial_quality_scores(self) -> None:
        samples = [
            {
                "id": "unit-mprometheus-direct",
                "dataset": "unit",
                "task_type": "open_qa",
                "human_label": "A>B",
                "prompt": "Explain calibration.",
                "answer_a": "A complete calibrated answer.",
            }
        ]
        config = JudgeConfig(
            name="m_prometheus_3b_base",
            version=DEFAULT_M_PROMETHEUS_MODEL_PATH,
            backend="m_prometheus",
            mode="direct",
            model_path=DEFAULT_M_PROMETHEUS_MODEL_PATH,
        )

        summary, rows = evaluate_samples(
            samples,
            config,
            backend=MockPrometheusBackend('{"overall_score":4.2,"relevance":5,"completeness":4,"factuality":4}'),
        )

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["judge"]["backend"], "m_prometheus")
        self.assertEqual(rows[0]["judge_backend"], "m_prometheus")
        self.assertEqual(rows[0]["prompt_template"], "m_prometheus_direct_v1")
        self.assertEqual(rows[0]["parsed_scores"]["overall_score"], 4.2)
        self.assertEqual(rows[0]["parsed_scores"]["relevance"], 5.0)


if __name__ == "__main__":
    unittest.main()
