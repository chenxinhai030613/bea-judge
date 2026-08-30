import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_features import (  # noqa: E402
    build_evidence_profile,
    evidence_feature_dict,
    factuality_decision_signal,
)


class EvidenceFeaturesTest(unittest.TestCase):
    def test_supported_answer_has_higher_context_support_than_unsupported_answer(self) -> None:
        supported = {
            "id": "fact-1",
            "dataset": "unit",
            "task_type": "factuality_rag",
            "human_label": "supported",
            "context": "Marie Curie won the Nobel Prize in Physics in 1903.",
            "reference": "Marie Curie won the 1903 Nobel Prize in Physics.",
            "answer_a": "Marie Curie won the Nobel Prize in Physics in 1903.",
            "answer_b": None,
            "metadata": {"factuality_task_form": "single_answer"},
        }
        unsupported = {
            **supported,
            "id": "fact-2",
            "human_label": "unsupported",
            "answer_a": "Marie Curie won the Nobel Prize in Chemistry in 1999.",
        }

        supported_profile = build_evidence_profile(supported)
        unsupported_profile = build_evidence_profile(unsupported)

        self.assertGreater(
            supported_profile["evidence"]["context_support_a"],
            unsupported_profile["evidence"]["context_support_a"],
        )
        self.assertLess(supported_profile["evidence"]["evidence_risk"], unsupported_profile["evidence"]["evidence_risk"])

    def test_numeric_evidence_gap_is_flagged_when_answer_number_is_absent_from_evidence(self) -> None:
        sample = {
            "id": "fact-num",
            "dataset": "unit",
            "task_type": "factuality_rag",
            "human_label": "unsupported",
            "context": "The trial enrolled 120 participants and ended in 2021.",
            "reference": "The study involved 120 participants.",
            "answer_a": "The trial enrolled 450 participants.",
            "answer_b": None,
            "metadata": {"factuality_task_form": "single_answer"},
        }

        profile = build_evidence_profile(sample)

        self.assertGreater(profile["evidence"]["numeric_evidence_gap_a"], 0.0)
        self.assertIn("numeric_evidence_gap_a", profile["evidence"]["reasons"])

    def test_entity_and_date_gaps_increase_local_hallucination_risk(self) -> None:
        supported = {
            "id": "fact-entity-ok",
            "dataset": "ragtruth",
            "task_type": "factuality_rag",
            "human_label": "supported",
            "context": "Anne Frank and Margot Frank likely died before March 1945.",
            "reference": "",
            "answer_a": "Anne Frank and Margot Frank likely died before March 1945.",
            "answer_b": None,
            "metadata": {"factuality_task_form": "single_answer"},
        }
        unsupported = {
            **supported,
            "id": "fact-entity-bad",
            "human_label": "unsupported",
            "answer_a": "Anne Frank and Margot Frank met Dr. Evelyn Carter in 1999.",
        }

        ok_features = evidence_feature_dict(supported)
        bad_features = evidence_feature_dict(unsupported)

        self.assertGreater(bad_features["evidence_entity_gap_a"], ok_features["evidence_entity_gap_a"])
        self.assertGreater(bad_features["evidence_date_gap_a"], ok_features["evidence_date_gap_a"])
        self.assertGreater(
            bad_features["evidence_local_hallucination_risk_a"],
            ok_features["evidence_local_hallucination_risk_a"],
        )

    def test_low_support_anchor_sentence_gap_flags_local_hallucination(self) -> None:
        sample = {
            "id": "fact-anchor-gap",
            "dataset": "ragtruth",
            "task_type": "factuality_rag",
            "human_label": "unsupported",
            "context": "The report says the mission launched from Florida in 2021.",
            "reference": "",
            "answer_a": "The mission launched from Florida in 2021. It later landed in Oslo with Dr. Vera Lang.",
            "answer_b": None,
            "metadata": {"factuality_task_form": "single_answer"},
        }

        profile = build_evidence_profile(sample)
        features = evidence_feature_dict(sample)

        self.assertGreater(features["evidence_low_support_anchor_sentence_ratio_a"], 0.0)
        self.assertGreater(features["evidence_max_low_support_anchor_gap_a"], 0.0)
        self.assertGreater(features["evidence_anchored_hallucination_severity_a"], 0.0)
        self.assertIn("low_support_anchor_sentence_ratio_a", profile["evidence"]["reasons"])
        self.assertIn("max_low_support_anchor_gap_a", profile["evidence"]["reasons"])

    def test_negation_and_comparative_mismatch_are_numeric_risk_features(self) -> None:
        sample = {
            "id": "fact-neg-comp",
            "dataset": "ragtruth",
            "task_type": "factuality_rag",
            "human_label": "unsupported",
            "context": "The medication reduced fever and had lower risk than placebo.",
            "reference": "The medication reduced fever with lower risk.",
            "answer_a": "The medication did not reduce fever and had higher risk than placebo.",
            "answer_b": None,
            "metadata": {"factuality_task_form": "single_answer"},
        }

        profile = build_evidence_profile(sample)
        features = evidence_feature_dict(sample)

        self.assertGreater(features["evidence_negation_mismatch_a"], 0.0)
        self.assertGreater(features["evidence_comparative_mismatch_a"], 0.0)
        self.assertIn("negation_mismatch_a", profile["evidence"]["reasons"])
        self.assertIn("comparative_mismatch_a", profile["evidence"]["reasons"])

    def test_entity_alias_gap_uses_acronym_normalization(self) -> None:
        supported = {
            "id": "fact-alias-ok",
            "dataset": "ragtruth",
            "task_type": "factuality_rag",
            "human_label": "supported",
            "context": "The National Aeronautics Space Administration launched Artemis.",
            "reference": "",
            "answer_a": "NASA launched Artemis.",
            "answer_b": None,
            "metadata": {"factuality_task_form": "single_answer"},
        }
        unsupported = {
            **supported,
            "id": "fact-alias-bad",
            "answer_a": "ESA launched Artemis.",
        }

        ok_features = evidence_feature_dict(supported)
        bad_features = evidence_feature_dict(unsupported)

        self.assertLess(ok_features["evidence_entity_alias_gap_a"], bad_features["evidence_entity_alias_gap_a"])

    def test_pairwise_support_delta_is_positive_when_a_is_better_supported(self) -> None:
        sample = {
            "id": "fact-pair",
            "dataset": "unit",
            "task_type": "factuality_rag",
            "human_label": "A>B",
            "context": "The capital of France is Paris. It is known for the Louvre.",
            "reference": "Paris is the capital of France.",
            "answer_a": "The capital of France is Paris.",
            "answer_b": "The capital of France is Lyon.",
            "metadata": {"factuality_task_form": "pairwise"},
        }

        profile = build_evidence_profile(sample)
        features = evidence_feature_dict(sample)

        self.assertGreater(profile["evidence"]["support_delta_a_minus_b"], 0.0)
        self.assertGreater(features["evidence_support_delta_a_minus_b"], 0.0)
        self.assertEqual(factuality_decision_signal(sample), "pairwise_factuality")

    def test_single_answer_missing_b_side_uses_null_profile_and_zero_numeric_features(self) -> None:
        sample = {
            "id": "fact-single",
            "dataset": "unit",
            "task_type": "factuality_rag",
            "human_label": "supported",
            "context": "Water boils at 100 degrees Celsius at sea level.",
            "reference": "At sea level, water boils at 100 degrees Celsius.",
            "answer_a": "Water boils at 100 degrees Celsius at sea level.",
            "answer_b": None,
            "metadata": {"factuality_task_form": "single_answer"},
        }

        profile = build_evidence_profile(sample)
        features = evidence_feature_dict(sample)

        self.assertIsNone(profile["evidence"]["context_support_b"])
        self.assertIsNone(profile["evidence"]["support_delta_a_minus_b"])
        self.assertEqual(features["evidence_context_support_b"], 0.0)
        self.assertEqual(features["evidence_support_delta_a_minus_b"], 0.0)

    def test_evidence_feature_dict_outputs_numeric_values_only(self) -> None:
        sample = {
            "id": "fact-features",
            "dataset": "unit",
            "task_type": "factuality_rag",
            "human_label": "supported",
            "context": "The device has a battery life of 10 hours.",
            "reference": "Battery life is 10 hours.",
            "answer_a": "The device has a 10 hour battery life.",
            "answer_b": None,
            "metadata": {"factuality_task_form": "single_answer"},
        }

        features = evidence_feature_dict(sample)

        self.assertTrue(features)
        for value in features.values():
            self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
