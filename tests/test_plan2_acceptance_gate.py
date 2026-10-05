from __future__ import annotations

import unittest

from scripts.run_plan2_acceptance_gate import threshold_report


class Plan2AcceptanceGateTests(unittest.TestCase):
    def inputs(self):
        generation = {
            "failure_rate": 0.0,
            "mean_rtf": 0.3,
            "peak_cuda_gib": 6.5,
        }
        control_raw = {"cer": 0.06, "speaker_cosine": 0.78}
        candidate_raw = {"cer": 0.055, "speaker_cosine": 0.80}
        trimmed = {
            "cer": 0.054,
            "speaker_cosine": 0.79,
            "quality_pass_rate": 0.83,
            "flat_top_run_count": 0,
            "maximum_true_peak_dbtp": -0.2,
        }
        manual = {"preference_counts": {"trained": 5, "control": 3}}
        return generation, control_raw, candidate_raw, trimmed, manual

    def test_all_three_gates_accept_safe_candidate(self) -> None:
        generation, control_raw, candidate_raw, trimmed, manual = self.inputs()
        report = threshold_report(
            generation,
            generation,
            control_raw,
            candidate_raw,
            trimmed,
            trimmed,
            manual,
            same_postprocess_config=True,
            provenance_matches=True,
        )
        self.assertTrue(report["accepted"])
        self.assertTrue(all(gate["passed"] for gate in report["gates"].values()))

    def test_one_failed_threshold_rejects_candidate(self) -> None:
        generation, control_raw, candidate_raw, trimmed, manual = self.inputs()
        failing_generation = {**generation, "mean_rtf": 1.01}
        report = threshold_report(
            generation,
            failing_generation,
            control_raw,
            candidate_raw,
            trimmed,
            trimmed,
            manual,
            same_postprocess_config=True,
            provenance_matches=True,
        )
        self.assertFalse(report["accepted"])
        self.assertFalse(report["gates"]["performance_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
