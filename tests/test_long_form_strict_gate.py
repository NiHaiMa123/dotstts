from __future__ import annotations

import unittest

import yaml
from pydantic import ValidationError

from dots_tts_lab.long_form_strict_gate import (
    ALL_GATES,
    DEFAULT_STRICT_GATE_CONFIG_PATH,
    ApprovalBinding,
    BatchBudget,
    CandidateBinding,
    GateVerdict,
    StrictGateConfig,
    SourceSpan,
    apply_batch_budget,
    evaluate_candidate,
    load_strict_gate_config,
    verify_approval_binding,
)


def _verdict(gate: str, status: str = "pass", calibrated: bool = True) -> GateVerdict:
    return GateVerdict(
        schema_version=1,
        gate=gate,
        status=status,
        reasons=[] if status == "pass" else [f"{gate} {status}"],
        evidence={"artifact_sha256": "a" * 64},
        evaluator="fixture-evaluator",
        calibrated=calibrated,
        recorded_at="2026-09-13T00:00:00Z",
    )


def _all_pass(calibrated: bool = True) -> dict[str, GateVerdict]:
    return {gate: _verdict(gate, calibrated=calibrated) for gate in ALL_GATES}


def _production_config() -> StrictGateConfig:
    payload = yaml.safe_load(DEFAULT_STRICT_GATE_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["mode"] = "production"
    for name in ("identity", "normal", "noise", "quality", "alignment"):
        payload["thresholds"][name] = {"value": 0.5, "calibrated": True}
    payload["thresholds"]["event"] = {
        key: {"value": 0.5, "calibrated": True}
        for key in payload["thresholds"]["event"]
    }
    return StrictGateConfig.model_validate(payload, strict=True)


def _binding(audio: str = "a" * 64, gate_config: str = "c" * 64) -> CandidateBinding:
    return CandidateBinding(
        schema_version=1,
        candidate_id="b" * 64,
        source_sha256="d" * 64,
        source_sample_rate_hz=48000,
        source_spans=[SourceSpan(source_start_frame=0, source_end_frame=96000)],
        final_audio_sha256=audio,
        enhancement_route="raw",
        gate_config_sha256=gate_config,
    )


class StrictGateContractTests(unittest.TestCase):
    def test_default_config_is_calibration_mode_and_hash_stable(self) -> None:
        config = load_strict_gate_config()
        self.assertEqual(config.mode, "calibration")
        self.assertEqual(config.certification.status, "none")
        self.assertEqual(config.batch_budget.total_candidates, 24)
        self.assertEqual(config.batch_budget.per_source, 6)
        self.assertEqual(load_strict_gate_config().config_sha256(), config.config_sha256())

    def test_required_gates_cannot_be_dropped(self) -> None:
        payload = yaml.safe_load(DEFAULT_STRICT_GATE_CONFIG_PATH.read_text(encoding="utf-8"))
        payload["required_gates"] = list(ALL_GATES[:-1])
        with self.assertRaises(ValidationError):
            StrictGateConfig.model_validate(payload, strict=True)

    def test_production_mode_rejects_uncalibrated_thresholds(self) -> None:
        payload = yaml.safe_load(DEFAULT_STRICT_GATE_CONFIG_PATH.read_text(encoding="utf-8"))
        payload["mode"] = "production"
        with self.assertRaisesRegex(ValidationError, "calibrated"):
            StrictGateConfig.model_validate(payload, strict=True)

    def test_all_pass_in_calibration_mode_yields_calibration_sample(self) -> None:
        outcome = evaluate_candidate(_all_pass(), load_strict_gate_config())
        self.assertEqual(outcome.disposition, "calibration_sample")

    def test_certified_production_yields_auto_verified(self) -> None:
        payload = yaml.safe_load(DEFAULT_STRICT_GATE_CONFIG_PATH.read_text(encoding="utf-8"))
        payload["mode"] = "production"
        for name in ("identity", "normal", "noise", "quality", "alignment"):
            payload["thresholds"][name] = {"value": 0.5, "calibrated": True}
        payload["thresholds"]["event"] = {
            key: {"value": 0.5, "calibrated": True}
            for key in payload["thresholds"]["event"]
        }
        payload["certification"] = {
            "status": "certified",
            "report_sha256": "e" * 64,
            "report_path": "data/reports/long_form/strict_gate/cert.json",
        }
        config = StrictGateConfig.model_validate(payload, strict=True)
        outcome = evaluate_candidate(_all_pass(), config)
        self.assertEqual(outcome.disposition, "auto_verified")

    def test_pending_certification_caps_at_strict_candidate(self) -> None:
        config = _production_config()
        outcome = evaluate_candidate(_all_pass(), config)
        self.assertEqual(outcome.disposition, "strict_candidate")

    def test_single_fail_rejects_despite_other_passes(self) -> None:
        verdicts = _all_pass()
        verdicts["G5"] = _verdict("G5", status="fail")
        outcome = evaluate_candidate(verdicts, _production_config())
        self.assertEqual(outcome.disposition, "reject")
        self.assertEqual(outcome.failed_gates, ["G5"])

    def test_missing_gate_quarantines(self) -> None:
        verdicts = _all_pass()
        del verdicts["G2"]
        outcome = evaluate_candidate(verdicts, _production_config())
        self.assertEqual(outcome.disposition, "quarantine")
        self.assertIn("G2", outcome.unknown_gates)

    def test_unknown_status_quarantines(self) -> None:
        verdicts = _all_pass()
        verdicts["G6"] = _verdict("G6", status="unknown")
        outcome = evaluate_candidate(verdicts, _production_config())
        self.assertEqual(outcome.disposition, "quarantine")

    def test_uncalibrated_pass_is_downgraded_in_production(self) -> None:
        verdicts = _all_pass()
        verdicts["G3"] = _verdict("G3", calibrated=False)
        outcome = evaluate_candidate(verdicts, _production_config())
        self.assertEqual(outcome.disposition, "quarantine")
        self.assertIn("G3", outcome.unknown_gates)

    def test_g9_failure_quarantines_not_rejects(self) -> None:
        verdicts = _all_pass()
        verdicts["G9"] = _verdict("G9", status="fail")
        outcome = evaluate_candidate(verdicts, _production_config())
        self.assertEqual(outcome.disposition, "quarantine")


class BatchBudgetTests(unittest.TestCase):
    def test_budget_enforces_total_and_per_source_caps(self) -> None:
        budget = BatchBudget(total_candidates=4, per_source=2, calibration_batch_max=6)
        candidates = [f"c{i}" for i in range(8)]
        sources = {f"c{i}": f"s{i % 3}" for i in range(8)}
        result = apply_batch_budget(candidates, sources, budget)
        kept = [c for c, d in result.items() if d == "strict_candidate"]
        self.assertEqual(len(kept), 4)
        for source in ("s0", "s1", "s2"):
            self.assertLessEqual(
                sum(1 for c in kept if sources[c] == source), budget.per_source
            )
        self.assertTrue(all(d == "reserve_budget" for c, d in result.items() if c not in kept))

    def test_per_source_cap_forces_reserve_before_total(self) -> None:
        budget = BatchBudget(total_candidates=24, per_source=2, calibration_batch_max=6)
        candidates = [f"a{i}" for i in range(5)] + ["b0"]
        sources = {**{f"a{i}": "s1" for i in range(5)}, "b0": "s2"}
        result = apply_batch_budget(candidates, sources, budget)
        self.assertEqual(result["a2"], "reserve_budget")
        self.assertEqual(result["b0"], "strict_candidate")


class ApprovalBindingTests(unittest.TestCase):
    def test_identical_binding_reuses_approval(self) -> None:
        candidate = _binding()
        approval = ApprovalBinding(
            schema_version=1,
            decision="human_confirmed",
            batch_or_report_sha256="f" * 64,
            recorded_at="2026-09-13T00:00:00Z",
            bound=candidate,
        )
        self.assertEqual(verify_approval_binding(approval, candidate), [])

    def test_changed_audio_or_config_blocks_migration(self) -> None:
        approval = ApprovalBinding(
            schema_version=1,
            decision="human_confirmed",
            batch_or_report_sha256="f" * 64,
            recorded_at="2026-09-13T00:00:00Z",
            bound=_binding(),
        )
        changed_audio = _binding(audio="0" * 64)
        mismatches = verify_approval_binding(approval, changed_audio)
        self.assertIn("final_audio_sha256", mismatches)
        changed_config = _binding(gate_config="9" * 64)
        self.assertIn(
            "gate_config_sha256", verify_approval_binding(approval, changed_config)
        )

    def test_auto_verified_requires_matching_certification(self) -> None:
        approval = ApprovalBinding(
            schema_version=1,
            decision="auto_verified",
            batch_or_report_sha256="f" * 64,
            recorded_at="2026-09-13T00:00:00Z",
            bound=_binding().model_copy(
                update={"calibration_report_sha256": "f" * 64}
            ),
        )
        candidate = _binding()
        self.assertIn(
            "calibration_report_sha256",
            verify_approval_binding(approval, candidate),
        )


if __name__ == "__main__":
    unittest.main()
