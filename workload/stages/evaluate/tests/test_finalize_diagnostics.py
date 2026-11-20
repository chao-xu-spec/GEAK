"""What an A/B can and cannot show is reported beside its verdict."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
FINALIZE = STAGE / "finalize.py"


def decode_line(running, accept, throughput):
    return ("[2026-01-01 00:00:00 TP0 EP0] Decode batch, #running-req: %d, accept len: %.2f, "
            "gen throughput (token/s): %.2f, #queue-req: 0\n" % (running, accept, throughput))


class DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / "applyback" / "out"
        (self.out / "e2e").mkdir(parents=True)
        reference = self.root / "baseline" / "out"
        reference.mkdir(parents=True)
        (reference / "baseline.json").write_text(json.dumps({"measured": True, "throughput": 100.0}))

    def leg(self, name, throughput, step_throughput=None):
        """A leg whose receipt sits beside a serving log; `step_throughput` fills the log."""
        run = self.root / "runs" / name
        run.mkdir(parents=True, exist_ok=True)
        receipt = run / (name + ".json")
        receipt.write_text("{}")
        if step_throughput:
            (run / "server.log").write_text(decode_line(4, 4.0, step_throughput) * 40)
        return {"tag": name, "throughput": throughput, "provenance": str(receipt)}

    def arm(self, legs):
        values = [leg["throughput"] for leg in legs]
        return {"throughput": sum(values) / len(values), "metric_basis": "tok/s",
                "repeats": len(legs), "legs": legs}

    def finalize(self, candidate, stock, params, integration=None):
        if integration is not None:                       # the artifacts applyback hands over
            (self.out / "applied.json").write_text(json.dumps(integration))
            (self.out / "overlay").mkdir(exist_ok=True)
        summary = dict(candidate, stock_control=dict(stock, admissible=True))
        (self.out / "e2e" / "bench_summary.json").write_text(json.dumps(summary))
        (self.out / "params.json").write_text(json.dumps(params))
        env = dict(os.environ, OUT=str(self.out), RUN_ROOT=str(self.root),
                   PARAMS=str(self.out / "params.json"), STAGE="applyback",
                   STAGE_SKILL="overlay-ab")
        code = ("import sys, json; sys.path.insert(0, %r); import finalize; "
                "sys.exit(finalize.main(json.loads(sys.argv[1]) if sys.argv[1] != 'None' else None))"
                % str(STAGE))
        subprocess.run([sys.executable, "-c", code, json.dumps(integration)], env=env,
                       capture_output=True, text=True)
        return (json.loads((self.out / "result.json").read_text()),
                json.loads((self.out / "ab.json").read_text()))

    def test_band_far_wider_than_the_measured_spread_is_flagged(self):
        stock = self.arm([self.leg("s1", 100.0), self.leg("s2", 100.5)])       # 0.5% apart
        cand = self.arm([self.leg("c1", 97.0), self.leg("c2", 97.4)])
        res, ab = self.finalize(cand, stock, {"noise_band": 1.045})
        self.assertEqual(ab["verdict"], "no_change")                 # policy is unchanged
        self.assertAlmostEqual(ab["measured_noise_pct"], 0.5, places=1)
        self.assertIn("4.50%", ab["band_warning"])
        self.assertIn("still reported as no_change", ab["band_warning"])
        self.assertIn("configured noise band", res["notes"])

    def test_a_band_in_line_with_the_measured_spread_is_not_flagged(self):
        stock = self.arm([self.leg("s1", 100.0), self.leg("s2", 100.5)])
        cand = self.arm([self.leg("c1", 97.0), self.leg("c2", 97.4)])
        _, ab = self.finalize(cand, stock, {"noise_band": 1.01})
        self.assertEqual(ab["verdict"], "regression")
        self.assertIsNone(ab["band_warning"])

    def test_single_legs_have_no_measured_spread(self):
        _, ab = self.finalize(self.arm([self.leg("c", 99.0)]), self.arm([self.leg("s", 100.0)]), {})
        self.assertIsNone(ab["measured_noise_pct"])
        self.assertIsNone(ab["band_warning"])

    def test_decode_step_table_and_optional_regression_gate(self):
        stock = self.arm([self.leg("s1", 100.0, 600.0)])
        cand = self.arm([self.leg("c1", 99.5, 500.0)])               # 20% slower steps, 0.5% tps
        res, ab = self.finalize(cand, stock, {"noise_band": 1.045})
        self.assertEqual(ab["verdict"], "no_change")
        self.assertAlmostEqual(ab["decode_step_by_batch"][0]["ratio"], 1.2)
        self.assertIn("| Decode batch |", (self.out / "REPORT.md").read_text())
        res, ab = self.finalize(cand, stock, {"noise_band": 1.045,
                                              "max_decode_step_regression_pct": 5})
        self.assertEqual(ab["verdict"], "regression")
        self.assertIn("Decode steps got slower", res["notes"])
        self.assertEqual(res["status"], "ok")

    def test_amdahl_ceiling_of_installed_kernels_is_reported_against_the_band(self):
        roofline = self.root / "roofline" / "out"
        roofline.mkdir(parents=True)
        (roofline / "roofline.json").write_text(json.dumps({"kernels": [
            {"kernel": "taskA", "id": "taskA", "task": "taskA", "pct_gpu_time": 20.0, "in_selection": True}]}))
        integration = {"applied": [{"kernel": "taskA", "verified_speedup": 1.10, "target": "m:f"}]}
        stock = self.arm([self.leg("s", 100.0)])
        cand = self.arm([self.leg("c", 99.0)])
        res, ab = self.finalize(cand, stock, {"noise_band": 1.045}, integration)
        self.assertAlmostEqual(ab["amdahl_ceiling_pct"], 20.0 * (1 - 1 / 1.10), places=3)
        self.assertEqual(ab["op_level"][0]["share_pct"], 20.0)
        self.assertIn("Amdahl ceiling", res["notes"])
        self.assertIn("cannot confirm", res["notes"])

    def test_missing_share_evidence_leaves_the_ceiling_unreported(self):
        integration = {"applied": [{"kernel": "taskA", "verified_speedup": 1.10, "target": "m:f"}]}
        _, ab = self.finalize(self.arm([self.leg("c", 99.0)]), self.arm([self.leg("s", 100.0)]), {},
                              integration)
        self.assertIsNone(ab["amdahl_ceiling_pct"])


if __name__ == "__main__":
    unittest.main()
