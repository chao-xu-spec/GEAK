"""Decode step time by batch size, read from an engine's serving log."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decode_step  # noqa: E402


def line(running, accept, throughput, rank=0):
    return ("[2026-01-01 00:00:00 TP%d EP%d] Decode batch, #running-req: %d, #full token: 10, "
            "accept len: %.2f, accept rate: 0.44, cuda graph: True, gen throughput (token/s): "
            "%.2f, #queue-req: 0\n" % (rank, rank, running, accept, throughput))


class DecodeStepTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def log(self, name, lines):
        path = self.dir / name
        path.write_text("".join(lines))
        return str(path)

    def test_step_time_follows_from_batch_accept_length_and_throughput(self):
        path = self.log("a.log", [line(3, 4.0, 500.0)])
        self.assertEqual(decode_step.step_times(path), {3: [24.0]})      # 3 * 4.0 / 500 s

    def test_other_ranks_idle_lines_and_other_engines_are_ignored(self):
        path = self.log("a.log", [line(2, 4.0, 400.0, rank=1), line(2, 4.0, 1.0),
                                  "INFO unrelated line\n", "Prefill batch, #new-seq: 1\n",
                                  line(2, 4.0, 400.0)])
        self.assertEqual(decode_step.step_times(path), {2: [20.0]})

    def test_other_sglang_dialects_are_read(self):
        path = self.log("d.log", [
            # no speculative decoding: one token per request per step
            "[2026-01-01 00:00:00 TP0] Decode batch, #running-req: 4, #token: 900, token usage: 0.01, "
            "cuda graph: True, gen throughput (token/s): 200.00, #queue-req: 0\n",
            # attention-context-parallel prefix, hybrid-pool token fields, dotted separator
            "[2026-01-01 00:00:00 ATTN_CP0 TP0] Decode batch. #running-req: 2, #full token: 10, "
            "full token usage: 0.00, #swa token: 5, swa token usage: 0.00, accept len: 3.00, "
            "gen throughput (token/s): 300.00, #queue-req: 0\n",
            # a numbered batch
            "[2026-01-01 00:00:00 TP0] Decode batch [12], #running-req: 1, #token: 5, accept len: 2.00, "
            "gen throughput (token/s): 100.00, #queue-req: 0\n"])
        steps = decode_step.step_times(path)
        self.assertEqual({bs: [round(v, 3) for v in vals] for bs, vals in steps.items()},
                         {4: [20.0], 2: [20.0], 1: [20.0]})

    def test_candidate_slower_at_one_batch_size_is_reported_with_its_ratio(self):
        stock = self.log("s.log", [line(1, 4.0, 200.0)] * 40 + [line(4, 4.0, 600.0)] * 40)
        cand = self.log("c.log", [line(1, 4.0, 200.0)] * 40 + [line(4, 4.0, 500.0)] * 40)
        rows = decode_step.compare([stock], [cand])
        self.assertEqual([r["batch"] for r in rows], [1, 4])
        self.assertAlmostEqual(rows[0]["ratio"], 1.0)
        self.assertAlmostEqual(rows[1]["ratio"], 1.2)
        slower = decode_step.regressions(rows, 5)
        self.assertEqual([r["batch"] for r in slower], [4])
        self.assertEqual(decode_step.regressions(rows, 25), [])

    def test_batch_sizes_seen_too_few_times_or_on_one_side_are_left_out(self):
        stock = self.log("s.log", [line(1, 4.0, 200.0)] * 40 + [line(2, 4.0, 300.0)] * 5)
        cand = self.log("c.log", [line(1, 4.0, 200.0)] * 40 + [line(2, 4.0, 300.0)] * 40
                        + [line(7, 4.0, 700.0)] * 40)
        self.assertEqual([r["batch"] for r in decode_step.compare([stock], [cand])], [1])

    def test_legs_are_pooled_per_batch_size(self):
        a = self.log("a.log", [line(1, 4.0, 200.0)] * 20)
        b = self.log("b.log", [line(1, 4.0, 200.0)] * 20)
        rows = decode_step.compare([a, b], [a, b])
        self.assertEqual(rows[0]["n_stock"], 40)


if __name__ == "__main__":
    unittest.main()
