"""A GPU another tenant is using is not handed a task and is not measured on."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

WORKLOAD = Path(__file__).resolve().parents[2]
SHARED = WORKLOAD / "stages" / "_shared"
CAMPAIGN = WORKLOAD / "engine" / "campaign"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(CAMPAIGN))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu_idle = load(SHARED / "gpu_idle.py")
scheduler = load(CAMPAIGN / "scheduler.py")


class FakeDrm:
    """A /sys/class/drm and /sys/class/kfd look-alike. Render minors are deliberately not a
    fixed stride apart: GPU N is the Nth GPU in topology order, wherever its node is."""

    MINORS = (129, 137, 151, 160)

    def __init__(self, root, minors=MINORS):
        self.base, self.minors = Path(root), list(minors)
        self.root = self.base / "drm"
        self.topology = self.base / "topology"
        cpu = self.topology / "0"
        cpu.mkdir(parents=True)
        (cpu / "properties").write_text("cpu_cores_count 8\ndrm_render_minor 0\n")
        for index, minor in enumerate(self.minors, 1):
            node = self.topology / str(index)
            node.mkdir()
            (node / "properties").write_text("cpu_cores_count 0\ndrm_render_minor %d\n" % minor)
        self.env = {"GEAK_DRM_ROOT": str(self.root), "GEAK_KFD_TOPOLOGY": str(self.topology)}

    def set(self, gpu, busy=0, vram_mb=283, busy_counter=True, vram_counter=True):
        device = self.root / ("renderD%d" % self.minors[gpu]) / "device"
        device.mkdir(parents=True, exist_ok=True)
        if busy_counter:
            (device / "gpu_busy_percent").write_text("%d\n" % busy)
        if vram_counter:
            (device / "mem_info_vram_used").write_text("%d\n" % (vram_mb * (1 << 20)))


class GpuIdleTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.drm = FakeDrm(tmp.name)
        patcher = patch.dict(os.environ, self.drm.env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_idle_gpus_are_not_flagged(self):
        self.drm.set(0), self.drm.set(1)
        self.assertEqual(gpu_idle.foreign_use([0, 1]), [])

    def test_a_busy_gpu_and_an_idle_server_holding_the_memory_are_both_flagged(self):
        self.drm.set(0, busy=100, vram_mb=1837)       # running work
        self.drm.set(1, busy=0, vram_mb=286 * 1024)   # a loaded model with no traffic
        self.drm.set(2)
        self.assertEqual([r["gpu"] for r in gpu_idle.foreign_use([0, 1, 2])], [0, 1])

    def test_the_memory_is_still_seen_when_the_busy_counter_cannot_be_read(self):
        self.drm.set(0, busy_counter=False, vram_mb=286 * 1024)
        self.drm.set(1, busy_counter=False, vram_mb=283)
        rows = gpu_idle.foreign_use([0, 1])
        self.assertEqual([r["gpu"] for r in rows], [0])
        self.assertIn("busy unreadable", gpu_idle.describe(rows))

    def test_not_knowing_is_not_evidence_of_a_tenant(self):
        self.drm.set(0, busy_counter=False, vram_counter=False)
        self.assertEqual(gpu_idle.foreign_use([0, 3]), [])        # no counters; no render node

    def test_thresholds_and_the_switch_come_from_the_environment(self):
        self.drm.set(0, busy=40, vram_mb=500)
        with patch.dict(os.environ, {"GEAK_GPU_MAX_BUSY_PCT": "50"}):
            self.assertEqual(gpu_idle.foreign_use([0]), [])
        with patch.dict(os.environ, {"GEAK_GPU_REQUIRE_IDLE": "0"}):
            self.assertEqual(gpu_idle.foreign_use([0]), [])
        self.assertEqual(len(gpu_idle.foreign_use([0])), 1)

    def test_gpu_index_follows_topology_order_not_a_render_minor_stride(self):
        self.drm.set(0), self.drm.set(1, busy=100), self.drm.set(2), self.drm.set(3)
        self.assertEqual(gpu_idle.render_minors(), [129, 137, 151, 160])
        self.assertEqual([r["gpu"] for r in gpu_idle.foreign_use([0, 1, 2, 3])], [1])

    def test_an_explicit_container_mapping_overrides_the_topology(self):
        self.drm.set(0), self.drm.set(1, busy=100)
        rows = gpu_idle.foreign_use([0, 1], render_paths=["/dev/dri/renderD129", "/dev/dri/renderD137"])
        self.assertEqual([r["gpu"] for r in rows], [1])

    def test_wait_returns_once_the_gpu_clears_and_gives_the_reason_when_it_does_not(self):
        self.drm.set(0, busy=100)
        polls = []

        def clear_after_two(_):
            polls.append(1)
            if len(polls) == 2:
                self.drm.set(0)

        self.assertEqual(gpu_idle.wait_for_idle([0], 900, sleep=clear_after_two), [])
        self.assertEqual(len(polls), 2)
        self.drm.set(0, busy=100)
        rows = gpu_idle.wait_for_idle([0], 60, interval=30, sleep=lambda _: None)
        self.assertEqual([r["gpu"] for r in rows], [0])

    def run_cli(self, *argv):
        return subprocess.run([sys.executable, str(SHARED / "gpu_idle.py"), *argv],
                              env=dict(os.environ, **self.drm.env), capture_output=True, text=True)

    def test_command_line_exit_code_and_reason(self):
        self.drm.set(0), self.drm.set(1, busy=99, vram_mb=5000)
        idle, busy = self.run_cli("0"), self.run_cli("--wait", "0", "0,1")
        self.assertEqual((idle.returncode, idle.stdout), (0, ""))
        self.assertEqual(busy.returncode, 1)
        self.assertIn("GPU 1 busy 99%", busy.stdout)

    def test_gpu_lock_asks_the_same_implementation_before_running_a_command(self):
        self.drm.set(0, busy=100, vram_mb=300000)
        res = subprocess.run(["bash", str(SHARED / "gpu_lock.sh"), "0", "echo RAN"],
                             env=dict(os.environ, **self.drm.env, GEAK_GPU_REQUIRE_IDLE="1",
                                      KERNEL_EXTRACT_LOCK_DIR=str(Path(self.drm.base) / "locks")),
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 1, res.stdout + res.stderr)
        self.assertIn("foreign work", res.stderr)
        self.assertNotIn("RAN", res.stdout)


class SchedulerDispatchTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.drm = FakeDrm(self.root / "drm")
        for name in ("state", "logs", "runs"):
            (self.root / "work" / name).mkdir(parents=True)

    def tick(self, gpus, width, **extra):
        started = []

        class Backend:
            def alive(self, *a):
                return False

            def start(self, kernel, slot, env):
                started.append((kernel, list(slot), env.get("GEAK_PHYSICAL_GPUS")))
                return True

        args = SimpleNamespace(
            work_root=str(self.root / "work"), tasks_dir=str(self.root / "tasks"), target=1000,
            gpus=gpus, width=width, bad_node_limit=2, worker_env={}, keep_overrides={},
            slot_of={}, started_at={}, reconciled_workers=True, **extra)
        with patch.dict(os.environ, self.drm.env):
            scheduler.tick(args, Backend(), list(width), {}, {})
        return started

    def test_a_contended_gpu_is_skipped_and_the_task_goes_to_an_idle_one(self):
        self.drm.set(0, busy=0, vram_mb=286 * 1024)
        self.drm.set(1)
        self.assertEqual(self.tick([0, 1], {"taskA": 1}), [("taskA", [1], "1")])
        self.assertIn("not dispatching to gpu0", (self.root / "work" / "scheduler.log").read_text())

    def test_only_the_contended_gpu_is_left_out_not_its_whole_group(self):
        self.drm.set(0), self.drm.set(1, busy=100), self.drm.set(2), self.drm.set(3)
        started = self.tick([0, 1, 2, 3], {"wide": 2, "narrow": 1})
        self.assertEqual(started, [("wide", [2, 3], "2,3"), ("narrow", [0], "0")])

    def test_nothing_is_dispatched_while_every_gpu_is_contended(self):
        self.drm.set(0, busy=100)
        self.assertEqual(self.tick([0], {"taskA": 1}), [])

    def test_the_operator_can_allow_busy_gpus(self):
        self.drm.set(0, busy=100)
        self.assertEqual(self.tick([0], {"taskA": 1}, allow_busy_gpus=True), [("taskA", [0], "0")])


class WorkerFlowTest(unittest.TestCase):
    """The real scheduler and worker, on a generated fake task and a stub runner, against a fake
    driver: what happens to a task whose GPU is, or is not, in use when its attempt starts."""

    def run_campaign(self, busy):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        drm = FakeDrm(root / "drm")
        drm.set(0, busy=100 if busy else 0, vram_mb=300000 if busy else 283)
        subprocess.run([sys.executable, str(WORKLOAD / "engine" / "tools" / "make_fake_tasks.py"),
                        "--dir", str(root / "tasks"), "--count", "1"], check=True, capture_output=True)
        env = dict(os.environ, **drm.env, GEAK_GPU_IDLE_WAIT="0", STUB_RC="78")   # the stub ends the attempt
        subprocess.run([sys.executable, str(CAMPAIGN / "scheduler.py"), "run",
                        "--work-root", str(root / "work"), "--tasks-dir", str(root / "tasks"),
                        "--skill-root", str(WORKLOAD / "stages" / "optimize" / "geak-kernel-workflow"),
                        "--runner", str(WORKLOAD / "runners" / "stub.sh"), "--backend", "local",
                        "--gpus", "0", "--target", "60", "--tick-sec", "1", "--max-wall-sec", "30",
                        "--allow-busy-gpus"], env=env, capture_output=True, text=True, timeout=120)
        worker_log = (root / "work" / "logs" / "fake_kernel_0" / "worker.log").read_text()
        status = json.loads((root / "work" / "state" / "fake_kernel_0.status.json").read_text())
        return worker_log, status

    def test_a_task_on_a_gpu_that_stays_in_use_is_blocked_before_any_attempt(self):
        log, status = self.run_campaign(busy=True)
        self.assertEqual(status["status"], "blocked")
        self.assertIn("FATAL attempt 1", log)
        self.assertIn("refusing to measure on them", log)
        self.assertNotIn("runner rc=", log)

    def test_a_task_on_an_idle_gpu_starts_its_attempt(self):
        log, _ = self.run_campaign(busy=False)
        self.assertNotIn("FATAL attempt 1", log)
        self.assertIn("attempt 1: cap=", log)


if __name__ == "__main__":
    unittest.main()
