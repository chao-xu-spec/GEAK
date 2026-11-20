"""A GPU another tenant is using is not handed a task and is not measured on."""
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CAMPAIGN = Path(__file__).resolve().parents[1] / "campaign"
sys.path.insert(0, str(CAMPAIGN))


def load(name):
    spec = importlib.util.spec_from_file_location(name, CAMPAIGN / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu_idle = load("gpu_idle")
scheduler = load("scheduler")


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

    def set(self, gpu, busy=0, vram_mb=283, counters=True):
        device = self.root / ("renderD%d" % self.minors[gpu]) / "device"
        device.mkdir(parents=True, exist_ok=True)
        if counters:
            (device / "gpu_busy_percent").write_text("%d\n" % busy)
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
        for gpu in range(3):
            self.drm.set(gpu)
        self.assertEqual(gpu_idle.foreign_use([0, 1, 2]), [])

    def test_a_busy_gpu_and_an_idle_server_holding_the_memory_are_both_flagged(self):
        self.drm.set(0, busy=100, vram_mb=1837)
        self.drm.set(1, busy=0, vram_mb=286 * 1024)      # a loaded model with no traffic
        self.drm.set(2)
        rows = gpu_idle.foreign_use([0, 1, 2])
        self.assertEqual([r["gpu"] for r in rows], [0, 1])
        self.assertIn("GPU 1 busy 0%, 292864 MB VRAM in use", gpu_idle.describe(rows))

    def test_unreadable_counters_are_not_evidence_of_a_tenant(self):
        self.drm.set(0, counters=False)
        self.assertEqual(gpu_idle.foreign_use([0, 5]), [])   # GPU 5 is not in the topology

    def test_thresholds_and_the_switch_come_from_the_environment(self):
        self.drm.set(0, busy=10, vram_mb=2000)
        self.assertEqual(len(gpu_idle.foreign_use([0])), 1)
        env = {"GEAK_GPU_MAX_BUSY_PCT": "50", "GEAK_GPU_MAX_VRAM_MB": "4096"}
        self.assertEqual(gpu_idle.foreign_use([0], environ=dict(os.environ, **env)), [])
        self.assertEqual(gpu_idle.foreign_use([0], environ=dict(os.environ, GEAK_GPU_REQUIRE_IDLE="0")), [])

    def test_gpu_index_follows_topology_order_not_a_render_minor_stride(self):
        for gpu in range(4):
            self.drm.set(gpu)
        self.drm.set(2, busy=100)                      # minor 151, not 128 + 8 * 2
        rows = gpu_idle.foreign_use([0, 1, 2, 3])
        self.assertEqual([r["gpu"] for r in rows], [2])
        self.assertEqual(gpu_idle.render_minors(), [129, 137, 151, 160])

    def test_an_explicit_container_mapping_overrides_the_topology(self):
        for gpu in range(4):
            self.drm.set(gpu)
        self.drm.set(3, busy=100)                      # minor 160
        rows = gpu_idle.foreign_use([0], render_paths=["/dev/dri/renderD160", "/dev/dri/renderD129"])
        self.assertEqual([r["gpu"] for r in rows], [0])

    def test_an_unreadable_topology_means_no_mapping_and_no_block(self):
        self.drm.set(0, busy=100)
        with patch.dict(os.environ, {"GEAK_KFD_TOPOLOGY": "/nonexistent"}):
            self.assertEqual(gpu_idle.foreign_use([0]), [])

    def test_command_line_exit_code_and_reason(self):
        self.drm.set(0)
        self.drm.set(1, busy=99, vram_mb=5000)
        env = dict(os.environ, **self.drm.env)
        idle = subprocess.run([sys.executable, str(CAMPAIGN / "gpu_idle.py"), "0"], env=env,
                              capture_output=True, text=True)
        busy = subprocess.run([sys.executable, str(CAMPAIGN / "gpu_idle.py"), "0,1"], env=env,
                              capture_output=True, text=True)
        self.assertEqual((idle.returncode, idle.stdout), (0, ""))
        self.assertEqual(busy.returncode, 1)
        self.assertIn("GPU 1 busy 99%", busy.stdout)


class SchedulerDispatchTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.drm = FakeDrm(self.root / "drm")
        for name in ("state", "logs", "runs"):
            (self.root / "work" / name).mkdir(parents=True)

    def tick(self, gpus, **extra):
        started = []

        class Backend:
            def alive(self, *a):
                return False

            def start(self, kernel, slot, env):
                started.append((kernel, list(slot), env.get("GEAK_PHYSICAL_GPUS")))
                return True

        args = SimpleNamespace(
            work_root=str(self.root / "work"), tasks_dir=str(self.root / "tasks"), target=1000,
            gpus=gpus, width={"taskA": 1}, bad_node_limit=2, worker_env={}, keep_overrides={},
            slot_of={}, started_at={}, reconciled_workers=True, **extra)
        with patch.dict(os.environ, self.drm.env):
            scheduler.tick(args, Backend(), ["taskA"], {}, {})
        return started, args

    def test_a_contended_gpu_is_skipped_and_the_task_goes_to_an_idle_one(self):
        self.drm.set(0, busy=0, vram_mb=286 * 1024)
        self.drm.set(1)
        started, _ = self.tick([0, 1])
        self.assertEqual(started, [("taskA", [1], "1")])
        self.assertIn("not dispatching taskA to gpu0",
                      (self.root / "work" / "scheduler.log").read_text())

    def test_nothing_is_dispatched_while_every_gpu_is_contended(self):
        self.drm.set(0, busy=100)
        started, _ = self.tick([0])
        self.assertEqual(started, [])

    def test_the_operator_can_allow_busy_gpus(self):
        self.drm.set(0, busy=100)
        started, _ = self.tick([0], allow_busy_gpus=True)
        self.assertEqual(started, [("taskA", [0], "0")])


class WorkerGuardTest(unittest.TestCase):
    """The pre-attempt block of worker.sh, run against a fake driver and a stub `note`."""

    def run_guard(self, drm, wait):
        source = (CAMPAIGN / "worker.sh").read_text()
        block = re.search(r'(?ms)^  if \[ -n "\$\{GEAK_PHYSICAL_GPUS:-\}" \]; then\n.*?^  fi\n', source).group()
        script = "\n".join([
            'GPU_IDLE=%s' % (CAMPAIGN / "gpu_idle.py"),
            'note() { echo "NOTE: $*"; }',
            'write_status() { echo "STATUS: $*"; }',
            'sleep() { :; }',
            'N=3; CRED=10; BEST=1.0x; DONE_ATTEMPTS=2',
            block,
            'echo PROCEEDED'])
        env = dict(os.environ, **drm.env, GEAK_PHYSICAL_GPUS="0,1",
                   GEAK_GPU_IDLE_WAIT=str(wait))
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.drm = FakeDrm(tmp.name)

    def test_idle_gpus_let_the_attempt_start(self):
        self.drm.set(0), self.drm.set(1)
        res = self.run_guard(self.drm, 0)
        self.assertEqual((res.returncode, res.stdout.strip()), (0, "PROCEEDED"))

    def test_a_tenant_that_stays_blocks_the_task_instead_of_being_measured_on(self):
        self.drm.set(0), self.drm.set(1, busy=100, vram_mb=300000)
        res = self.run_guard(self.drm, 0)
        self.assertEqual(res.returncode, 78)
        self.assertNotIn("PROCEEDED", res.stdout)
        self.assertIn("FATAL attempt 3", res.stdout)
        self.assertIn("refusing to measure on them", res.stdout)
        self.assertIn("STATUS: blocked 10 1.0x 2", res.stdout)


if __name__ == "__main__":
    unittest.main()
