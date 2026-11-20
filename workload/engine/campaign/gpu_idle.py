#!/usr/bin/env python3
"""Is another tenant using these physical GPUs?

A measurement taken on a GPU that somebody else is using is not a measurement of the
candidate. The kernel driver knows: `gpu_busy_percent` and `mem_info_vram_used` under the
GPU's sysfs device directory, the same signals `stages/_shared/gpu_lock.sh` reads before it
runs a command. amd-smi is deliberately not used: it returns nothing while another process
holds the GPU, which is exactly when the answer matters.

    gpu_idle.py 0,1,2,3        # exit 0 when every GPU is idle, 1 and the reason when not

Thresholds (environment): GEAK_GPU_MAX_BUSY_PCT (default 5) and GEAK_GPU_MAX_VRAM_MB
(default 1024). GEAK_GPU_REQUIRE_IDLE=0 turns the check off. A GPU whose render node or
counters cannot be found is treated as idle, as in gpu_lock.sh: not knowing is not evidence of a
tenant. GPU N is the Nth GPU in ROCm topology order (GEAK_KFD_TOPOLOGY overrides the sysfs path).
"""
import os
import sys

DRM_ROOT = "/sys/class/drm"
KFD_TOPOLOGY = "/sys/class/kfd/kfd/topology/nodes"


def render_minors(topology=None, environ=None):
    """Render-node minors of the GPUs in ROCm topology order (CPU nodes carry none).

    GPU index N is the Nth entry, the same mapping the container backend uses for its
    --device arguments. The minors are not assumed to advance by a fixed stride."""
    env = os.environ if environ is None else environ
    root = topology or env.get("GEAK_KFD_TOPOLOGY") or KFD_TOPOLOGY
    minors = []
    try:
        for name in sorted(os.listdir(root), key=int):
            with open(os.path.join(root, name, "properties")) as f:
                fields = dict(line.split() for line in f.read().splitlines() if len(line.split()) == 2)
            minor = int(fields.get("drm_render_minor", 0))
            if minor:
                minors.append(minor)
    except (OSError, ValueError):
        return []
    return minors


def device_dir(gpu, drm_root=None, minors=None):
    """The sysfs device directory of physical GPU `gpu`, or None when the mapping is unknown."""
    minors = render_minors() if minors is None else minors
    if not 0 <= gpu < len(minors):
        return None
    root = drm_root or os.environ.get("GEAK_DRM_ROOT") or DRM_ROOT
    path = os.path.join(root, "renderD%d" % minors[gpu], "device")
    return os.path.realpath(path) if os.path.isdir(path) else None


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def foreign_use(gpus, drm_root=None, environ=None, minors=None, render_paths=None):
    """[{gpu, busy_pct, vram_mb}] for each of `gpus` above the idle thresholds.

    `minors` (or the render nodes of `render_paths`, as an explicit container mapping) override
    the topology lookup. A GPU whose render node or counters cannot be found counts as idle."""
    env = os.environ if environ is None else environ
    if env.get("GEAK_GPU_REQUIRE_IDLE", "1") != "1":
        return []
    if minors is None and render_paths:
        minors = [int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0) for p in render_paths]
    minors = render_minors(environ=env) if minors is None else minors
    max_busy = int(env.get("GEAK_GPU_MAX_BUSY_PCT", "5"))
    max_vram = int(env.get("GEAK_GPU_MAX_VRAM_MB", "1024"))
    busy_gpus = []
    for gpu in gpus:
        dev = device_dir(gpu, drm_root, minors)
        if dev is None:
            continue
        busy = _read_int(os.path.join(dev, "gpu_busy_percent"))
        vram = _read_int(os.path.join(dev, "mem_info_vram_used"))
        if busy is None:
            continue                                  # cannot tell: do not block the run
        vram_mb = (vram or 0) // (1 << 20)
        if busy > max_busy or vram_mb > max_vram:
            busy_gpus.append({"gpu": gpu, "busy_pct": busy, "vram_mb": vram_mb})
    return busy_gpus


def describe(rows):
    return "; ".join("GPU %d busy %d%%, %d MB VRAM in use" % (r["gpu"], r["busy_pct"], r["vram_mb"])
                     for r in rows)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    rows = foreign_use([int(g) for g in argv[0].split(",") if g.strip() != ""])
    if rows:
        print(describe(rows))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
