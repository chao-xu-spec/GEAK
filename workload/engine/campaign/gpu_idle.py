#!/usr/bin/env python3
"""Is another tenant using these physical GPUs?

A measurement taken on a GPU that somebody else is using is not a measurement of the
candidate. The kernel driver knows: `gpu_busy_percent` and `mem_info_vram_used` under the
GPU's sysfs device directory, the same signals `stages/_shared/gpu_lock.sh` reads before it
runs a command. amd-smi is deliberately not used: it returns nothing while another process
holds the GPU, which is exactly when the answer matters.

    gpu_idle.py 0,1,2,3        # exit 0 when every GPU is idle, 1 and the reason when not

Thresholds (environment): GEAK_GPU_MAX_BUSY_PCT (default 5) and GEAK_GPU_MAX_VRAM_MB
(default 1024). GEAK_GPU_REQUIRE_IDLE=0 turns the check off. A GPU whose counters cannot be
read is treated as idle, as in gpu_lock.sh: not knowing is not evidence of a tenant.
"""
import os
import sys

DRM_ROOT = "/sys/class/drm"


def device_dir(gpu, drm_root=None):
    """The sysfs device directory of physical GPU `gpu` (render node 128 + 8 * gpu), or None."""
    root = drm_root or os.environ.get("GEAK_DRM_ROOT") or DRM_ROOT
    path = os.path.join(root, "renderD%d" % (128 + 8 * gpu), "device")
    return os.path.realpath(path) if os.path.isdir(path) else None


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def foreign_use(gpus, drm_root=None, environ=None):
    """[{gpu, busy_pct, vram_mb}] for each of `gpus` above the idle thresholds."""
    env = os.environ if environ is None else environ
    if env.get("GEAK_GPU_REQUIRE_IDLE", "1") != "1":
        return []
    max_busy = int(env.get("GEAK_GPU_MAX_BUSY_PCT", "5"))
    max_vram = int(env.get("GEAK_GPU_MAX_VRAM_MB", "1024"))
    busy_gpus = []
    for gpu in gpus:
        dev = device_dir(gpu, drm_root)
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
