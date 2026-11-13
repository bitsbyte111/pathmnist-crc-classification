"""Run independent experiments concurrently, one per device (e.g. Kaggle's "GPU T4 x2").

Ablation candidates, seeds and benchmark models are independent, so the most efficient way
to use several GPUs is one experiment per GPU (no gradient synchronisation, no batch
splitting). Each worker process is pinned to one device for its whole life; jobs are
handed out as workers become free.

    devices = resolve_devices("auto")            # ["cuda:0", "cuda:1"] on a 2-GPU machine
    results = run_jobs([(cfg_a, dir_a), (cfg_b, dir_b)], devices)

With zero or one device, jobs run sequentially in the current process.
A job that raises is reported as ``{"status": "failed", ...}`` and the others continue.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import apply_overrides
from .utils import get_logger

_DEVICE: str | None = None


def resolve_devices(spec: str | list | None = "auto") -> list[str]:
    """``auto`` -> all CUDA GPUs if more than one, else []; ``"0,1"`` -> cuda:0, cuda:1;
    ``"none"`` -> []; an explicit list such as ``["cpu", "cpu"]`` is returned as is (tests)."""
    if spec is None or (isinstance(spec, str) and spec.lower() in ("", "none", "off")):
        return []
    if isinstance(spec, (list, tuple)):
        return [str(d) for d in spec]
    if spec.lower() == "auto":
        import torch

        n = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n)] if n > 1 else []
    return [d if ":" in d or d == "cpu" else f"cuda:{d}" for d in spec.replace(" ", "").split(",") if d]


def _init_worker(queue) -> None:  # pragma: no cover - runs in child processes
    global _DEVICE
    _DEVICE = queue.get()
    os.environ["CRC_LOG_TAG"] = f"[{_DEVICE}] "
    if _DEVICE.startswith("cuda"):
        import torch

        torch.cuda.set_device(int(_DEVICE.split(":")[1]))


def _failed(e: Exception) -> dict:
    return {"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
            "val": {}, "model": {}}


def _run_one(cfg: dict, run_dir: str, device: str | None) -> dict:
    from .engine import run_experiment

    if device:
        cfg = apply_overrides(cfg, {"experiment.device": device})
    try:
        return run_experiment(cfg, run_dir)
    except Exception as e:  # keep the rest of the queue going
        get_logger().error("Run %s failed: %s\n%s", run_dir, e, traceback.format_exc())
        return _failed(e)


def _worker_job(cfg: dict, run_dir: str) -> dict:  # pragma: no cover - child process
    return _run_one(cfg, run_dir, _DEVICE)


def run_jobs(jobs: list[tuple[dict, str | Path]], devices: list[str]) -> list[dict]:
    """Run ``(config, run_dir)`` jobs; returns results in the same order as ``jobs``."""
    log = get_logger()
    if not jobs:
        return []
    if len(devices) <= 1:
        dev = devices[0] if devices else None
        return [_run_one(cfg, str(rd), dev) for cfg, rd in jobs]

    log.info("Running %d jobs on %d devices in parallel: %s", len(jobs), len(devices), ", ".join(devices))
    ctx = mp.get_context("spawn")  # CUDA requires spawn, not fork
    manager = ctx.Manager()
    queue = manager.Queue()
    for d in devices:
        queue.put(d)
    results: list[dict | None] = [None] * len(jobs)
    try:
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx, initializer=_init_worker,
                                 initargs=(queue,)) as pool:
            futures = {pool.submit(_worker_job, cfg, str(rd)): i for i, (cfg, rd) in enumerate(jobs)}
            for done, fut in enumerate(as_completed(futures), 1):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:  # worker crashed (e.g. CUDA out of memory killed it)
                    results[i] = _failed(e)
                status = results[i].get("status")
                log.info("[%d/%d] finished %s (%s)", done, len(jobs), jobs[i][1], status)
    finally:
        manager.shutdown()
    n_failed = sum(r.get("status") == "failed" for r in results)
    if n_failed:
        log.warning("%d of %d jobs failed - see the logs above; re-run to retry only those", n_failed, len(jobs))
    return results  # type: ignore[return-value]
