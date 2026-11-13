#!/usr/bin/env python
"""Run an experiment suite (E1-E5). Finished runs are skipped, so this is safe to re-run.

    python scripts/run_suite.py --suite configs/suites/e1_e3_benchmark.yaml
    python scripts/run_suite.py --suite ... --only P1_resnet50 P5_deit_small --seeds 42
    python scripts/run_suite.py --suite ... --platform kaggle --gpus auto     # one run per GPU
"""
import argparse

import _bootstrap  # noqa: F401

from crc_tissuenet.config import cli_overrides
from crc_tissuenet.parallel import resolve_devices
from crc_tissuenet.suite import run_suite


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="run ids to run")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("-o", "--override", nargs="*", default=[])
    ap.add_argument("--platform", default=None, help="kaggle | colab | local | path/to/profile.yaml")
    ap.add_argument("--gpus", default="auto", help="auto (all GPUs if >1) | 0,1 | none - one run per GPU in parallel")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    extra = cli_overrides(args.platform, args.override)
    results = run_suite(args.suite, args.output_dir, args.only, args.seeds, extra, args.dry_run,
                        resolve_devices(args.gpus))
    if any(r.get("status") == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
