#!/usr/bin/env python
"""One-time export of PathMNIST as ready-to-use (uncompressed) .npy arrays.

Run once in a CPU-only Kaggle notebook. Its output then serves as an input of the main notebook,
which memory-maps the arrays in place: no download, no decompression and no idle GPU time in any
later version.

    python scripts/export_npy.py --sizes 28 224 --out /kaggle/working --tmp /kaggle/temp

* Uses a matching .npz already attached under --source (default /kaggle/input), otherwise
  downloads the official file from Zenodo into --tmp (checksum-verified).
* Writes <out>/pathmnist_<size>/{train,val,test}_{images,labels}.npy and deletes the download.
* Output size: about 16.2 GB for 224 px plus 0.3 GB for 28 px, inside Kaggle's 20 GB /kaggle/working limit.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from crc_tissuenet.data import (ARRAY_KEYS, EXPECTED_COUNTS, _check_disk, _npz_headers, download_pathmnist,
                                find_sources, inspect_source, npz_to_memmap)
from crc_tissuenet.utils import get_logger


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[28, 224], choices=[28, 64, 128, 224])
    ap.add_argument("--out", default="/kaggle/working", help="where the .npy folders are written (saved output)")
    ap.add_argument("--tmp", default="/kaggle/temp", help="scratch folder for downloads (not saved)")
    ap.add_argument("--source", nargs="*", default=["/kaggle/input"], help="where to look for existing .npz files")
    args = ap.parse_args()
    log = get_logger()
    out, tmp = Path(args.out), Path(args.tmp)

    for size in args.sizes:
        target = out / f"pathmnist_{size}"
        if all((target / f"{k}.npy").exists() for k in ARRAY_KEYS):
            log.info("%dpx already exported in %s", size, target)
            continue
        local = [s for s in find_sources(args.source) if s["size"] == size and s["complete"] and s["kind"] == "npz"]
        if local:
            npz, downloaded = Path(local[0]["path"]), False
            log.info("%dpx: using attached %s", size, npz)
        else:
            npz, downloaded = download_pathmnist(tmp, size), True
        needed = sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in _npz_headers(npz).values())
        _check_disk(target, int(needed))
        log.info("%dpx: writing %.1f GB of .npy arrays to %s", size, needed / 1e9, target)
        npz_to_memmap(npz, target)
        if downloaded:
            npz.unlink()
        info = inspect_source(target)
        status = "OK" if info["complete"] and info["counts"] == EXPECTED_COUNTS else f"CHECK: {info}"
        log.info("%dpx exported: %s [%s]", size, info["counts"], status)
    total = sum(f.stat().st_size for f in out.rglob("*.npy")) / 1e9
    log.info("Done. %.1f GB in %s (free on disk: %.1f GB)", total, out, shutil.disk_usage(out).free / 1e9)


if __name__ == "__main__":
    main()
