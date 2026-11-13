#!/usr/bin/env python
"""Locate, validate and cache PathMNIST - your own files or a fresh download.

    python scripts/prepare_data.py --platform kaggle --sizes 28 224     # data attached as a Kaggle dataset
    python scripts/prepare_data.py --source ~/Downloads/medmnist --sizes 28 224
    python scripts/prepare_data.py --platform kaggle --sizes 28 224 --download   # fetch what is missing from Zenodo
    python scripts/prepare_data.py --platform kaggle --list              # only show what was found

Accepted inputs (searched recursively): MedMNIST ``.npz`` files (pathmnist.npz,
pathmnist_64.npz, pathmnist_128.npz, pathmnist_224.npz - any file name works, contents are
checked), or folders containing train_images.npy / train_labels.npy / ... (an unpacked .npz).
"""
import argparse

import _bootstrap  # noqa: F401

from crc_tissuenet.config import platform_overrides
from crc_tissuenet.data import (CLASS_NAMES, EXPECTED_COUNTS, SPLITS, channel_stats, find_sources, load_split,
                                prepare_data)
from crc_tissuenet.utils import get_logger, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[28], choices=[28, 64, 128, 224])
    ap.add_argument("--platform", default=None, help="kaggle | colab | local")
    ap.add_argument("--source", nargs="*", default=None, help="folders to search for your data files")
    ap.add_argument("--root", default=None, help="writable cache folder")
    ap.add_argument("--download", action="store_true", help="download if not found")
    ap.add_argument("--list", action="store_true", help="only list the data sources that were found")
    args = ap.parse_args()
    plat = platform_overrides(args.platform)
    root = args.root or plat.get("data.root", "data")
    sources = args.source if args.source is not None else plat.get("data.source_dirs", ["data"])
    download = args.download or bool(plat.get("data.download", False))
    log = get_logger()

    found = find_sources([root, *sources])
    log.info("Searched %s -> %d PathMNIST source(s)", [root, *sources], len(found))
    for f in found:
        flag = "OK" if f["complete"] and f["official_counts"] else ("non-official split sizes" if f["complete"] else "; ".join(f["problems"]))
        log.info("  %-4spx %-7s %s  %s  [%s]", f["size"], f["kind"], f["counts"], f["path"], flag)
    if args.list:
        return

    for size in args.sizes:
        folder = prepare_data(root, size, download=download, source_dirs=sources)
        summary = {"source_folder": str(folder)}
        for s in SPLITS:
            arr = load_split(folder, s, in_memory=False)
            counts = {CLASS_NAMES[c]: int((arr.labels == c).sum()) for c in range(len(CLASS_NAMES))}
            summary[s] = {"n": len(arr), "shape": list(arr.images.shape), "class_counts": counts}
            ok = "" if len(arr) == EXPECTED_COUNTS[s] else f"  (official: {EXPECTED_COUNTS[s]})"
            log.info("%3dpx %-5s n=%6d%s %s", size, s, len(arr), ok, counts)
        mean, std = channel_stats(load_split(folder, "train", in_memory=False).images)
        save_json({"mean": mean, "std": std}, f"{root}/pathmnist_{size}_train_stats.json")
        save_json(summary, f"{root}/pathmnist_{size}_summary.json")
        log.info("%3dpx ready in %s | train mean %s std %s", size, folder,
                 [round(m, 4) for m in mean], [round(s, 4) for s in std])


if __name__ == "__main__":
    main()
