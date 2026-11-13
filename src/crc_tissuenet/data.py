"""PathMNIST data access.

Design choices
--------------
* **Official splits only** (train / val / test) so results are comparable with MedMNIST v2.
* **Low-RAM storage.** ``pathmnist_224.npz`` holds ~13.5 GB of training pixels, more than a
  free Colab session's RAM. :func:`npz_to_memmap` streams each array out of the ``.npz`` zip
  in chunks into an uncompressed ``.npy`` file that is then memory-mapped, so RAM use stays
  small at every resolution.
* **uint8 on the CPU, float on the GPU.** Loaders yield ``uint8`` tensors (N, 3, H, W);
  conversion to float, augmentation, resizing and normalisation happen on the GPU
  (see :mod:`crc_tissuenet.augment` and :class:`crc_tissuenet.models.wrapper.ClassifierWrapper`).
* **Synthetic mode** produces a tiny, learnable fake dataset for unit tests and CI.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import get
from .utils import get_logger, load_json, save_json, seed_worker

CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_DESCRIPTIONS = {
    "ADI": "Adipose", "BACK": "Background", "DEB": "Debris", "LYM": "Lymphocytes",
    "MUC": "Mucus", "MUS": "Smooth muscle", "NORM": "Normal colon mucosa",
    "STR": "Cancer-associated stroma", "TUM": "Colorectal adenocarcinoma epithelium",
}
NUM_CLASSES = len(CLASS_NAMES)
SPLITS = ("train", "val", "test")
VALID_SIZES = (28, 64, 128, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- files
def npz_filename(size: int, flag: str = "pathmnist") -> str:
    """MedMNIST naming: ``pathmnist.npz`` (28 px) and ``pathmnist_{size}.npz`` otherwise."""
    if size not in VALID_SIZES:
        raise ValueError(f"size must be one of {VALID_SIZES}, got {size}")
    return f"{flag}.npz" if size == 28 else f"{flag}_{size}.npz"


# Official MedMNIST v2 / MedMNIST+ files on Zenodo (same URLs and checksums as medmnist.INFO).
ZENODO_FILES = {
    28: ("https://zenodo.org/records/10519652/files/pathmnist.npz?download=1",
         "a8b06965200029087d5bd730944a56c1"),
    64: ("https://zenodo.org/records/10519652/files/pathmnist_64.npz?download=1",
         "55aa9c1e0525abe5a6b9d8343a507616"),
    128: ("https://zenodo.org/records/10519652/files/pathmnist_128.npz?download=1",
          "ac42d08fb904d92c244187169d1fd1d9"),
    224: ("https://zenodo.org/records/10519652/files/pathmnist_224.npz?download=1",
          "2c51a510bcdc9cf8ddb2af93af1eadec"),
}


def _download_file(url: str, folder: Path, filename: str, md5: str) -> None:
    """Download with resume-safe temp file and MD5 check (torchvision's downloader)."""
    from torchvision.datasets.utils import download_url

    download_url(url, root=str(folder), filename=filename, md5=md5)


def download_pathmnist(root: str | Path, size: int) -> Path:
    """Download the official ``pathmnist[_size].npz`` from Zenodo into ``root``.

    The file is verified against its MD5 checksum and is NOT loaded into memory (the
    ``medmnist`` dataset class would load each 224 px split fully into RAM).
    """
    url, md5 = ZENODO_FILES[size]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / npz_filename(size)
    if not target.exists():
        get_logger().info("Downloading %s from Zenodo (checksum-verified) ...", target.name)
        _download_file(url, root, target.name, md5)
    return target


def _read_npy_header(fp) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(fp)
    return np.lib.format.read_array_header_2_0(fp)


def npz_to_memmap(npz_path: str | Path, out_dir: str | Path, chunk_rows: int = 2048) -> dict[str, Path]:
    """Stream every array of an ``.npz`` archive into a standalone ``.npy`` file.

    Peak memory is ``chunk_rows`` rows (e.g. 2048 x 224 x 224 x 3 ~ 300 MB), never the full
    array. Already-converted arrays are skipped, so the call is idempotent.
    """
    npz_path, out_dir = Path(npz_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    with zipfile.ZipFile(npz_path) as zf:
        for member in zf.namelist():
            key = member[:-4] if member.endswith(".npy") else member
            target = out_dir / f"{key}.npy"
            outputs[key] = target
            if target.exists():
                continue
            tmp = target.with_suffix(".npy.partial")
            with zf.open(member) as fp:
                shape, fortran, dtype = _read_npy_header(fp)
                if fortran:
                    raise ValueError("Fortran-ordered arrays are not supported")
                out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=shape)
                row_items = int(np.prod(shape[1:])) if len(shape) > 1 else 1
                row_bytes = row_items * dtype.itemsize
                for start in range(0, shape[0], chunk_rows):
                    n = min(chunk_rows, shape[0] - start)
                    buf = fp.read(n * row_bytes)
                    out[start:start + n] = np.frombuffer(buf, dtype=dtype).reshape((n,) + tuple(shape[1:]))
                out.flush()
                del out
            tmp.rename(target)
    return outputs


EXPECTED_COUNTS = {"train": 89_996, "val": 10_004, "test": 7_180}
ARRAY_KEYS = tuple(f"{s}_{k}" for s in SPLITS for k in ("images", "labels"))


def _npz_headers(path: Path) -> dict[str, tuple]:
    """Shapes/dtypes of every array in an ``.npz`` - reads headers only (fast even for 12 GB files)."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            with zf.open(member) as fp:
                shape, _, dtype = _read_npy_header(fp)
            out[member[:-4] if member.endswith(".npy") else member] = (tuple(shape), np.dtype(dtype))
    return out


def _npy_dir_headers(folder: Path) -> dict[str, tuple]:
    out = {}
    for key in ARRAY_KEYS:
        f = folder / f"{key}.npy"
        if f.exists():
            with open(f, "rb") as fp:
                shape, _, dtype = _read_npy_header(fp)
            out[key] = (tuple(shape), np.dtype(dtype))
    return out


def inspect_source(path: str | Path) -> dict:
    """Validate a PathMNIST source (``.npz`` file or folder of ``.npy`` files) from headers.

    Returns ``{"path", "kind", "size", "counts", "complete", "problems"}``.
    """
    path = Path(path)
    kind = "npz" if path.is_file() else "npy_dir"
    try:
        headers = _npz_headers(path) if kind == "npz" else _npy_dir_headers(path)
    except (zipfile.BadZipFile, ValueError, OSError) as e:
        return {"path": str(path), "kind": kind, "size": None, "counts": {}, "complete": False,
                "problems": [f"unreadable: {e}"]}
    problems = [f"missing array '{k}'" for k in ARRAY_KEYS if k not in headers]
    size, counts = None, {}
    for split in SPLITS:
        img = headers.get(f"{split}_images")
        lab = headers.get(f"{split}_labels")
        if img is None or lab is None:
            continue
        shape, dtype = img
        if len(shape) != 4 or shape[3] != 3 or shape[1] != shape[2]:
            problems.append(f"{split}_images has shape {shape}, expected (N, S, S, 3)")
            continue
        if dtype != np.uint8:
            problems.append(f"{split}_images dtype {dtype}, expected uint8")
        if lab[0][0] != shape[0]:
            problems.append(f"{split}: {shape[0]} images but {lab[0][0]} labels")
        size = size or shape[1]
        counts[split] = shape[0]
    complete = not problems and size in VALID_SIZES
    return {"path": str(path), "kind": kind, "size": size, "counts": counts, "complete": complete,
            "problems": problems, "official_counts": counts == EXPECTED_COUNTS}


def find_sources(search_dirs) -> list[dict]:
    """Search directories recursively for PathMNIST ``.npz`` files or unpacked ``.npy`` folders."""
    if isinstance(search_dirs, (str, Path)):
        search_dirs = [search_dirs]
    found, seen = [], set()
    for d in search_dirs or []:
        d = Path(d)
        if not d.exists():
            continue
        candidates = list(d.rglob("*.npz")) + [p.parent for p in d.rglob("train_images.npy")]
        for c in candidates:
            key = str(c.resolve())
            if key in seen or c.name.endswith(".partial"):
                continue
            seen.add(key)
            info = inspect_source(c)
            if info["counts"]:
                found.append(info)
    return found


def _check_disk(target: Path, needed_bytes: int) -> None:
    import shutil

    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    if free < needed_bytes * 1.05:
        raise OSError(f"Not enough disk space in {target}: need ~{needed_bytes / 1e9:.1f} GB, "
                      f"have {free / 1e9:.1f} GB. Set data.root to a larger writable location.")


class _FileLock:
    """Inter-process lock (POSIX) so parallel workers never build the same cache twice."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w")
        try:
            import fcntl

            fcntl.flock(self.fh, fcntl.LOCK_EX)
        except ImportError:  # Windows: no locking (run one process at a time)
            pass
        return self

    def __exit__(self, *exc):
        self.fh.close()


def prepare_data(root: str | Path, size: int, download: bool = True, source_dirs=None) -> Path:
    with _FileLock(Path(root) / f".pathmnist_{size}.lock"):
        return _prepare_data(root, size, download, source_dirs)


def _prepare_data(root: str | Path, size: int, download: bool = True, source_dirs=None) -> Path:
    """Return a directory of ``{split}_{images,labels}.npy`` files for ``size``.

    Resolution order:
    1. an existing converted cache in ``root``;
    2. a folder of ``.npy`` files found under ``root`` or ``source_dirs`` - used in place
       (read-only is fine, e.g. a Kaggle dataset that was unpacked on upload);
    3. an ``.npz`` file found there - streamed into a memory-mappable cache in ``root``;
    4. download the official file from Zenodo (if ``download`` is true); after conversion the
       downloaded ``.npz`` is deleted to free scratch disk (it can always be re-downloaded).
    """
    root = Path(root)
    log = get_logger()
    mm_dir = root / f"pathmnist_{size}_mmap"
    if all((mm_dir / f"{k}.npy").exists() for k in ARRAY_KEYS):
        return mm_dir
    sources = find_sources([root, *([source_dirs] if isinstance(source_dirs, (str, Path)) else source_dirs or [])])
    matching = [s for s in sources if s["size"] == size and s["complete"]]
    for s in matching:
        if not s["official_counts"]:
            log.warning("%s has split sizes %s (official: %s)", s["path"], s["counts"], EXPECTED_COUNTS)
    npy = [s for s in matching if s["kind"] == "npy_dir"]
    if npy:
        log.info("Using unpacked arrays in place: %s", npy[0]["path"])
        return Path(npy[0]["path"])
    npz = [s for s in matching if s["kind"] == "npz"]
    downloaded = False
    if npz:
        src = Path(npz[0]["path"])
    elif download:
        src = download_pathmnist(root, size)
        downloaded = True
    else:
        found = ", ".join(f"{s['path']} ({s['size']}px)" for s in sources) or "none"
        raise FileNotFoundError(
            f"No complete PathMNIST {size}px data found in {root} or {source_dirs}. Found: {found}. "
            f"Expected '{npz_filename(size)}' (MedMNIST) or a folder with train_images.npy etc.")
    headers = _npz_headers(src)
    needed = sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in headers.values())
    _check_disk(mm_dir, needed)
    log.info("Converting %s -> %s (%.1f GB, one-off)", src, mm_dir, needed / 1e9)
    npz_to_memmap(src, mm_dir)
    if downloaded:
        src.unlink()
        log.info("Deleted the downloaded %s (converted cache kept)", src.name)
    return mm_dir


# --------------------------------------------------------------------------- arrays
@dataclass
class SplitArrays:
    images: np.ndarray  # (N, H, W, 3) uint8, possibly memory-mapped
    labels: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return len(self.labels)


def load_split(mm_dir: Path, split: str, in_memory: bool) -> SplitArrays:
    images = np.load(mm_dir / f"{split}_images.npy", mmap_mode=None if in_memory else "r")
    labels = np.load(mm_dir / f"{split}_labels.npy").reshape(-1).astype(np.int64)
    return SplitArrays(images, labels)


def make_synthetic_split(n: int, size: int, seed: int, noise: float = 0.15) -> SplitArrays:
    """Learnable fake data: each class has its own mean colour plus a stripe pattern."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, NUM_CLASSES, n).astype(np.int64)
    class_rgb = np.random.default_rng(0).uniform(0.2, 0.8, (NUM_CLASSES, 3))
    yy = np.arange(size)[None, :, None, None]
    imgs = class_rgb[labels][:, None, None, :] + 0.1 * np.sin(yy * (labels[:, None, None, None] + 1) / 3)
    imgs = imgs + rng.normal(0, noise, (n, size, size, 3))
    return SplitArrays((np.clip(imgs, 0, 1) * 255).astype(np.uint8), labels)


def stratified_subset(arr: SplitArrays, fraction: float, seed: int) -> SplitArrays:
    """Keep ``fraction`` of each class (for fast debugging runs)."""
    if fraction >= 1.0:
        return arr
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(arr.labels):
        idx = np.flatnonzero(arr.labels == c)
        keep.append(rng.choice(idx, max(1, int(round(len(idx) * fraction))), replace=False))
    keep = np.sort(np.concatenate(keep))
    return SplitArrays(np.asarray(arr.images[keep]), arr.labels[keep])


def channel_stats(images: np.ndarray, chunk: int | None = None,
                  max_chunk_bytes: int = 256 * 2**20) -> tuple[list[float], list[float]]:
    """Per-channel mean/std in [0, 1] units, computed in chunks (works on memmaps).

    The chunk size adapts to the image size so peak memory stays near ``max_chunk_bytes``
    (e.g. ~220 images at 224 px instead of several GB).
    """
    if chunk is None:
        per_image = int(np.prod(images.shape[1:])) * 8  # float64
        chunk = max(1, max_chunk_bytes // max(per_image, 1))
    s = np.zeros(3)
    ss = np.zeros(3)
    n = 0
    for i in range(0, len(images), chunk):
        x = np.asarray(images[i:i + chunk], dtype=np.float64) / 255.0
        s += x.sum(axis=(0, 1, 2))
        ss += (x ** 2).sum(axis=(0, 1, 2))
        n += x.shape[0] * x.shape[1] * x.shape[2]
    mean = s / n
    std = np.sqrt(ss / n - mean ** 2)
    return mean.tolist(), std.tolist()


# --------------------------------------------------------------------------- loaders
class UInt8ImageDataset(Dataset):
    """Returns ``(uint8 tensor [3, H, W], int label)``; reads lazily from memmaps."""

    def __init__(self, arrays: SplitArrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(np.ascontiguousarray(self.arrays.images[idx])).permute(2, 0, 1)
        return img, int(self.arrays.labels[idx])


class InMemoryBatchLoader:
    """Very fast loader for small images that fit in RAM: slices whole batches at once.

    Avoids per-sample Python overhead, which dominates training time at 28 px.
    """

    def __init__(self, arrays: SplitArrays, batch_size: int, shuffle: bool, seed: int,
                 drop_last: bool = False):
        self.images = torch.from_numpy(np.ascontiguousarray(arrays.images)).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.from_numpy(arrays.labels)
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.generator = torch.Generator().manual_seed(seed)
        self.dataset = arrays  # for len(loader.dataset)

    def __len__(self) -> int:
        n = len(self.labels)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        n = len(self.labels)
        order = torch.randperm(n, generator=self.generator) if self.shuffle else torch.arange(n)
        for b in range(len(self)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            yield self.images[idx], self.labels[idx]


@dataclass
class DataBundle:
    loaders: dict
    arrays: dict
    mean: list
    std: list
    class_counts: dict
    size: int


def build_data(cfg: dict) -> DataBundle:
    """Create train/val/test loaders from a resolved config."""
    seed = int(get(cfg, "experiment.seed", 42))
    size = int(get(cfg, "data.size", 28))
    bs = int(get(cfg, "train.batch_size", 128))
    eval_bs = int(get(cfg, "data.eval_batch_size", max(bs, 256)))
    frac = float(get(cfg, "data.subset_fraction", 1.0))
    root = Path(get(cfg, "data.root", "data"))

    if get(cfg, "data.synthetic", False):
        n = int(get(cfg, "data.synthetic_samples", 512))
        data_seed = int(get(cfg, "data.synthetic_seed", 0))  # fixed: same data for every run seed
        arrays = {s: make_synthetic_split(n if s == "train" else n // 2, size, data_seed + i)
                  for i, s in enumerate(SPLITS)}
        in_memory = True
        stats_file = None
    else:
        in_memory_opt = get(cfg, "data.in_memory", "auto")
        in_memory = (size <= 64) if in_memory_opt == "auto" else bool(in_memory_opt)
        mm_dir = prepare_data(root, size, download=bool(get(cfg, "data.download", True)),
                              source_dirs=get(cfg, "data.source_dirs", None))
        arrays = {s: load_split(mm_dir, s, in_memory) for s in SPLITS}
        stats_file = root / f"pathmnist_{size}_train_stats.json"

    # Normalisation statistics: always from the FULL training split (never val/test: no leakage),
    # computed once and cached, so subset runs use the same statistics as full runs.
    if stats_file is not None and stats_file.exists():
        stats = load_json(stats_file)
        mean, std = stats["mean"], stats["std"]
    else:
        mean, std = channel_stats(arrays["train"].images)
        if stats_file is not None:
            save_json({"mean": mean, "std": std}, stats_file)

    # Subsets (hyperparameter studies, debugging) shrink the TRAINING split only: validation and
    # test are always complete, so every candidate is judged on identical data.
    arrays["train"] = stratified_subset(arrays["train"], frac, seed)

    workers = int(get(cfg, "data.num_workers", 2))
    loaders = {}
    for s in SPLITS:
        train = s == "train"
        batch = bs if train else eval_bs
        # Training drops the last incomplete batch: a batch of 1 breaks BatchNorm, and tiny batches give
        # very noisy updates. With shuffling, a different handful of images is skipped each epoch.
        drop_last = train and len(arrays[s]) >= batch
        if in_memory:
            loaders[s] = InMemoryBatchLoader(arrays[s], batch, shuffle=train, seed=seed, drop_last=drop_last)
        else:
            g = torch.Generator().manual_seed(seed)
            loaders[s] = DataLoader(
                UInt8ImageDataset(arrays[s]), batch_size=batch, shuffle=train, num_workers=workers,
                pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker, generator=g,
                persistent_workers=workers > 0, drop_last=drop_last,
            )
    counts = {s: np.bincount(arrays[s].labels, minlength=NUM_CLASSES).tolist() for s in SPLITS}
    return DataBundle(loaders, arrays, mean, std, counts, size)
