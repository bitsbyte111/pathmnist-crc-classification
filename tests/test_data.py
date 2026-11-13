import numpy as np
import torch

from crc_tissuenet.augment import GPUAugment, hed_to_rgb, rgb_to_hed
from crc_tissuenet.data import (InMemoryBatchLoader, channel_stats, make_synthetic_split, npz_to_memmap,
                                stratified_subset)


def test_npz_to_memmap_roundtrip(tmp_path):
    imgs = np.random.randint(0, 255, (37, 8, 8, 3), dtype=np.uint8)
    labels = np.random.randint(0, 9, (37, 1))
    np.savez_compressed(tmp_path / "x.npz", train_images=imgs, train_labels=labels)
    out = npz_to_memmap(tmp_path / "x.npz", tmp_path / "mm", chunk_rows=5)
    assert np.array_equal(np.load(out["train_images"], mmap_mode="r"), imgs)
    assert np.array_equal(np.load(out["train_labels"]), labels)
    npz_to_memmap(tmp_path / "x.npz", tmp_path / "mm")  # idempotent


def test_stratified_subset_keeps_all_classes():
    arr = make_synthetic_split(900, 8, seed=0)
    sub = stratified_subset(arr, 0.1, seed=0)
    assert set(np.unique(sub.labels)) == set(np.unique(arr.labels))
    assert 80 <= len(sub) <= 100


def test_channel_stats_match_numpy():
    arr = make_synthetic_split(50, 8, seed=1)
    mean, std = channel_stats(arr.images, chunk=7)
    x = arr.images.astype(np.float64) / 255
    assert np.allclose(mean, x.mean((0, 1, 2)), atol=1e-6)
    assert np.allclose(std, x.std((0, 1, 2)), atol=1e-6)


def test_in_memory_loader_covers_all():
    arr = make_synthetic_split(100, 8, seed=2)
    loader = InMemoryBatchLoader(arr, 32, shuffle=True, seed=0)
    ys = torch.cat([y for _, y in loader])
    assert len(loader) == 4 and sorted(ys.tolist()) == sorted(arr.labels.tolist())
    x, _ = next(iter(loader))
    assert x.dtype == torch.uint8 and x.shape[1:] == (3, 8, 8)


def test_hed_roundtrip_and_augment_range():
    x = torch.rand(4, 3, 16, 16) * 0.9 + 0.05
    assert torch.allclose(hed_to_rgb(rgb_to_hed(x)), x, atol=1e-4)
    aug = GPUAugment(True, True, 0.2, 0.1).train()
    y = aug(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    assert torch.equal(aug.eval()(x), x)  # no augmentation at evaluation time


def test_train_loader_never_yields_batch_of_one(tmp_path):
    from crc_tissuenet.config import load_config
    from crc_tissuenet.data import build_data
    from conftest import ROOT

    cfg = load_config(ROOT / "configs/models/b1_cnn.yaml", {"data.synthetic": True, "data.synthetic_samples": 129,
                                                             "train.batch_size": 64})
    data = build_data(cfg)
    sizes = [len(y) for _, y in data.loaders["train"]]
    assert sizes == [64, 64]                                   # 129 = 2 x 64 + 1 -> the lone image is dropped
    assert sum(len(y) for _, y in data.loaders["val"]) == 64   # evaluation keeps everything
