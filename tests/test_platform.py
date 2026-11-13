"""Data discovery (your own files, read-only inputs), platform profiles and multi-device runs."""
import os
import stat

import numpy as np
import pytest
from conftest import ROOT

from crc_tissuenet.config import apply_overrides, cli_overrides, load_config, platform_overrides
from crc_tissuenet.data import build_data, find_sources, inspect_source, make_synthetic_split, prepare_data
from crc_tissuenet.parallel import resolve_devices, run_jobs
from crc_tissuenet.suite import run_suite


def _write_fake_pathmnist(folder, size=28, n=(90, 30, 30), as_npz=True, name="pathmnist.npz"):
    arrays = {}
    for i, (split, k) in enumerate(zip(("train", "val", "test"), n)):
        a = make_synthetic_split(k, size, seed=i)
        arrays[f"{split}_images"] = a.images
        arrays[f"{split}_labels"] = a.labels.reshape(-1, 1)
    folder.mkdir(parents=True, exist_ok=True)
    if as_npz:
        np.savez_compressed(folder / name, **arrays)
        return folder / name
    for k, v in arrays.items():
        np.save(folder / f"{k}.npy", v)
    return folder


def _read_only(path):
    for p in [path, *path.rglob("*")]:
        os.chmod(p, stat.S_IREAD | stat.S_IEXEC if p.is_dir() else stat.S_IREAD)


def test_inspect_and_find(tmp_path):
    npz = _write_fake_pathmnist(tmp_path / "input" / "my-upload", name="anything.npz")
    folder = _write_fake_pathmnist(tmp_path / "input" / "unpacked" / "pathmnist_64", size=64, as_npz=False)
    (tmp_path / "input" / "junk.npz").write_bytes(b"not a zip")
    info = inspect_source(npz)
    assert info["complete"] and info["size"] == 28 and info["counts"]["train"] == 90
    assert not info["official_counts"]
    found = {f["size"]: f for f in find_sources(tmp_path / "input")}
    assert set(found) == {28, 64} and found[64]["kind"] == "npy_dir" and found[64]["path"] == str(folder)


def test_prepare_from_read_only_input(tmp_path):
    src = tmp_path / "kaggle_input"
    _write_fake_pathmnist(src / "ds", name="pathmnist.npz")
    _write_fake_pathmnist(src / "ds2", size=64, as_npz=False)
    _read_only(src)
    try:
        mm = prepare_data(tmp_path / "cache", 28, download=False, source_dirs=[src])
        assert mm.parent == tmp_path / "cache"                       # npz converted to writable cache
        assert prepare_data(tmp_path / "cache", 64, download=False, source_dirs=[src]) == src / "ds2"  # used in place
        with pytest.raises(FileNotFoundError, match="Found"):
            prepare_data(tmp_path / "cache", 128, download=False, source_dirs=[src])
    finally:
        for p in [src, *src.rglob("*")]:
            os.chmod(p, stat.S_IRWXU)


def test_build_data_with_platform_profile(tmp_path):
    _write_fake_pathmnist(tmp_path / "input" / "ds")
    cfg = load_config(ROOT / "configs/models/b1_cnn.yaml", {
        **platform_overrides("kaggle"), "data.source_dirs": [str(tmp_path / "input")],
        "data.root": str(tmp_path / "temp")})
    data = build_data(cfg)
    assert len(data.arrays["train"]) == 90 and data.size == 28


def test_platform_profiles():
    k = platform_overrides("kaggle")
    assert k["data.source_dirs"] == ["/kaggle/input"] and k["data.download"] is False
    merged = cli_overrides("kaggle", ["data.num_workers=8"])
    assert merged["data.num_workers"] == 8 and "data.root" in merged
    assert platform_overrides(None) == {}


def test_resolve_devices():
    assert resolve_devices("none") == []
    assert resolve_devices("0,1") == ["cuda:0", "cuda:1"]
    assert resolve_devices(["cpu", "cpu"]) == ["cpu", "cpu"]


def test_parallel_jobs_and_failure_isolation(synthetic_cfg, tmp_path):
    cfg = apply_overrides(synthetic_cfg, {"train.epochs": 2})
    bad = apply_overrides(cfg, {"model.params.depth": 9})  # invalid: too deep for 28 px
    jobs = [(apply_overrides(cfg, {"experiment.seed": s}), tmp_path / f"s{s}") for s in (0, 1, 2)]
    jobs.append((bad, tmp_path / "bad"))
    res = run_jobs(jobs, ["cpu", "cpu"])                     # 2 worker processes (spawn)
    assert [r["status"] for r in res[:3]] == ["completed"] * 3
    assert res[3]["status"] == "failed" and "too large" in res[3]["error"]
    assert all((tmp_path / f"s{s}" / "metrics.json").exists() for s in (0, 1, 2))
    # same result as a sequential run with the same seed (process/device independent)
    seq = run_jobs([(apply_overrides(cfg, {"experiment.seed": 0}), tmp_path / "seq")], [])
    assert seq[0]["val"]["accuracy"] == res[0]["val"]["accuracy"]


def test_parallel_suite(tmp_path):
    out = tmp_path / "suite"
    run_suite(ROOT / "configs/suites/smoke.yaml", out, only=["mlp", "cnn"], devices=["cpu", "cpu"],
              extra_overrides={"train.epochs": 1})
    assert len(list(out.glob("*/seed*/metrics.json"))) == 4
    assert run_suite(ROOT / "configs/suites/smoke.yaml", out, only=["mlp", "cnn"], devices=["cpu", "cpu"]) == []


def test_suite_skips_missing_tuned_config(tmp_path):
    import yaml

    suite = {"name": "s", "seeds": [0], "runs": [{"id": "missing", "config": str(tmp_path / "nope.yaml")}]}
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(suite))
    assert run_suite(tmp_path / "s.yaml", tmp_path / "out") == []


def test_zenodo_table_matches_official_medmnist():
    medmnist = pytest.importorskip("medmnist")
    from crc_tissuenet.data import ZENODO_FILES

    info = medmnist.INFO["pathmnist"]
    for size, (url, md5) in ZENODO_FILES.items():
        suffix = "" if size == 28 else f"_{size}"
        assert url == info[f"url{suffix}"] and md5 == info[f"MD5{suffix}"]


def test_download_missing_size_then_cleanup(tmp_path, monkeypatch):
    import crc_tissuenet.data as data

    calls = []

    def fake_download(url, folder, filename, md5):  # stands in for the Zenodo download
        calls.append((url, filename, md5))
        _write_fake_pathmnist(folder, name=filename)

    monkeypatch.setattr(data, "_download_file", fake_download)
    src = tmp_path / "input"
    _write_fake_pathmnist(src / "kaggle-28px")                 # 28 px attached as a dataset
    mm28 = prepare_data(tmp_path / "temp", 28, download=True, source_dirs=[src])
    assert calls == [] and mm28.exists()                        # found locally -> no download
    monkeypatch.setattr(data, "ZENODO_FILES", {**data.ZENODO_FILES, 28: ("https://zenodo.example/x", "abc")})
    mm2 = prepare_data(tmp_path / "temp2", 28, download=True, source_dirs=[])  # nothing local -> download
    assert calls[0][1] == "pathmnist.npz" and calls[0][2] == "abc"
    assert (mm2 / "train_images.npy").exists()
    assert not (tmp_path / "temp2" / "pathmnist.npz").exists()  # downloaded npz deleted after conversion


def test_export_npy_script(tmp_path, monkeypatch):
    """export_npy: converts an attached npz, downloads a missing size, output is used in place."""
    import subprocess
    import sys

    src = tmp_path / "input"
    _write_fake_pathmnist(src / "ds", name="pathmnist.npz")
    out = tmp_path / "working"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/export_npy.py"), "--sizes", "28", "--out", str(out),
                        "--tmp", str(tmp_path / "tmp"), "--source", str(src)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "pathmnist_28" / "train_images.npy").exists()
    # The main notebook then finds the folder and memory-maps it in place (no conversion, no download)
    folder = prepare_data(tmp_path / "temp", 28, download=False, source_dirs=[out])
    assert folder == out / "pathmnist_28" and not (tmp_path / "temp" / "pathmnist_28_mmap").exists()
