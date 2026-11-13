import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def synthetic_cfg(tmp_path):
    from crc_tissuenet.config import load_config

    return load_config(ROOT / "configs/models/b1_cnn.yaml", {
        "data.synthetic": True, "data.synthetic_samples": 256, "train.epochs": 8, "train.warmup_epochs": 0,
        "experiment.output_dir": str(tmp_path), "experiment.device": "cpu", "eval.bootstrap": 20,
    })
