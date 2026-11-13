# CRC-TissueNet

**Benchmarking CNNs, vision transformers and MLP models for colorectal cancer tissue classification in H&E histopathology (PathMNIST)**

[![CI](https://github.com/bitsbyte111/pathmnist-crc-classification/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Data](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)

Capstone project, MS Health Data Science, Aga Khan University (AKU), Karachi.
Israr Hussain and Sumreen Fatima. Supervisors: Dr. Zarmin, Dr. Sumaira.

> **Research and education only.** These models are not medical devices and must not be used for diagnosis.

---

## Overview

Each H&E-stained colorectal tissue patch is classified into one of nine tissue types:
adipose, background, debris, lymphocytes, mucus, smooth muscle, normal mucosa, cancer-associated stroma and tumour epithelium.
This patch-level tissue map underlies computational-pathology tools such as tumour detection and the deep stroma score.

The project has three parts:

1. **Building a strong from-scratch CNN.** A staged ablation of 18 design factors, one at a time. Factors include the optimiser, learning rate, batch size, activation, initialisation, filter size, number of filters, depth, pooling, batch norm, dropout, weight decay and augmentation. Every stage is explained with learning curves and gradient-norm plots.
2. **Benchmarking transfer learning.** ImageNet-pretrained CNNs (ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt-T), vision transformers (DeiT-S, Swin-T) and MLP-Mixer are compared on the patient-independent test set. Metrics use bootstrap confidence intervals, and paired tests use Holm correction.
3. **Explaining the best model with SHAP.** Pixel-level expected-gradient SHAP and region-level partition SHAP are used. A stain-space variant attributes each decision to haematoxylin (nuclei) versus eosin (cytoplasm/stroma). A weight-randomisation sanity check confirms the explanations reflect what the model learned.

## Highlights

| | |
|---|---|
| **Leakage control** | Model selection, ablation and tuning use the validation split only. The test cohort (50 unseen patients) is evaluated once per final run. Unit tests enforce this. |
| **Reproducible** | Fixed seeds, deterministic kernels and a resolved `config.yaml` saved with every run. Environment and library versions are recorded in `metrics.json`. The same seed gives identical results (tested). |
| **Resumable** | Finished runs are skipped. Per-epoch `last.pt` checkpoints survive Colab disconnects. Suites can be split across teammates with `--only`. |
| **Multi-GPU** | `--gpus auto` runs one experiment per GPU (e.g. Kaggle T4 ×2), with results identical to sequential runs (tested). |
| **Runs on free GPUs** | 224 px data (~14 GB) are streamed from a memory map, so RAM use stays low. Augmentation and resizing run on the GPU. Mixed precision is used throughout. |
| **Modular** | Every model, optimiser, schedule and augmentation is chosen from YAML. Configs inherit with `_base_`. Any key can be overridden on the command line. |
| **Tested** | 59 unit and end-to-end tests run on synthetic data. GitHub Actions CI runs tests, lint and a smoke suite on CPU. |

## Hyperparameter studies and observations

The project's main deliverable is a set of controlled experiments in which each hyperparameter is varied on its own, measured on validation data, and written up.

- **Study 1, CNN from scratch** (`configs/ablation/cnn_ablation.yaml`): 18 stages of optimisation, architecture and regularisation choices.
- **Study 2, transfer learning** (`configs/ablation/transfer_ablation.yaml`): 10 stages of fine-tuning choices for ResNet-50 or DeiT-S.
- **Study 3, joint Optuna search** (`scripts/tune.py`).

Each study writes `OBSERVATIONS.md`. For every stage it contains the hypothesis from theory, a measurement table, automatically computed observations and space for your interpretation. The measurements are validation macro F1, convergence epoch, train–validation gap, first-layer gradient norm, time and parameters; the observations cover ranking versus noise, convergence, over-fitting, gradient flow and cost.

## Objectives → experiments → code

| Objective (proposal §5) | Experiment | Command | Output |
|---|---|---|---|
| Strong from-scratch baseline | Study 1: CNN hyperparameters (18 stages) | `scripts/run_ablation.py` | `runs/cnn_ablation/OBSERVATIONS.md`, `configs/models/b1_cnn_tuned.yaml` |
| Transfer-learning settings | Study 2: fine-tuning hyperparameters (10 stages) | `run_ablation.py --plan configs/ablation/transfer_ablation.yaml` | `runs/transfer_ablation/OBSERVATIONS.md`, `configs/models/p1_resnet50_tuned.yaml` |
| (optional) joint tuning | Optuna TPE + median pruning | `scripts/tune.py` | `runs/optuna/` |
| **PO2**: transfer-learning strategies | E2 (S1 linear probe / S2 partial / S3 full) | `run_suite.py --suite configs/suites/e2_transfer.yaml` | `reports/e2_transfer/` |
| **PO1, PO2**: pretrained vs scratch, model families | E1 + E3 (B0–B2, P1–P7, 3 seeds) | `run_suite.py --suite configs/suites/e1_e3_benchmark.yaml` | `reports/e1_e3_benchmark/REPORT.md` |
| PO1: colour / stain augmentation | E4 | `configs/suites/e4_color_aug.yaml` | `reports/e4_color_aug/` |
| **SO1**: resolution vs cost | E5 (28 / 64 / 128 / 224 px) | `configs/suites/e5_resolution.yaml` | `reports/e5_resolution/` |
| **SO2**: errors and interpretability | SHAP, stain SHAP, Grad-CAM, clinical-error table | `scripts/explain.py` | `<run>/explain/` |

## Dataset

[PathMNIST](https://medmnist.com/) is part of MedMNIST v2 / MedMNIST+ and is licensed CC BY 4.0. It is derived from NCT-CRC-HE-100K and CRC-VAL-HE-7K (Kather et al.).

| Split | Patches | Source |
|---|---|---|
| Train | 89,996 | NCT-CRC-HE-100K (86 slides), 90% |
| Validation | 10,004 | NCT-CRC-HE-100K, 10% (same slides as train, so optimistic) |
| Test | 7,180 | CRC-VAL-HE-7K, **50 different patients** |

The images are available at 28, 64, 128 and 224 px. The data are not included in the repository. `scripts/prepare_data.py` uses files you already have (e.g. an attached Kaggle dataset) and downloads any missing size from the official MedMNIST Zenodo record, verifying its checksum.

## Quick start

### Google Colab / Kaggle (recommended, free GPU)

Open [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb). It mounts Google Drive so results survive disconnects, installs the package, downloads the data and runs each phase.

### Kaggle (2× T4, recommended for the 224 px models)

Upload your PathMNIST files as a Kaggle dataset, import [`notebooks/kaggle_crc_tissuenet.ipynb`](notebooks/kaggle_crc_tissuenet.ipynb), and select **GPU T4 x2** with Internet on. Independent runs are spread one per GPU.

```bash
python scripts/prepare_data.py --platform kaggle --sizes 28 224                 # find + validate your data
python scripts/run_ablation.py --plan configs/ablation/cnn_ablation.yaml --platform kaggle --gpus auto
python scripts/run_suite.py --suite configs/suites/e1_e3_benchmark.yaml --platform kaggle --gpus auto
```

See [docs/KAGGLE.md](docs/KAGGLE.md) for uploading data, committed background runs, resuming across sessions and offline use.

### Local

```bash
git clone https://github.com/bitsbyte111/pathmnist-crc-classification.git && cd crc-tissuenet
pip install -e ".[tune,dev]"        # or: pip install -r requirements.txt
pytest -q                           # 59 tests, synthetic data, CPU
make smoke                          # end-to-end pipeline check (~1 min, no download)

python scripts/prepare_data.py --sizes 28 224 --download   # or --source /path/to/your/files
```

### The workflow

```bash
# Phase 1: reproduce the published MedMNIST v2 ResNet-18 result (pipeline validation; acc ~0.907)
python scripts/train.py --config configs/models/b2_resnet18.yaml -o eval.test=true

# Phase 2: CNN ablation; writes configs/models/b1_cnn_tuned.yaml
python scripts/run_ablation.py --plan configs/ablation/cnn_ablation.yaml
python scripts/run_ablation.py --plan configs/ablation/cnn_ablation.yaml -o data.subset_fraction=0.2   # quick dry pass

# Phase 3: transfer-learning strategies (validation only)
python scripts/run_suite.py --suite configs/suites/e2_transfer.yaml
python scripts/report.py   --suite configs/suites/e2_transfer.yaml

# Phase 4: final benchmark on the test cohort (split work between teammates with --only)
python scripts/run_suite.py --suite configs/suites/e1_e3_benchmark.yaml --only B0_mlp B1_cnn B1_cnn_tuned
python scripts/run_suite.py --suite configs/suites/e1_e3_benchmark.yaml --only P1_resnet50 P5_deit_small
python scripts/report.py   --suite configs/suites/e1_e3_benchmark.yaml

# Phase 5: explain the best model
python scripts/explain.py --run runs/e1_e3_benchmark/<BEST_ID>/seed42                  # pixel SHAP + sanity check + Grad-CAM
python scripts/explain.py --run runs/e1_e3_benchmark/<BEST_ID>/seed42 --stain-space    # haematoxylin / eosin SHAP
python scripts/explain.py --run runs/e1_e3_benchmark/<BEST_ID>/seed42 --method partition
```

Any config value can be overridden from the command line:

```bash
python scripts/train.py --config configs/models/b1_cnn.yaml -o train.optimizer=nesterov train.lr=0.02 model.params.kernel_size=5
```

## Repository structure

```
crc-tissuenet/
├── configs/
│   ├── default.yaml              # every setting, documented
│   ├── models/                   # B0 MLP, B1 CNN, B2 ResNet-18, P1–P7 pretrained (+ shared bases)
│   ├── ablation/cnn_ablation.yaml   # the 18-stage ablation plan
│   ├── suites/                   # E1–E5 experiment suites + CI smoke suite
│   ├── platforms/                # kaggle / colab / local profiles (--platform)
│   └── tuning/optuna_cnn.yaml
├── src/crc_tissuenet/
│   ├── data.py                   # download, streaming npz→memmap, loaders, synthetic data
│   ├── augment.py                # GPU flips/rot90, colour jitter, H&E stain jitter
│   ├── models/                   # MLP, ConfigurableCNN, ResNet-18, timm wrapper, fine-tuning strategies
│   ├── optim.py                  # 10 optimisers, warm-up + cosine/step, layer-wise LR decay
│   ├── engine.py                 # Trainer: AMP, grad-norm logging, early stopping, resume, evaluation
│   ├── ablation.py               # staged greedy ablation with result caching
│   ├── metrics.py                # metrics, bootstrap CIs, paired bootstrap, McNemar, Holm, ECE
│   ├── suite.py                  # multi-seed suites and statistical comparison
│   ├── parallel.py               # one experiment per GPU (process pool)
│   ├── explain/                  # SHAP (gradient, partition, stain space, sanity check), Grad-CAM
│   └── viz.py                    # all figures
├── scripts/                      # command-line entry points
├── notebooks/                    # kaggle_crc_tissuenet.ipynb, colab_quickstart.ipynb
├── kaggle/                       # Kaggle API metadata (dataset + notebook)
├── docs/
│   ├── METHODS.md                # design decisions, leakage control, statistics
│   ├── ABLATION_GUIDE.md         # concept behind each ablation stage + what to look for
│   ├── KAGGLE.md                 # running on Kaggle step by step
│   └── PROJECT_PLAN.md           # 5-week plan, roles, GPU budget, checklists
└── tests/                        # pytest suite (synthetic data)
```

Each run writes a self-contained folder:

```
runs/<suite>/<id>/seed42/
  config.yaml  train.log  history.csv  metrics.json  predictions.npz  best.pt  learning_curves.png
```

## Models

| ID | Model | Input | Params | Training |
|---|---|---|---|---|
| B0 | MLP (2,352 → 512 → 128 → 9) | 28 px | 1.27 M | scratch |
| B1 | Small CNN (32-64-128), then the **ablation-tuned** version | 28 px | ~0.1 M (default) | scratch |
| B2 | ResNet-18, MedMNIST v2 protocol | 28 px | 11.2 M | scratch |
| P1–P4 | ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt-T | 224 px | 25.6 / 8.0 / 5.3 / 28.6 M | ImageNet → fine-tune |
| P5–P6 | DeiT-S, Swin-T | 224 px | 22.1 / 28.3 M | ImageNet → fine-tune (LLRD 0.75) |
| P7 | MLP-Mixer-B/16 | 224 px | 59.9 M | ImageNet-21k → fine-tune (LLRD 0.75) |

## Results

> Fill these tables in from `runs/cnn_ablation/ABLATION_REPORT.md` and `reports/e1_e3_benchmark/REPORT.md` after running.
> Report all numbers as mean ± SD over seeds with 95% CIs.

**Published reference (MedMNIST v2, PathMNIST test set):** ResNet-50 (28 px) accuracy 0.911, AUC 0.990 is the PO1 target. ResNet-18 (28 px) accuracy 0.907, AUC 0.983 is the B2 reproduction target.

| Model | Test accuracy | Macro F1 | Macro AUC | Balanced acc | TUM recall | Params | ms / image |
|---|---|---|---|---|---|---|---|
| B1 CNN (default) | | | | | | | |
| B1 CNN (tuned) | | | | | | | |
| best pretrained CNN | | | | | | | |
| best transformer | | | | | | | |

## Evaluation and statistics

- **Metrics:** accuracy, macro AUC (the MedMNIST standard), macro F1 and balanced accuracy (used for model selection), per-class precision/recall/F1, confusion matrix, expected calibration error, parameter count, training time and inference time.
- **Uncertainty:** 1,000-sample percentile bootstrap 95% CIs over test patches. Mean ± SD over 3 seeds.
- **Comparisons:** paired bootstrap of the macro-F1 difference and McNemar's test against the tuned scratch CNN, with Holm-adjusted p-values.
- **Clinical errors:** tumour recall, and tumour patches misclassified as normal mucosa or stroma.
- **Known limitation:** PathMNIST has no patient IDs, so patch-level bootstrap CIs ignore within-patient clustering and are likely too narrow.

See [docs/METHODS.md](docs/METHODS.md) for details.

## Reporting guidelines

Results are reported following TRIPOD+AI and CLAIM. [docs/METHODS.md](docs/METHODS.md) maps each repository artefact to the relevant checklist items.

## Citation

If you use this code, cite this repository and the dataset papers:

```bibtex
@article{yang2023medmnist,
  title   = {MedMNIST v2 -- A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author  = {Yang, Jiancheng and Shi, Rui and Wei, Donglai and others},
  journal = {Scientific Data}, volume = {10}, number = {1}, pages = {41}, year = {2023}
}
@article{kather2019predicting,
  title   = {Predicting survival from colorectal cancer histology slides using deep learning: a retrospective multicenter study},
  author  = {Kather, Jakob Nikolas and Krisam, Johannes and Charoentong, Pornpimol and others},
  journal = {PLoS Medicine}, volume = {16}, number = {1}, pages = {e1002730}, year = {2019}
}
```

## Acknowledgements

This project builds on standard open-source tools: [MedMNIST](https://github.com/MedMNIST/MedMNIST) (data and the official benchmark protocol), [timm](https://github.com/huggingface/pytorch-image-models) (pretrained models, layer-wise LR decay), [SHAP](https://github.com/shap/shap), [Optuna](https://github.com/optuna/optuna), PyTorch and scikit-learn.

## License

The code is released under the MIT license. The data are licensed CC BY 4.0 by their creators.
