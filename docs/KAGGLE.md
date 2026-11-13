# Running on Kaggle

**Why Kaggle rather than Colab.** Free Colab gives one GPU with usage limits that vary and are not published. Sessions also disconnect when the browser is idle, which is a problem for runs that take many hours.

Kaggle gives a published weekly GPU quota, two GPUs at once (T4 ×2), which this code uses by running one experiment per GPU, and 4 CPUs with ~29 GB RAM instead of 2 CPUs and ~12 GB. Most importantly, *committed* versions run in the background for up to 12 h with the browser closed, and every output is versioned. Colab remains supported (`notebooks/colab_quickstart.ipynb`) for quick interactive work.


Kaggle gives each verified account a weekly GPU quota with **GPU T4 ×2** (two 16 GB GPUs) or **P100** (one 16 GB GPU), 4 vCPUs and about 29 GB RAM. Check your current quota under *Settings → Accelerator*.

This repository uses Kaggle as follows:

| Kaggle constraint | How the code handles it |
|---|---|
| `/kaggle/input` is read-only | Your data are found and validated there and never modified. `.npz` files are converted once to a memory-map cache on scratch disk; folders of `.npy` files are used in place. |
| `/kaggle/working` is saved with the version but capped at 20 GB | Only runs, reports and checkpoints are written there. The ~15 GB 224 px cache goes to `/kaggle/temp` (or `/tmp`, whichever has more space). |
| Two GPUs | `--gpus auto` runs independent experiments in parallel, one per GPU (ablation candidates, seeds, models). For one large model, `-o train.data_parallel=true` splits batches across both GPUs instead. |
| 12 h per session | Committed versions run in the background. Finished runs are skipped, and the notebook restores earlier runs from a previous version's output. |
| Internet may be off | Code can come from a dataset. Pretrained weights can be loaded from a file with `model.weights_file`. |

## 1. Data: nothing to upload

The notebook gets PathMNIST inside Kaggle; nothing passes through your computer:

1. **28 px:** attach the public Kaggle dataset [dongquan/pathmnist-colon-pathology-dataset](https://www.kaggle.com/datasets/dongquan/pathmnist-colon-pathology-dataset) (*Add Input → Datasets*).
2. **224 px (and any other missing size):** with `DOWNLOAD_MISSING = True` (the default), the notebook downloads it directly from the official MedMNIST record on Zenodo (record 10519652). It verifies the MD5 checksum, converts the file to a memory-mapped cache on scratch disk, and deletes the download.

Section 5 of the notebook lists each size with `[OK]` and 89,996 / 10,004 / 7,180 patches if the data are the official files. If the attached 28 px dataset is not, detach it and the notebook downloads the official 28 px file too.

Scratch disk is not saved between versions, so every committed version would repeat the download and the decompression, which takes several minutes and leaves the GPUs idle but counting.

**Recommended: export once.** In a CPU-only notebook (*Accelerator: None*, so no GPU quota is used), run:

```bash
!git clone -q https://github.com/bitsbyte111/pathmnist-crc-classification.git && cd crc-tissuenet && python scripts/export_npy.py --sizes 28 224
```

Commit it with *Save & Run All*. It writes uncompressed arrays (about 16.5 GB) to its output. Attach that output to the main notebook (*Add Input → Your Work*) instead of any PathMNIST dataset. Every version then memory-maps the arrays directly: no download, no decompression.

## 2. Create the notebook

**Option A (simplest):** *Code → New Notebook → File → Import Notebook* and select `notebooks/kaggle_crc_tissuenet.ipynb`. Then:
- *Add Input* → the 28 px PathMNIST dataset (the 224 px file is downloaded automatically).
- *Settings → Accelerator → GPU T4 x2*.
- *Settings → Internet → On* (requires a phone-verified account).
- Edit `REPO_URL` in the first code cell.

**Option B (API):** edit the ids in `kaggle/code/kernel-metadata.json`, then run `kaggle kernels push -p kaggle/code`.

If your GitHub repository is private, either make it public, or zip the repository, upload it as a dataset and set `CODE_DATASET` in the notebook.

## 3. Work through the studies

The notebook is organised as five studies, each stated as hypothesis → experiment → observation:
1. CNN hyperparameters (18 stages, 28 px).
2. Transfer-learning hyperparameters (10 stages, ResNet-50 or DeiT-S at 224 px).
3. A joint Optuna search.
4. The test-cohort benchmark.
5. SHAP explanations.

Turn studies on in the first code cell and commit a version:

| Version | Switches | Approx. runtime on T4 ×2 (estimate) |
|---|---|---|
| v1 | `RUN_CHECKS`, `RUN_CNN_STUDY`, `RUN_B2` | 3–4 h (also builds the data caches) |
| v2 | `RUN_TRANSFER_STUDY`, `RUN_OPTUNA` | 2–4 h |
| v3 | `RUN_BENCHMARK`, `BENCHMARK_ONLY = ["B0_mlp","B1_cnn","B1_cnn_tuned","B2_resnet18","P1_resnet50","P1_resnet50_tuned","P2_densenet121","P3_efficientnet_b0"]` | ≤ 10 h |
| v4 | `RUN_BENCHMARK`, `BENCHMARK_ONLY = ["P4_convnext_tiny","P5_deit_small","P6_swin_tiny","P7_mixer_b16"]` | ≤ 10 h |
| v5 | `RUN_BENCHMARK` (all finished: report only), `RUN_EXPLAIN` | ~1 h |

**A second person doubles the GPU hours.** Sumreen can *Copy & Edit* the notebook and, while you run v1, run the transfer study on the vision transformer: `RUN_TRANSFER_STUDY = True`, `TRANSFER_BASE = "configs/models/p5_deit_small.yaml"`. Attach her output to your later versions. The two transfer studies, CNN vs transformer, then make a direct comparison in the report.

Time the first stage of each study (shown in the log) and adjust the plan. If a version hits the 12 h limit, Kaggle may not keep its output, so keep each version comfortably below it.

Each new version must restore the previous ones: *Add Input → Your Work → this notebook* (newest version), plus any teammate notebooks. Section 4 of the notebook copies their `runs/` and tuned configs back, so earlier runs are skipped.

Interactive sessions are good for checking cells. Use *Save Version → Save & Run All (Commit)* for anything long, because interactive sessions stop when idle.

**What each study writes:**
- `runs/<study>/OBSERVATIONS.md`, with per-stage hypothesis, measurement table, automatic observations and a space for your interpretation;
- `figures/stage_*.png`;
- `tables/*.csv`;
- at the end of every version, `/kaggle/working/OBSERVATION_LOG.md` (all studies so far), `gpu_usage.png` (utilisation of both GPUs over time) and `crc_tissuenet_results.zip`.

## 4. Get the results

The version's *Output* tab contains `crc_tissuenet_results.zip` with every report, observation file, figure and tuned config, but no checkpoints. The full `crc-tissuenet/runs/` folder with checkpoints is there too, for resuming. Copy `reports/*/REPORT.md` and its figures into the GitHub repository; `git add -f` is needed because reports are git-ignored by default.

## Running offline (Internet off)

1. Upload the repository zip as a dataset and set `CODE_DATASET = "/kaggle/input/<name>"`.
2. Pretrained weights must come from files. On a machine with internet, run:
   ```python
   import timm
   m = timm.create_model("resnet50.a1_in1k", pretrained=True)
   ```
   Then upload the cached `model.safetensors` from `~/.cache/huggingface/hub/` as a dataset. Pass it per model, e.g.
   `-o model.weights_file=/kaggle/input/timm-weights/resnet50.safetensors`.
3. Ensure `shap` and `optuna` are available in the Kaggle image (they normally are); `medmnist` is not needed when you supply the data.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No complete PathMNIST 224px data found` | The message lists what *was* found. Check the dataset is attached, and that the file holds `train_images`, `val_images`, `test_images` and the matching `_labels` arrays. |
| `Not enough disk space` | Pass `-o data.root=/tmp/crc_data` (or another location with ~15 GB free), or prepare only 28 px until you need 224 px. |
| CUDA out of memory (224 px) | Lower `train.batch_size` (e.g. `-o train.batch_size=32`), or run one job at a time with `--gpus 0`. |
| A run shows `status: failed` | The error and traceback are in the log. Re-running the same command retries only failed or unfinished runs. |
| GPU utilisation low at 224 px | Data loading is CPU-bound with 4 vCPUs. This is expected with two parallel runs; `-o data.num_workers=3` can help for single runs. |
