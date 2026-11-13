# Methods and design decisions

This document explains *why* the pipeline is built the way it is. Use it when writing the Methods section of the report.

## 1. Data handling

- **Official splits only.** Using the MedMNIST train/val/test splits keeps results comparable with the published benchmark. The test split (CRC-VAL-HE-7K) comes from 50 patients who do not appear in the training data. This is a non-random, cohort-based split (TRIPOD+AI analysis type 2b).
- **Validation is optimistic.** Train and validation are a random 9:1 split of patches from the same 86 slides, so validation scores overestimate performance on new patients. Expect a visible validation–test gap, and report it.
- **Low-RAM storage.** `npz_to_memmap` streams each array out of the official `.npz` zip in 2,048-row chunks into an uncompressed `.npy` file, which is then memory-mapped. RAM stays below about 1 GB even for the 224 px data, which holds roughly 13.5 GB of training pixels.
- **Normalisation.** Scratch models use per-channel mean and standard deviation computed from the training split only. Pretrained models use the statistics their weights were trained with, taken from timm's `pretrained_cfg` (ImageNet statistics for most models; 0.5/0.5 for MLP-Mixer).
- **Preprocessing inside the model.** `ClassifierWrapper` performs resizing and normalisation on the GPU, as part of the model. Training, evaluation and explanation therefore always use identical preprocessing, and SHAP attributions are expressed in raw pixel space.
- **Augmentation (training only).** The options are flips, 90° rotations, colour jitter and H&E stain jitter. Stain jitter converts RGB to optical density, projects onto the Ruifrok–Johnston H&E vectors, applies a random scale and shift per stain, and converts back.

## 2. Leakage control

| Decision | Data used |
|---|---|
| Early stopping, best-epoch checkpoint | validation macro F1 |
| Ablation winners | validation macro F1 |
| Optuna objective | validation macro F1 |
| Transfer-strategy choice (E2) | validation |
| Final numbers, hypothesis tests | **test, once per final run** |

`eval.test` defaults to `false`. The ablation runner forces it off, and a unit test (`test_test_set_not_used_by_default`) checks that test predictions are not even computed by default.

## 3. Training

- **Loss:** cross-entropy, with optional label smoothing and class weights.
- **Optimisers:** biases and normalisation parameters are excluded from weight decay.
- **Learning-rate schedule:** linear warm-up, then cosine or step decay, applied per step.
- **Layer-wise learning-rate decay (transformers and Mixer):** timm's `create_optimizer_v2(layer_decay=0.75)` assigns each layer group a learning rate of `lr × 0.75^(distance from head)`. timm stores this factor as `lr_scale`, which only timm's own schedulers apply. We multiply it into `lr` explicitly so that PyTorch schedulers respect it.
- **Fine-tuning strategies:**
  - S1, linear probe: only the classifier is trained, and BatchNorm layers stay in eval mode so frozen running statistics are not updated.
  - S2, partial: the top 25% of timm layer groups are trained, plus the head.
  - S3, full: all parameters are trained.
- **Mixed precision:** fp16 autocast with gradient scaling.
- **Logged gradient norms:** the global norm and the first-layer norm, for diagnosing vanishing or exploding gradients.
- **Divergence handling:** a non-finite loss stops the run with status `diverged`; the run is reported, not silently dropped.
- **Seeds:** three per final model (42, 43, 44). Deterministic cuDNN is used, and `torch.use_deterministic_algorithms(True, warn_only=True)` is set.

## 4. Statistics

| Quantity | Method |
|---|---|
| Point metrics | accuracy, balanced accuracy, macro F1, macro one-vs-rest AUC, log-loss, ECE (15 bins) |
| 95% CI | percentile bootstrap over test patches, 1,000 resamples |
| Seed variability | mean ± SD over 3 seeds |
| Model A vs B | paired bootstrap of the macro-F1 difference (same resampled patches), plus McNemar's test on paired correctness (exact binomial when discordant pairs < 25) |
| Multiple comparisons | Holm–Bonferroni across all models compared with the reference |

Paired tests use the same seed (42) for both models. A seed ensemble of the best model, averaging probabilities across seeds, is also reported.

**Limitation:** the bootstrap treats patches as independent, but patches from the same patient are correlated and PathMNIST has no patient IDs. The CIs are therefore likely too narrow, and p-values too small. State this explicitly in the report.

**Sample size:** with 7,180 test patches and accuracy near 0.90, the binomial standard error is about 0.0035, so the 95% CI is roughly ±0.7 percentage points (before accounting for clustering).

## 5. Explainability (SO2)

1. **Expected-gradients SHAP** (`shap.GradientExplainer`). The background is 100 random training patches, and attributions are computed on softmax outputs for the predicted class.
2. **Partition SHAP** with a blur masker. This gives region-level attributions that are easier to read histologically.
3. **Stain-space SHAP.** The model is re-parameterised as f(HED) = model(hed_to_rgb(HED)), so attributions fall on the haematoxylin, eosin and residual channels.
   - Expected pattern: LYM and TUM decisions should rely mainly on haematoxylin (nuclear density and morphology).
   - ADI, MUC, MUS and STR should rely more on eosin, or on its absence.
   - This expectation is a hypothesis to test, not a result.
4. **Randomisation sanity check** (Adebayo et al. 2018). The Spearman correlation between attribution maps of the trained model and a weight-randomised copy should be low. A high correlation would mean the maps mostly reflect the input image, not the model.
5. **Grad-CAM** on the last convolution (CNNs only), for comparison.

For the report, examine at least 20 patches per class (proposal SO2). Run `--per-class 10`, which yields up to 10 correct and 10 misclassified patches per class. Record whether highlighted regions match the expected histology: glandular epithelium for TUM, dense small round nuclei for LYM, empty fat-cell lumens for ADI. Have a pathologist confirm these interpretations if possible.

## 6. Reporting checklist mapping (TRIPOD+AI / CLAIM)

| Item | Where |
|---|---|
| Data source, dates, eligibility | README Dataset section; `data/pathmnist_*_summary.json` |
| Data partitions, leakage | §1–2 above; `metrics.json` → `class_counts` |
| Preprocessing, augmentation | §1; `config.yaml` of every run |
| Model specification | `configs/models/*.yaml`; `metrics.json` → `model` |
| Training details, hyperparameter search | `configs/`, `runs/cnn_ablation/ablation_results.csv`, `runs/optuna/trials.csv` |
| Performance with CIs, calibration | `reports/*/REPORT.md`, reliability diagrams |
| Error analysis | confusion matrix, per-class F1, `clinical_errors` in `metrics.json` |
| Explainability | `<run>/explain/` |
| Software and hardware | `metrics.json` → `environment` |
| Code availability | this repository |
| Limitations, intended use | report Discussion; README disclaimer |
