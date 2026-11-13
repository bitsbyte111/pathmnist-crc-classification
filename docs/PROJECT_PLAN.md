# Project plan (5 weeks)

**Roles.** The split below is a suggestion; swap tasks freely. Each person owns some experiments end to end, and each reviews the other's results before they go in the report.

- **Israr: optimisation and benchmark lead.** Owns the data pipeline, the CNN ablation (optimisation and architecture stages), the scratch baselines, the CNN pretrained models (P1–P4), and the results and statistics sections.
- **Sumreen: transfer learning and explainability lead.** Owns the transfer strategies (E2), the transformer and Mixer models (P5–P7), colour augmentation (E4), SHAP and error analysis, and the clinical interpretation and discussion.

Both share the regularisation stages of the ablation, the resolution study, and the final report and presentation.

## GPU budget (estimates; check them against your first runs)

| Block | Runs | Rough cost per run | Total |
|---|---|---|---|
| B2 reproduction (100 epochs, 28 px) | 1 | 1–2 h | 1–2 h |
| CNN ablation (18 stages, about 45 unique runs after caching, 28 px) | ~45 | 2–5 min | 2–4 h |
| E2 transfer strategies (10 epochs, 224 px) | 9 | 20–60 min | 4–8 h |
| E1/E3 benchmark: P1–P7 × 3 seeds (224 px) | 21 | 30–90 min (proposal estimate) | 15–30 h |
| E1/E3 baselines B0/B1/B1-tuned × 3 seeds | 9 | minutes | < 1 h |
| E4 colour augmentation (4 × 3 seeds) | 12 | 30–90 min | 6–15 h |
| E5 resolution (8 runs) | 8 | 5–60 min | 3–5 h |

Kaggle gives each account a weekly GPU quota, and Colab's free tier is less predictable, so check the current limits and time the first 224 px epoch before committing to this plan.

If time is short, in this order: run E1/E3 with 1 seed for P6/P7 first and add seeds later; reduce E4 to 1 seed; treat E5 as optional (SO1 is secondary).

Always save runs to Google Drive (see the notebook) so finished runs are never repeated.

## Week 1: setup, data, pipeline validation

- [ ] Create the GitHub repository and push this code; check that CI passes (both).
- [ ] Run `make smoke` locally or in Colab (both).
- [ ] Download data at 28 px and 224 px; record class counts per split (Israr).
- [ ] Exploratory data analysis: sample patches per class, colour histograms of train vs test (both).
- [ ] **Reproduce B2** (MedMNIST v2 ResNet-18, 28 px). The target is test accuracy ≈ 0.907. This validates the pipeline (Israr).
- [ ] Time one 224 px epoch of ResNet-50 and DeiT-S, then update the GPU budget (Sumreen).
- [ ] Ask the AKU Ethics Review Committee to confirm the exemption (both).

**Milestone:** the B2 reproduction is within about 1 percentage point of the published value, or there is a written explanation of the gap.

## Week 2: CNN ablation (the base model)

- [ ] Optional quick pass first: `run_ablation.py -o data.subset_fraction=0.2` to catch problems early.
- [ ] Full ablation, optimisation stages 1–4 (Israr).
- [ ] Architecture stages 5–12 (Israr), regularisation stages 13–18 (Sumreen). Run them sequentially; the runner resumes automatically.
- [ ] Rerun with 3 seeds any stage whose top two candidates are within 0.005 F1.
- [ ] Optional: Optuna joint search around the winner (`scripts/tune.py`).
- [ ] Write the interpretation for each stage, using `docs/ABLATION_GUIDE.md` (both).

**Milestone:** `configs/models/b1_cnn_tuned.yaml` is committed, and `ABLATION_REPORT.md` is drafted.

## Week 3: transfer learning

- [ ] E2 strategies on ResNet-50, DeiT-S and MLP-Mixer (Sumreen). Choose the best strategy per family on validation data and update the P-configs if it is not full fine-tuning.
- [ ] Start E1/E3: baselines, then P1–P4 (Israr) and P5–P7 (Sumreen) with `--only`.

**Milestone:** E2 report done; at least one seed of every model is finished.

## Week 4: benchmark, augmentation, explainability

- [ ] Finish all seeds of E1/E3 and run `scripts/report.py` (Israr). Check PO1 (H1) and PO2 (H2a, H2b).
- [ ] E4 colour/stain augmentation on the best model (Sumreen).
- [ ] SHAP on the best overall model **and** on the tuned CNN: pixel SHAP, stain-space SHAP, partition SHAP, sanity check, Grad-CAM (Sumreen).
- [ ] Error analysis: confusion pairs, TUM → NORM/STR errors, at least 20 patches per class reviewed (both).

**Milestone:** final benchmark table and SHAP figures.

## Week 5: secondary objectives, write-up

- [ ] E5 resolution study, if time permits (Israr).
- [ ] Report written following the TRIPOD+AI and CLAIM checklists (`docs/METHODS.md` §6) (both).
- [ ] Discussion points:
  - the validation–test gap
  - patch clustering and the CI limitation
  - whether results from German labs transfer to Pakistan
  - cost versus accuracy for low-resource labs
- [ ] Fill the README results table, commit the final reports and figures (`git add -f`), and tag release `v1.0` (both).
- [ ] Presentation slides and a demo.

## Definition of done

- Every number in the report can be regenerated from a config plus a seed.
- The test set was used only for the final runs listed in the suites.
- CI is green, and the README results table and figures match `reports/`.
