# Ablation guide: what each stage teaches and what to look for

The ablation (`configs/ablation/cnn_ablation.yaml`) improves the from-scratch CNN one factor at a time. Each stage is a small experiment about one course concept.

The guide states the theory behind each stage, what to expect, and which plot shows it. When you write `ABLATION_REPORT.md`, replace each "Interpretation (fill in)" line with what you actually observed. If a result contradicts the expectation, say so and explain why; a contradiction is a finding, not a failure.

Each stage writes a figure `runs/cnn_ablation/figures/stage_<name>.png` with three panels: (1) mean validation macro F1 per candidate, with the winner in red; (2) validation macro F1 per epoch; (3) first-layer gradient norm per epoch (log scale).

## How the search works and its limits

The search is greedy coordinate descent: every stage keeps the winners of all earlier stages fixed. This keeps each comparison clean, but it can miss interactions. For example, the best learning rate depends on the batch size, and batch norm changes which learning rates are stable.

Stage order (optimisation → architecture → regularisation) limits the damage. After the ablation, `scripts/tune.py` searches learning rate, weight decay, batch size, dropout and label smoothing jointly with Optuna to probe interactions. Report both results.

Differences smaller than about 0.005 macro F1 from a single seed are often noise. If two candidates are that close, rerun the stage with `--seeds 42 43 44 --stages <name>` before drawing conclusions. Ties go to the smaller model.

All ablation runs use 20 epochs with early stopping (patience 5). The winning configuration is then retrained for up to 50 epochs and evaluated on the test set in the benchmark suite.

---

## Optimisation

### 1. `optimizer`: gradient-descent variants

| Optimiser | Update idea |
|---|---|
| SGD | `w -= lr * g`. One global step size; noisy but often generalises well. |
| Momentum | Accumulates a velocity `v = mu*v + g`; damps oscillations in ravines. |
| Nesterov | Evaluates the gradient at the look-ahead point; slightly better-behaved momentum. |
| Adagrad | Per-parameter step `lr / sqrt(sum g^2)`; the step keeps shrinking, so learning can stall. |
| Adadelta | Adagrad with a decaying window; nearly learning-rate-free. |
| RMSprop | Exponential average of `g^2` fixes Adagrad's shrinking steps. |
| Adam | Momentum + RMSprop + bias correction. Weight decay is coupled L2 (it gets rescaled by the adaptive denominator). |
| AdamW | Adam with decoupled weight decay (Loshchilov & Hutter). |
| NAdam | Adam with Nesterov momentum. |

**Expect:** adaptive methods converge fastest in early epochs. Well-tuned SGD with momentum can catch up or generalise slightly better. Plain SGD and Adagrad usually lag at a fixed epoch budget.

**Note:** each optimiser gets its own conventional learning rate; the next stage then tunes it. Comparing all optimisers at one learning rate would be unfair.

### 2. `learning_rate`: step size (×0.1 … ×10 of the stage-1 value)

**Expect:** a U-shaped curve. If the rate is too low, validation F1 is still rising when training stops. If it is too high, loss oscillates or the run ends with status `diverged`. Look at the learning curves, not only the final bars.

### 3. `batch_size`: 32 to 512

Small batches give noisier gradient estimates, which act as implicit regularisation, and more updates per epoch. Large batches are faster per epoch on a GPU but often generalise slightly worse at a fixed learning rate. This is the "generalisation gap" discussed by Keskar et al.

**Note:** the learning rate is kept fixed here. The linear scaling rule (scale the learning rate with the batch size) is a common alternative. Mention this interaction in your discussion and probe it in the Optuna stage.

Record training time per run (`train_time_s` in the CSV); it is part of the trade-off.

### 4. `lr_schedule`: constant vs step vs cosine (1 warm-up epoch)

**Expect:** decaying schedules finish in a sharper minimum, so the final epochs improve noticeably. A constant learning rate plateaus with more epoch-to-epoch noise.

---

## Architecture

### 5. `activation`

| Activation | Property |
|---|---|
| Sigmoid | Saturates; derivative ≤ 0.25, so gradients shrink through each layer (vanishing gradient). Output is not zero-centred. |
| Tanh | Zero-centred but still saturating (derivative ≤ 1). |
| ReLU | No saturation for x > 0 and cheap to compute; units can "die" if stuck at x < 0. |
| Leaky ReLU / ELU | Allow a small negative signal, which avoids dead units. |
| GELU / SiLU (Swish) / Mish | Smooth, non-monotonic; standard in modern networks. |

**The key plot:** the third panel shows the **first-layer gradient norm**. With sigmoid it is typically several times smaller from the first epoch, which is the vanishing-gradient problem made visible. In our synthetic test it was about 5× smaller than with ReLU.

Batch norm partly rescues saturating activations by keeping pre-activations near zero. Note this when interpreting the results, and optionally rerun sigmoid with batch norm off.

### 6. `initialization`

| Scheme | Weight variance |
|---|---|
| He (Kaiming) | `2 / fan_in`. Preserves activation variance through ReLU layers. |
| Xavier (Glorot) | `2 / (fan_in + fan_out)`. Designed for tanh/sigmoid. |
| `normal_0.01` | Naive small weights; deliberately bad. |
| PyTorch default | Kaiming-uniform with a = √5. |

**Expect:** with batch norm, the differences are small because BN re-normalises activations. This is a useful lesson in itself. Without BN, `normal_0.01` trains much more slowly. To see this effect, you can run the stage with `-o model.params.batch_norm=false`.

### 7. `kernel_size` (filter size): 3×3, 5×5, 7×7

A larger kernel sees more context per layer but has more parameters: (k² × C_in × C_out). Two stacked 3×3 layers cover the same 5×5 receptive field with fewer parameters and an extra non-linearity (see stage 10).

At 28 px, a 112 µm patch is compressed into very few pixels. Each pixel covers about 4 µm, roughly the size of a nucleus, so local texture dominates.

### 8. `width`: number of filters (16-32-64 / 32-64-128 / 64-128-256)

Capacity grows roughly with width². Look for over-fitting: train accuracy keeps rising while validation macro F1 flattens or the validation loss increases.

### 9. `depth`: 2 / 3 / 4 blocks

Each block halves the resolution (28 → 14 → 7 → 3 → 1). Deeper networks have a larger receptive field and more abstract features. With depth 4, the last block works on 3×3 maps, so returns diminish.

### 10. `convs_per_block`: 1 vs 2 (the VGG idea)

Stacking two convolutions per block adds depth and non-linearity without more downsampling.

### 11. `downsampling`: max pool vs average pool vs strided convolution

Max pooling keeps the strongest response, which suits detecting nuclei and edges. Average pooling keeps overall texture statistics. Strided convolution learns how to downsample.

### 12. `classifier_head`: global average pooling vs flatten + dense

Global average pooling (GAP) has far fewer parameters and is translation-invariant, a good fit for texture-like tissue. Flatten + dense keeps spatial layout but adds parameters and over-fitting risk. Compare the `params` column.

---

## Regularisation

### 13. `batch_norm`

Batch norm normalises each layer's inputs per mini-batch. Training is faster and more stable, larger learning rates become usable, and there is mild regularisation.

**Expect:** removing BN slows convergence. At the learning rate tuned with BN, training may even become unstable, which is a clear example of a factor interaction.

### 14. `dropout`: 0 / 0.25 / 0.5 before the classifier

**Expect:** with about 90,000 training patches and a small model, the effect is small. Heavy dropout can under-fit.

### 15. `weight_decay`: 0 to 1e-3

Weight decay is an L2 penalty that shrinks weights. With AdamW the decay is decoupled from the adaptive step; with Adam it is coupled (see stage 1).

### 16. `augmentation`

| Candidate | What changes |
|---|---|
| None | No augmentation. |
| Geometric | Flips and 90° rotations; tissue has no canonical orientation, so these are label-preserving. |
| Colour | Brightness, contrast and saturation jitter. |
| H&E light | Random scaling and shifting in haematoxylin-eosin optical-density space, simulating staining differences between laboratories (Tellez et al. 2019). |

**Important caveat:** validation patches come from the **same slides** as the training data. Colour augmentation mainly helps on **new** patients and labs, so its benefit may not appear in validation scores. That is why experiment E4 tests augmentation on the independent test cohort. Discuss this; it is a central point about domain shift.

### 17. `label_smoothing`: 0 vs 0.1

Label smoothing replaces one-hot targets with 0.9 / 0.0125 and discourages over-confidence. **Look at log-loss and ECE**, not just F1. Smoothing often barely changes accuracy but improves calibration.

### 18. `class_weighting`

Classes in PathMNIST are only roughly balanced. Weighted cross-entropy up-weights rarer classes. It may raise macro F1 or balanced accuracy slightly while lowering overall accuracy.

---

## Study 2: transfer-learning hyperparameters (`configs/ablation/transfer_ablation.yaml`)

The same machinery is applied to an ImageNet-pretrained ResNet-50, or DeiT-Small with `--base`, at 224 px. Each setting trains on a stratified 25 % of the training split for 6 epochs and is judged on the full validation split. The point is relative comparison, so the cheaper budget is fair as long as every setting gets the same one.

| Stage | Question | Concept |
|---|---|---|
| finetune_strategy | Retrain the head only, the top quarter, or everything? | How far ImageNet features transfer to H&E tissue |
| optimizer | AdamW vs momentum/Nesterov SGD | Optimiser behaviour on pretrained weights |
| learning_rate | ×0.1 to ×10 around 1e-4 | Large steps destroy transferred features (catastrophic forgetting) |
| batch_size | 32 / 64 / 128 | Memory, throughput, gradient noise at 224 px |
| warmup | 0 vs 1 epoch | Protecting pretrained weights from noisy first updates |
| input_normalization | ImageNet vs PathMNIST statistics | The input distribution a pretrained network expects |
| dropout_head | 0 / 0.1 / 0.3 | Regularising the new classifier |
| weight_decay | 0 / 0.01 / 0.05 | Decay pulls weights away from the pretrained solution |
| augmentation | geometric / + colour / + H&E stain | Domain shift between slides and labs |
| label_smoothing | 0 vs 0.1 | Calibration of a large network |

Running the study on both ResNet-50 and DeiT-Small answers a question worth a paragraph in the report: do CNNs and transformers prefer different fine-tuning settings, especially learning rate and warm-up?

## Automatic observations

`OBSERVATIONS.md` is written by `crc_tissuenet.observations` from every run's per-epoch history. For each stage it reports:
- the ranking, and whether the top two differ by more than single-seed noise (~0.005 F1);
- how much the hyperparameter matters (the spread between best and worst setting);
- convergence speed among competitive settings;
- over-fitting signals (train–validation gap, rising validation loss);
- gradient-flow differences (first-layer gradient norm in epoch 1);
- cost (time and parameters) against the F1 gained.

These are measurements, not explanations. The *Interpretation* line under each stage is yours to write, linking what was measured to the concept. That writing is where the course concepts are demonstrated.

## Writing it up (suggested structure per stage)

1. **Hypothesis:** one sentence from the theory above.
2. **Result:** the table row(s) and figure.
3. **Interpretation:** why the theory did or did not hold here. Consider the image size, batch norm, the epoch budget and noise.
4. **Decision:** what was carried forward, and the cost (parameters, time).

The `progress.png` figure shows how validation macro F1 rose through the stages, which makes a good summary for the final presentation.
