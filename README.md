# 🧠 RSNA Intracranial Aneurysm Detection

This repo implements a **2.5D deep learning pipeline** for detecting intracranial aneurysms from head CTA scans.
It was created for the **RSNA Intracranial Aneurysm Detection** challenge, where the goal is to classify multiple arterial segments for aneurysm presence. The workflow uses **cached 3D DICOM series**, **vessel localization augmentations**, and a **MaxViT backbone** for efficient training and inference.
All experiments were executed in a reproducible Kaggle-compatible environment with optional offline inference.

---

## Project Overview

Intracranial aneurysm detection is formulated as a **multi-label classification** task.
Each CTA series is processed into a fixed number of 2D slices, optionally augmented with **localizer-based crops** derived from radiological metadata. The final model predicts aneurysm presence across 13 arterial segments plus a global *“Aneurysm Present”* label.

The pipeline includes:

1. **DICOM series preprocessing** -> fixed 2.5D NumPy caches
2. **MaxViT-based classifier training** (32 slices + localizer crops as channels)
3. **Offline inference with ensemble support** (multi-seed, multi-fold)

---

## Notebooks

### `segmentation_eda.ipynb`

* Exploratory data analysis of the RSNA dataset.
* Visualizes representative DICOM slices, arterial segment locations, and metadata distributions.
* Explains how slice thickness, spacing, and acquisition planes influence model input design.

### `rsna2025-32ch-img-maxvit-base-tf-384.ipynb`

* Main **training notebook** for the 2.5D MaxViT model.
* Implements dataset assembly, caching, and training loop with per-epoch seeding.
* Supports:

  * Localizer-based auxiliary crops (`max_localizer_crops`)
  * Anti-leakage stochastic dropout (`p_localizer_dropout`)
  * Full-series global off-switch (`p_global_localizer_off`)
* Tracks training and validation AUCs, including weighted AUC (`wAUC`) over 14 classes.
* Saves checkpoints under structured directories:

  ```
  outputs/
  ├── maxvitbasemodel_seed42_fold0/
  │   ├── best_ema.pth
  │   └── metrics.json
  ```

### `rsna2025-32ch-img-maxvit-base-tf-384-inference.ipynb`

* **Offline inference pipeline** used for Kaggle submission testing.
* Loads cached shards and reconstructs models from multiple seeds or folds.
* Automatically adapts to correct `in_chans` from checkpoints.
* Reproduces training-time pre/postprocessing (minus augmentation) to ensure inference consistency.
* Supports ensemble averaging across all discovered checkpoints.

---

## Model Details

* **Architecture:** `timm` MaxViT-B/TF-384
* **Input:** 32 axial slices + up to 3 localizer crops = 35 channels total
* **Resolution:** 384×384
* **Loss:** BCE with label smoothing (0.02)
* **Optimizer:** AdamW
* **Scheduler:** Cosine LR with warmup
* **Precision:** Mixed AMP (fp16)

---

## Running Locally or on Kaggle

1. **Preprocess & cache series:**

   ```python
   from preproc import DICOMPreprocessorKaggle
   pre = DICOMPreprocessorKaggle(target_shape=(32, 384, 384))
   vol = pre.process_series("path/to/series")
   ```

2. **Train model:**

   ```python
   !python rsna2025-32ch-img-maxvit-base-tf-384.ipynb
   ```

3. **Run inference:**

   ```python
   !python rsna2025-32ch-img-maxvit-base-tf-384-inference.ipynb
   ```

4. **Evaluate / submit:**
   Predictions are formatted as a multi-label CSV for Kaggle submission.

---

## Configuration Highlights

| Parameter                | Default              | Description                          |
| ------------------------ | -------------------- | ------------------------------------ |
| `base_slices`            | 32                   | Axial slices per volume              |
| `max_localizer_crops`    | 3                    | Number of localizer-derived crops    |
| `p_localizer_dropout`    | 0.30                 | Randomly drop localizers per epoch   |
| `p_global_localizer_off` | 0.10                 | Randomly disable localizers entirely |
| `img_size`               | 384                  | Target resize                        |
| `in_chans`               | auto (32 + K)        | Computed dynamically                 |
| `model_name`             | `maxvit_base_tf_384` | Backbone                             |
| `epochs`                 | 34                   | Training epochs                      |
| `batch_size`             | 2                    | Per-GPU batch size                   |

---

## Dependencies

* Python ≥ 3.10
* PyTorch ≥ 2.2
* timm ≥ 1.0
* pydicom, opencv-python, numpy, pandas, polars
* tqdm, scikit-learn
* Kaggle environment for evaluation

---

## Results Summary

| Model                                      | Channels | Localizers | wAUC (val) | wAUC (LB)      |
| ------------------------------------------ | -------- | ---------- | ---------- | -------------- |
| Baseline MaxViT                            | 32       | None       | 0.67       | 0.67           |
| MaxViT + localizers (no regularization)    | 35       | Yes        | 0.XX       | 0.XX           |
| MaxViT + localizers + epoch-seeded dropout | 35       | Yes        | 0.XX       | 0.XX           |

---

## Citation / Credit

* RSNA Intracranial Aneurysm Detection Challenge 2024–2025
**Maintainer:** *Matt Graham*
**Last Updated:** October 2025

---
