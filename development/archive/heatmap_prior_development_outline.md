# Heatmap Prior Development (AABB Detection)

## 📌 Context

This project extends **RTMDet** (AABB parcel detector) with a **prior heatmap channel** (`PriorH`).
The goal is to integrate temporal and sensor-based priors (e.g., Kalman filter state estimates, conveyor logic, scan events) into training and inference, while ensuring the model does not become over-reliant on priors.

---

## 🎯 Goals

* Implement a **single-channel prior heatmap** that encodes parcel location probability.
* Train the detector with GT-derived priors first, before integrating real priors (from sensors/Kalman).
* Evaluate improvements in detection robustness, particularly under occlusion, glare, or motion blur.
* Ensure minimal inference cost (<3% FPS drop).

---

## 📂 Repo Structure

```
vault_mmdetection/
├── mmdet/
│   ├── datasets/transforms/priorh.py      # New transform (generate PriorH)
│   ├── models/utils/prior_adapter.py      # Input adapter for 4-channel images
│   └── ...
├── tools/
│   ├── train.py                           # Training entry
│   ├── test.py                            # Evaluation entry
│   └── vis_prior.py                       # Debugging / visualization script
└── configs/
    └── rtmdet/
        └── rtmdet_priorh_config.py        # Config with PriorH pipeline + adapter
```

---

## 🔄 Pipeline

```
LoadImage → LoadAnnotations → Resize/Pad → GeneratePriorH → PackDetInputs → RTMDet
```

---

## 🛠️ Implementation Plan

### 1. Generate Prior Heatmap

* **Transform**: `GeneratePriorH`
  * Input: GT bounding boxes (x, y, w, h).
  * Output: heatmap channel (HxW, float32).
  * Method: Gaussian centered at bbox, std scaled by size.
    ```python
    σ_x = k_w × width, σ_y = k_h × height  # defaults: k_w=k_h=0.15
    σ_min = 2.0 px  # prevent needle peaks
    ```
  * Normalization: clamp to `[0,1]`.
  * **Training noise**: Random jitter (±1% image size), σ scaling (0.9-1.1x), intensity (0.9-1.1x).
  * **PriorDrop**: 30% chance to zero entire heatmap (prevent over-reliance).

* **Testing**
  * Unit test: check Gaussian symmetry & correct center.
  * Augmentations: verify heatmap flips/resizes with image.

---

### 2. Modify RTMDet Input

* **Adapter**: `PriorInputAdapter`
  * Accept 4-channel tensors (RGB + PriorH).
  * Initialize prior channel weight to ~0 (so baseline unaffected).
  * Training can learn to use PriorH progressively.

* **⚠️ Critical**: Place `GeneratePriorH` **AFTER** all geometric transforms (Resize/Pad).
  * Ensures heatmap matches final image geometry.

---

### 3. Training Strategy

* **Robust Learning**: Train with PriorDrop (30% zero heatmaps) + noise from start.
* **Progressive Enhancement**: Model learns to use priors when available, ignore when missing.
* **Weight Initialization**: Prior channel starts near-zero contribution, learns gradually.
* **Validation**: Uses clean GT priors (no dropout/noise) for consistent evaluation.

---

### 4. Validation Setup

* Validation uses **GT-based priors** (clean, no dropout/noise).
* Clearly mark in logs: *"GT prior — max potential benefit"*.
* Later, replace with Kalman-derived priors for production tests.

---

## ✅ Acceptance Criteria

* **Primary KPI**: ≥ +2 mAP_l (large objects) compared to baseline RTMDet.
* **Secondary KPIs**:
  * mAP_m: ≥ +1 improvement.
  * FPS drop: <3% on RTX 4090.
  * No loss of baseline performance when `PriorH=0`.

---

## 🔍 Testing

1. **Unit Tests**
   * `tests/test_priorh.py`: Gaussian center, augmentations.

2. **Smoke Training**
   * Run 2 epochs, ensure loss curves ~baseline.

3. **Ablations**
   * Baseline RTMDet vs RTMDet+PriorH.
   * Check robustness under occlusion / blur subsets.

4. **Visualization**
   * `tools/vis_prior.py` overlays PriorH onto sample images.
   * Required for PR review.

---

## 🚦 Workflow

* Branch: `feature/rtmdet-priorh`.
* PR 1: Add transform + tests.
* PR 2: Add adapter + config.
* PR 3: Run ablations, update docs.

---

## 📈 Future Steps

* Replace GT priors with **Kalman-based priors** projected into image space.
* Experiment with **multi-channel priors** (e.g., separate "top/bottom" vertex heatmaps).
* Evaluate **transformer-based detectors** with priors as additional tokens.

---

## 🔧 Quick Start

```python
# Core heatmap generation (reference)
def generate_prior_h(bboxes_xyxy, image_shape, kw=0.15, kh=0.15):
    H, W = image_shape
    prior = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    for x1, y1, x2, y2 in bboxes_xyxy:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        sx = max(kw * (x2 - x1), 2.0)
        sy = max(kh * (y2 - y1), 2.0)
        
        gaussian = np.exp(-((xx - cx)**2 / (2*sx**2) + (yy - cy)**2 / (2*sy**2)))
        prior += gaussian
    
    return np.clip(prior, 0, 1)
```