# ⚡ 3-Class Image Classifier: Airplane · Automobile · Ship

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://lovnishverma-airplane-car-ship-classifier.hf.space)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/datasets/princelv84/airplane-car-ship/data)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Training_Pipeline-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/code/princelv84/airplanes-cars-ships)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00.svg)](https://gradio.app/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-FF6F00.svg?logo=tensorflow)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A production-ready deep learning classifier** that distinguishes between airplanes, automobiles, and ships with **99.83% test accuracy** — built with MobileNetV2 transfer learning, Grad-CAM explainability, and INT8 TFLite quantization for edge deployment.

**🔴 Live Demo:** [lovnishverma-airplane-car-ship-classifier.hf.space](https://lovnishverma-airplane-car-ship-classifier.hf.space)  
**📂 GitHub:** [Airplane-Car-Ship-Classifier-with-Explainability](https://github.com/lovnishverma/Airplane-Car-Ship-Classifier-with-Explainability)

---

<img width="1918" height="967" alt="Gradio App Demo" src="https://github.com/user-attachments/assets/84e473fb-b4db-4070-a33c-f4422039eaf0" />

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Performance Results](#-performance-results)
3. [Dataset](#-dataset)
4. [Architecture & Methodology](#-architecture--methodology)
5. [Two-Phase Training Strategy](#-two-phase-training-strategy)
6. [Data Pipeline & Augmentation](#-data-pipeline--augmentation)
7. [Training Curves Analysis](#-training-curves-analysis)
8. [Model Interpretability — Grad-CAM](#-model-interpretability--grad-cam)
9. [Edge Deployment — INT8 TFLite Export](#-edge-deployment--int8-tflite-export)
10. [Gradio Web App](#-gradio-web-app)
11. [Engineering Fixes & Design Decisions](#-engineering-fixes--design-decisions)
12. [Local Installation & Usage](#%EF%B8%8F-local-installation--usage)
13. [File Structure](#-file-structure)
14. [Author](#-author)

---

## 🎯 Project Overview

This project was built as part of the **AIML programme with IIT Ropar** and demonstrates a complete, production-grade image classification pipeline — from raw data to a deployed web app with explainability. It goes beyond a typical "train and evaluate" notebook by addressing three real-world concerns that are often ignored in academic projects:

| Concern | Solution |
|---|---|
| **Model Accuracy** | Two-phase transfer learning (Feature Extraction → Fine-Tuning) |
| **Model Trust** | Grad-CAM attention maps to visually verify what the model looks at |
| **Edge Deployment** | INT8 TFLite quantization → 3.08 MB model for mobile/IoT |

The pipeline includes 7 documented engineering fixes applied over iterative development to ensure stability, correctness, and reproducibility.

---

## 📊 Performance Results

The fine-tuned model achieves near-perfect, state-of-the-art results on both validation and a fully held-out test set.

| Metric | Value |
|---|---|
| **Validation Accuracy** | **99.65%** (Loss: 0.0184) |
| **Test Accuracy** | **99.83%** (Loss: 0.0076) |
| **TFLite INT8 Sanity Check** | **~98.8%** (on CPU, no GPU) |
| **Model Size (TFLite INT8)** | **~3.08 MB** |

### Classification Report — Validation Set

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **airplane** | 1.000 | 0.990 | 0.995 | 200 |
| **automobile** | 0.995 | 1.000 | 0.998 | 200 |
| **ship** | 0.994 | 1.000 | 0.997 | 176 |
| *Weighted Avg* | | | *0.997* | *576* |

### Classification Report — Test Set (Held-out)

| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **airplane** | 1.000 | 0.995 | 0.997 | 189 |
| **automobile** | 1.000 | 1.000 | 1.000 | 193 |
| **ship** | 0.995 | 1.000 | 0.998 | 200 |
| *Weighted Avg* | | | *0.998* | *582* |

### Confusion Matrices

The confusion matrices confirm the near-perfect separation between all three classes. On the held-out test set, only **1 airplane was misclassified as a ship** out of 582 total predictions.

> *Validation (left) and Test held-out (right) confusion matrices*

---

## 🗂️ Dataset

- **Source:** [Kaggle — princelv84/airplane-car-ship](https://www.kaggle.com/datasets/princelv84/airplane-car-ship/data)
- **Classes:** `airplane`, `automobile`, `ship`
- **Input Size:** 128 × 128 × 3 (RGB)

**Expected directory structure:**
```
dataset/
├── train/
│   ├── airplane/
│   ├── automobile/
│   └── ship/
├── val/
│   ├── airplane/
│   ├── automobile/
│   └── ship/
└── test/
    ├── airplane/
    ├── automobile/
    └── ship/
```

---

## 🧠 Architecture & Methodology

### Model Architecture

The classifier is built on a **MobileNetV2** backbone pre-trained on ImageNet, with a custom classification head attached on top.

```
Input (128×128×3)
    ↓
Rescaling (÷255, inline normalization)
    ↓
MobileNetV2 Backbone (pre-trained on ImageNet)
   └── Feature extractor: Conv layers → out_relu (last conv activation)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu)  →  BatchNormalization  →  Dropout(0.4)
    ↓
Dense(128, relu)  →  Dropout(0.3)
    ↓
Dense(3, softmax)   ← [airplane | automobile | ship]
```

**Why MobileNetV2?**
- Designed specifically for efficiency on mobile and edge devices (uses depthwise separable convolutions)
- Pre-trained on ImageNet (1.2M images, 1000 classes) — brings rich, general-purpose feature detectors for edges, textures, and shapes out of the box
- Bottleneck residual blocks with linear activations prevent information loss in low-dimensional feature spaces
- Input size of 128×128 keeps inference fast without sacrificing accuracy for this task

---

## 🔄 Two-Phase Training Strategy

Training is split into two deliberate phases. This is **not arbitrary** — it is a principled approach to prevent catastrophic forgetting while maximizing task-specific adaptation.

### Phase 1 — Feature Extraction (Frozen Backbone)

**What happens:** The entire MobileNetV2 backbone is frozen (`base.trainable = False`). Only the newly added classification head (Dense → BN → Dropout → Dense → Softmax) is trained.

**Why this is necessary:**  
When a new, randomly initialized classification head is attached to a pre-trained backbone, its initial weights are completely random. If the whole network is trained from scratch, the **massive gradient signals** from these random errors propagate backward and irreversibly destroy the delicate, pre-trained feature maps that MobileNetV2 spent millions of training steps learning. This is called **catastrophic forgetting**.

By freezing the backbone, we allow the classification head to "warm up" — it learns to correctly interpret the feature representations that MobileNetV2 already provides, without disturbing them.

**Hyperparameters (Phase 1):**
- Optimizer: `Adam(lr=1e-3)`
- Epochs: up to 15 (with EarlyStopping)
- Callbacks: `EarlyStopping(patience=5)`, `ReduceLROnPlateau(factor=0.5, patience=3)`, `ModelCheckpoint`

---

### Phase 2 — Fine-Tuning (Unfreezing Top 30 Layers)

**What happens:** The best Phase 1 checkpoint is reloaded. The **top 30 layers** of the MobileNetV2 backbone are unfrozen. The entire model is retrained end-to-end with a very small learning rate.

**Why only the top 30 layers?**  
MobileNetV2 layers are not equal. Early layers (close to the input) detect low-level primitives — edges, corners, color blobs — that are universal across all vision tasks. Destroying these is wasteful and harmful. Later layers (close to the output) detect high-level, abstract patterns — specific shapes, object parts — that are more dataset-specific. By unfreezing only the top 30 layers, we allow the model to adapt its *high-level* feature detectors specifically to the geometry of **planes, cars, and ships** while preserving the foundational low-level feature extractors.

**Why CosineDecayRestarts?**  
Standard fixed learning rates often get stuck in sharp local minima. `CosineDecayRestarts` smoothly cycles the learning rate from a small initial value down to near-zero and then "restarts" — this allows the optimizer to escape sharp minima and find flatter, more generalizable solutions.

> ⚠️ **Critical fix applied:** `CosineDecayRestarts` is a learning rate *schedule* object. It must be passed directly **inside the optimizer** (`Adam(learning_rate=schedule)`), NOT listed as a separate callback. Passing it as a callback is a silent bug that does nothing.

**Hyperparameters (Phase 2):**
- Optimizer: `Adam(learning_rate=CosineDecayRestarts(initial_lr=1e-5, first_decay_steps=5, t_mul=2.0))`
- Epochs: up to 15 (with EarlyStopping)
- Trainable base layers: top 30 / total MobileNetV2 layers

---

## 🔧 Data Pipeline & Augmentation

The `tf.data` API is used for all data loading to maximize GPU utilization through prefetching and parallel preprocessing.

### Augmentation Strategy

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),      # horizontal only — see below
    layers.RandomRotation(0.15),          # ±15% rotation
    layers.RandomZoom(0.15),              # ±15% zoom
    layers.RandomTranslation(0.1, 0.1),   # ±10% translation
    layers.RandomContrast(0.2),           # ±20% contrast
])
```

**Why `horizontal` flip only (not `horizontal_and_vertical`)?**  
This is a domain-specific decision. An upside-down airplane or an inverted ship is physically unrealistic — such images do not appear in real-world deployments. Applying vertical flips would inject out-of-distribution training samples, teaching the model patterns that will never appear at inference time. This is a common but silent mistake in generic augmentation pipelines.

### Pipeline Fix: `drop_remainder=True`

```python
train_ds_raw = train_ds_raw.batch(BATCH, drop_remainder=True)
val_ds = val_ds_raw.batch(BATCH, drop_remainder=True)
```

**Why this matters:** The dataset size may not be perfectly divisible by the batch size. The final "remainder" batch is smaller than `BATCH`. When MixUp augmentation computes weighted combinations of two randomly sampled batches, it assumes both batches have identical shapes. A smaller remainder batch causes a **silent shape mismatch** that crashes or corrupts training. `drop_remainder=True` discards the final partial batch, guaranteeing all batches are exactly `BATCH` size throughout every epoch.

---

## 📈 Training Curves Analysis

The training curves reveal the expected two-phase behaviour:

**Phase 1 (Epochs 0–7, before dashed line):**
- **Validation accuracy** starts very high (~99.2%) immediately — because the pre-trained MobileNetV2 backbone already extracts excellent features
- **Training accuracy** climbs steadily from ~91.5% to ~97% as the classification head learns to use those features
- The gap between train and val accuracy is **not overfitting** — it is a natural consequence of augmentation being applied only during training (augmented images are harder)

**Phase 2 (Epochs 7–15, after dashed line):**
- Training accuracy **dips sharply** at the start of Phase 2 — this is expected and correct. The `CosineDecayRestarts` schedule resets the learning rate, and unfreezing layers temporarily destabilizes the network as it begins adapting backbone weights
- Both training and validation accuracy converge upward as the backbone fine-tunes
- Validation loss remains low and stable (~0.018–0.040), confirming no overfitting

---

## 🔍 Model Interpretability — Grad-CAM

Deep learning classifiers should never be deployed as "black boxes". Grad-CAM (Gradient-weighted Class Activation Mapping) provides a visual explanation of **what spatial regions the model attends to** when making a prediction.

### How Grad-CAM Works (Technical)

1. **Target layer:** The last convolutional activation in MobileNetV2 — `out_relu`. This is the richest spatial feature map before global pooling collapses spatial information.

2. **Forward pass (with tape):** Run the input image through the network. Record the feature map at `out_relu`. Tell `tf.GradientTape` to watch it.

3. **Gradient computation:** Compute the gradient of the predicted class score with respect to every spatial location in the feature map:
   ```
   grads = ∂(class_score) / ∂(feature_map at out_relu)
   ```

4. **Importance weighting:** Average the gradients across all channels to get per-channel importance weights:
   ```
   pooled_grads = mean(grads, axis=[height, width, batch])
   ```

5. **Heatmap generation:** Compute a weighted sum of the feature map channels using the importance weights:
   ```
   heatmap = feature_map @ pooled_grads
   ```

6. **Post-processing:** Apply ReLU (discard negative contributions), normalize to [0,1], resize to input image dimensions, apply JET colormap, and overlay at 40% opacity on the original image.

### Implementation Note: Eager Execution

The Grad-CAM implementation uses a custom layer-by-layer eager execution loop rather than building a sub-model with `tf.keras.Model(inputs, outputs)`. This is required because the MobileNetV2 backbone is nested as a single layer inside the outer model — a standard sub-model approach would not correctly intercept the internal `out_relu` activation. The solution explicitly:
1. Executes layers *before* the backbone manually
2. Uses a `grad_model = tf.keras.Model(backbone_inputs, [conv_out, backbone_output])` to expose the internal activation
3. Executes the classification head layers manually *after* the backbone

### What the Heatmaps Show

From the Grad-CAM output:
- **Airplane predictions:** The model attends to fuselage body, wings, and tail structures — not the sky or runway background
- The activation is correctly centered on the aircraft structure across widely varying viewpoints (ground-level, in-flight, close-up)
- All 12 displayed validation samples show **green titles** (correct predictions), and the attention maps are physically meaningful

This confirms the model has learned genuinely discriminative visual features, not spurious correlations with background colors (sky, road, ocean).

---

## 🚀 Edge Deployment — INT8 TFLite Export

### Why Quantize?

The full Keras model (`best_finetuned.keras`) is ~26 MB — impractical for mobile or IoT devices. INT8 quantization compresses the model by representing weights as 8-bit integers instead of 32-bit floats.

### How Full INT8 Quantization Works

**Step 1 — Calibration with representative data:**  
Unlike dynamic-range quantization (which estimates scale factors analytically), full INT8 quantization requires observing the actual distribution of activations at every layer. A `representative_dataset_gen` function feeds **50 calibration batches** from the validation set through the model:

```python
def representative_data_gen():
    for images, _ in val_ds.take(50):
        yield [tf.cast(images, tf.float32)]
```

**Step 2 — Scale and zero-point estimation:**  
The TFLite converter observes the min/max activation values at every layer and computes the optimal `scale` and `zero_point` parameters to map the float32 range to INT8 (–128 to 127) with minimum quantization error.

**Step 3 — Conversion:**

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,   # fallback for ops not supporting INT8
]
converter.inference_input_type = tf.float32   # float I/O kept for compatibility
converter.inference_output_type = tf.float32
```

**Why keep float32 I/O?**  
Setting input/output to `float32` means the host application does not need to handle quantization scaling at the API boundary — you pass a normal float image array in, and get float probabilities out. The INT8 computation happens internally, giving the size/speed benefits without complicating the deployment interface.

**Result:**

| Metric | Value |
|---|---|
| Original Keras model | ~26 MB |
| INT8 TFLite model | **~3.08 MB** |
| Compression ratio | **~8.4×** |
| Sanity-check accuracy (CPU) | **~98.8%** |

---

## 🌐 Gradio Web App

The `app.py` file provides an interactive web interface with real-time Grad-CAM visualization.

**How it works:**
1. User uploads any image via the Gradio UI
2. Image is resized to 128×128 and passed to the Keras model
3. Top-3 class probabilities are returned as a labeled bar chart
4. Grad-CAM heatmap is computed using the same eager-execution approach described above
5. Heatmap is overlaid on the original image (40% heat, 60% original) and displayed alongside the predictions

**Key implementation detail in `app.py`:**  
The heatmap returned by OpenCV's `applyColorMap` is in **BGR** color space. It must be explicitly converted to RGB before blending with the original image:
```python
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
```
Without this conversion, the heatmap colors are inverted (red becomes blue), making the visualization misleading.

---

## 🛠️ Engineering Fixes & Design Decisions

This version implements 7 documented fixes over a baseline implementation. Each fix addresses a real, non-obvious bug or design flaw:

| # | Fix | Problem Solved |
|---|---|---|
| 1 | `drop_remainder=True` on all dataset batches | Silent shape mismatch at epoch end when batch size doesn't divide dataset size — breaks MixUp and shape-sensitive ops |
| 2 | Base model accessed via `get_layer(BASE_NAME)` by name | Positional index (`model.layers[1]`) breaks if any layer is inserted/removed during model construction |
| 3 | Full `classification_report` + per-class confusion matrix | Aggregate accuracy hides per-class failure modes; needed for trust and deployment readiness |
| 4 | INT8 TFLite with `representative_dataset_gen` | Dynamic-range-only quantization gives worse accuracy-size tradeoff; calibration data needed for proper INT8 scale estimation |
| 5 | `RandomFlip("horizontal")` only | Vertical flip creates physically impossible images (upside-down vehicles/aircraft) — out-of-distribution training noise |
| 6 | `CosineDecayRestarts` passed into `Adam(learning_rate=...)` | Passing it as a `callbacks=` list entry is a silent no-op; it only works as an optimizer argument |
| 7 | Eager layer-by-layer Grad-CAM loop | Standard sub-model Grad-CAM fails when the backbone is a nested `Model` layer — requires explicit traversal to access internal activations |

---

## 🛠️ Local Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/lovnishverma/Airplane-Car-Ship-Classifier-with-Explainability.git
cd Airplane-Car-Ship-Classifier-with-Explainability
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `tensorflow`, `numpy`, `gradio`, `opencv-python-headless`, `scikit-learn`, `pillow`

### 3. Run the Gradio app

```bash
python app.py
```

The app will launch at `http://localhost:7860`. Upload any airplane, car, or ship image to get predictions and a Grad-CAM visualization.

### 4. Run the training notebook

The full training pipeline is available as a Kaggle notebook:  
🔗 [kaggle.com/code/princelv84/airplanes-cars-ships](https://www.kaggle.com/code/princelv84/airplanes-cars-ships)

To run locally:
```bash
jupyter notebook airplanes_cars_ships.ipynb
```

---

## 📁 File Structure

```
├── app.py                    # Gradio web app with Grad-CAM visualization
├── best_finetuned.keras      # Saved Keras model (tracked via Git LFS, ~26 MB)
├── model_int8.tflite         # INT8 quantized TFLite model (~3.08 MB)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── outputs/                  # Training artifacts
│   ├── training_curves.png   # Phase 1 → Phase 2 accuracy/loss plots
│   ├── confusion_matrices.png # Val and test confusion matrices
│   └── gradcam.png           # Grad-CAM attention maps (12 samples)
```

---

## 👨‍💻 Author

**Lovnish Verma**  
Project Engineer, NIELIT Ropar  
AIML Programme in collaboration with IIT Ropar

[![GitHub](https://img.shields.io/badge/GitHub-lovnishverma-181717?logo=github)](https://github.com/lovnishverma)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-lovnishverma-yellow)](https://huggingface.co/lovnishverma)
[![Kaggle](https://img.shields.io/badge/Kaggle-princelv84-20BEFF?logo=kaggle)](https://www.kaggle.com/princelv84)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-lovnishverma-0A66C2?logo=linkedin)](https://linkedin.com/in/lovnishverma)

---

⭐ **If this project helped you understand Transfer Learning, Grad-CAM, or TFLite quantization — give it a star!**
