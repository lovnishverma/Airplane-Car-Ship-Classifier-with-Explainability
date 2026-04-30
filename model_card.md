# Model Card — Airplane · Automobile · Ship Classifier

## Model Details

| Field | Value |
|---|---|
| **Model Name** | `airplane-car-ship-classifier` |
| **Version** | v1.0 |
| **Type** | Image Classification (3-class) |
| **Architecture** | MobileNetV2 + custom classification head |
| **Framework** | TensorFlow / Keras |
| **Input** | RGB image, resized to 128 × 128 × 3 |
| **Output** | Softmax probability vector over 3 classes |
| **Model Size** | ~26 MB (Keras) · ~3.08 MB (INT8 TFLite) |
| **License** | Apache-2.0 |
| **Author** | Lovnish Verma, Project Engineer, NIELIT Ropar |
| **Programme** | AIML programme in collaboration with IIT Ropar |
| **Date** | 2025 |

---

## Model Description

This model classifies images into one of three vehicle categories: **airplane**, **automobile**, or **ship**. It is built using transfer learning on a MobileNetV2 backbone pre-trained on ImageNet, with a custom two-layer dense classification head trained via a deliberate two-phase strategy (frozen feature extraction followed by selective fine-tuning of the top 30 backbone layers).

The repository includes a live Gradio web app with real-time Grad-CAM explainability and an INT8-quantized TFLite export for edge/mobile deployment.

**Live demo:** [lovnishverma-airplane-car-ship-classifier.hf.space](https://lovnishverma-airplane-car-ship-classifier.hf.space)  
**GitHub:** [Airplane-Car-Ship-Classifier-with-Explainability](https://github.com/lovnishverma/Airplane-Car-Ship-Classifier-with-Explainability)  
**Kaggle notebook:** [kaggle.com/code/princelv84/airplanes-cars-ships](https://www.kaggle.com/code/princelv84/airplanes-cars-ships)

---

## Intended Use

### Primary Use Cases
- Educational demonstration of transfer learning, Grad-CAM explainability, and TFLite edge deployment
- Academic reference for AIML/Deep Learning courses (NIELIT / IIT Ropar programme)
- Baseline for 3-class vehicle image classification tasks

### Out-of-Scope Uses
- Safety-critical or real-time aviation/maritime systems
- Classification of vehicle types not present in training data (e.g., trains, helicopters, boats)
- Adversarial or security-sensitive applications
- High-volume production inference without further validation

---

## Training Data

| Property | Value |
|---|---|
| **Source** | [Kaggle — princelv84/airplane-car-ship](https://www.kaggle.com/datasets/princelv84/airplane-car-ship/data) |
| **Classes** | `airplane`, `automobile`, `ship` |
| **Split** | Train / Validation / Test |
| **Image size** | 128 × 128 × 3 (RGB) |

### Data Augmentation (training only)

| Transform | Setting | Rationale |
|---|---|---|
| `RandomFlip` | horizontal only | Vertical flip creates physically unrealistic images (inverted aircraft/ships) |
| `RandomRotation` | ±15% | Handles varied viewing angles |
| `RandomZoom` | ±15% | Scale invariance |
| `RandomTranslation` | ±10% (H and W) | Handles off-centre subjects |
| `RandomContrast` | ±20% | Lighting robustness |

MixUp augmentation was also applied at the batch level. `drop_remainder=True` was used on all dataset batches to prevent shape mismatches.

---

## Architecture

```
Input (128×128×3)
    ↓
Rescaling (÷255, inline normalization)
    ↓
MobileNetV2 Backbone (pre-trained on ImageNet, top 30 layers fine-tuned)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu) → BatchNormalization → Dropout(0.4)
    ↓
Dense(128, relu) → Dropout(0.3)
    ↓
Dense(3, softmax)  ←  [airplane | automobile | ship]
```

### Training Strategy

**Phase 1 — Feature Extraction:** MobileNetV2 backbone frozen. Only the classification head trained (Adam, lr=1e-3, up to 15 epochs with EarlyStopping).

**Phase 2 — Fine-Tuning:** Best Phase 1 checkpoint reloaded. Top 30 layers of the backbone unfrozen. End-to-end training with `CosineDecayRestarts(initial_lr=1e-5, first_decay_steps=5, t_mul=2.0)`.

---

## Performance

### Summary Metrics

| Split | Accuracy | Loss |
|---|---|---|
| Validation | **99.65%** | 0.0184 |
| Test (held-out) | **99.83%** | 0.0076 |
| TFLite INT8 (CPU) | **~98.8%** | — |

### Classification Report — Validation Set

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| airplane | 1.000 | 0.990 | 0.995 | 200 |
| automobile | 0.995 | 1.000 | 0.998 | 200 |
| ship | 0.994 | 1.000 | 0.997 | 176 |
| **Weighted Avg** | | | **0.997** | **576** |

### Classification Report — Test Set (Held-out)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| airplane | 1.000 | 0.995 | 0.997 | 189 |
| automobile | 1.000 | 1.000 | 1.000 | 193 |
| ship | 0.995 | 1.000 | 0.998 | 200 |
| **Weighted Avg** | | | **0.998** | **582** |

**Notable failure:** On the held-out test set, 1 airplane was misclassified as a ship out of 582 total predictions.

---

## Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is integrated into the Gradio web app and the training notebook. Heatmaps are computed using an eager layer-by-layer execution loop targeting the `out_relu` layer of MobileNetV2 — the last convolutional activation before global pooling.

Inspection of 12 validation samples confirms the model attends to **physically meaningful structures**: fuselage/wings for airplanes, vehicle body for automobiles, and hull geometry for ships — not background (sky, road, ocean). This provides evidence the model has learned genuinely discriminative visual features rather than spurious correlations.

---

## Edge Deployment

An INT8-quantized TFLite model (`model_int8.tflite`) is provided for mobile and IoT deployment.

| Metric | Value |
|---|---|
| Original Keras model | ~26 MB |
| INT8 TFLite model | **~3.08 MB** |
| Compression ratio | **~8.4×** |
| Accuracy drop | ~1% (98.8% vs 99.83%) |

Full INT8 quantization was used with a `representative_dataset_gen` (50 calibration batches from the validation set). Float32 I/O is preserved at the API boundary for deployment compatibility.

---

## Limitations

- **Domain scope:** The model is trained and evaluated on a single curated Kaggle dataset. Real-world performance on very different image distributions (e.g., infrared, satellite, low-light) has not been evaluated.
- **Class scope:** Only three classes. Inputs from other categories (trains, helicopters, trucks) will be forced into one of the three labels with no out-of-distribution detection.
- **Input resolution:** Images are downscaled to 128×128. Fine-grained texture differences at higher resolutions are not captured.
- **Near-perfect accuracy caveat:** 99.83% accuracy on a clean, balanced test set does not generalise claims to all real-world settings. The classes (airplane, automobile, ship) are visually very distinct — this is an easier-than-average classification task.
- **No confidence calibration:** Softmax probabilities are not calibrated and should not be treated as well-calibrated probability estimates in downstream systems.

---

## Ethical Considerations

- This model is released for educational purposes and should not be used in safety-critical applications without independent validation.
- The training dataset was sourced from Kaggle and has not been audited for demographic or geographic bias in the depicted objects.
- Grad-CAM explainability provides visual interpretability but does not constitute a formal guarantee of model behaviour.

---

## Citation

If you use this model or any part of this pipeline in academic or research work, please cite:

```
@misc{verma2025airplanecarsship,
  author       = {Lovnish Verma},
  title        = {Airplane-Car-Ship Classifier with Explainability},
  year         = {2025},
  institution  = {NIELIT Ropar / IIT Ropar AIML Programme},
  url          = {https://github.com/lovnishverma/Airplane-Car-Ship-Classifier-with-Explainability}
}
```

---

## Contact

**Lovnish Verma**  
Project Engineer, NIELIT Ropar  
[github.com/lovnishverma](https://github.com/lovnishverma) · [huggingface.co/lovnishverma](https://huggingface.co/lovnishverma) · [linkedin.com/in/lovnishverma](https://linkedin.com/in/lovnishverma)
