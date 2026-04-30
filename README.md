---
title: Airplane Car Ship Classifier
emoji: ⚡
colorFrom: pink
colorTo: green
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
short_description: Airplane-Car-Ship Classifier
---

# ⚡ 3-Class Image Classifier: Airplane, Automobile, Ship

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://lovnishverma-airplane-car-ship-classifier.hf.space)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/princelv84/airplane-car-ship/data)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Training_Pipeline-blue.svg)](https://www.kaggle.com/code/princelv84/airplanes-cars-ships)
[![Gradio](https://img.shields.io/badge/UI-Gradio-pink.svg)](https://gradio.app/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange.svg)](https://tensorflow.org/)

A highly optimized, production-ready deep learning classifier built to distinguish between airplanes, automobiles (cars), and ships. This project goes beyond simple classification by integrating **Grad-CAM explainability** (to visually verify model reasoning) and **INT8 TFLite quantization** (for lightweight edge deployment).

**Live Demo:** [Hugging Face Space](https://lovnishverma-airplane-car-ship-classifier.hf.space)

**GitHub Repo:** https://github.com/lovnishverma/Airplane-Car-Ship-Classifier-with-Explainability

---

<img width="1918" height="967" alt="image" src="https://github.com/user-attachments/assets/84e473fb-b4db-4070-a33c-f4422039eaf0" />

---

## 🧠 Methodology: How It Works & Why

This model uses **Transfer Learning** with a `MobileNetV2` backbone, optimized for mobile and edge devices[cite: 1]. To achieve near-perfect accuracy without overfitting, the training is split into two strategic phases.

### 1. The Two-Phase Training Strategy

**Phase 1: Feature Extraction (Frozen Backbone)**
*   **How:** We attach a new, randomly initialized classification head (Dense layers + Dropout) to the `MobileNetV2` backbone[cite: 1]. The entire backbone is strictly frozen[cite: 1].
*   **Why:** If we trained the whole network immediately, the massive initial errors from the random head would propagate large gradients backward, violently destroying the delicate, pre-trained feature maps MobileNetV2 learned from ImageNet[cite: 1]. Freezing the base allows the new head to gently "warm up" and learn how to interpret the existing features.

**Phase 2: Fine-Tuning (Unfreezing Top Layers)**
*   **How:** Once the head is capable, we unfreeze the top 30 layers of the `MobileNetV2` backbone[cite: 1]. We apply a `CosineDecayRestarts` learning rate schedule directly within the Adam optimizer to smoothly adjust weights[cite: 1].
*   **Why:** Unfreezing the top layers allows the model to adapt its high-level, abstract feature detectors specifically to the geometries of planes, cars, and ships, drastically pushing the accuracy higher without wrecking foundational edge/texture detectors[cite: 1].

### 2. Pipeline & Engineering Improvements
Several robust engineering fixes were implemented in this iteration to ensure stability and performance:
*   **Stable MixUp / Batching:** Added `drop_remainder=True` to the `tf.data` pipelines to prevent unequal-length batches at the end of epochs, which silently breaks shape-dependent operations[cite: 1].
*   **Domain-Specific Augmentation:** Applied horizontal-only `RandomFlip` (along with rotation, zoom, and contrast tweaks), because vertical flipping (upside-down ships/cars) creates unrealistic domain data[cite: 1].
*   **Resilient Architecture Referencing:** The base model is extracted functionally by its explicit name (`mobilenetv2_1.00_128`) rather than relying on fragile positional indexing[cite: 1].

---

## 🔍 Model Interpretability (Grad-CAM)

Deep learning models shouldn't be "black boxes"[cite: 1]. To ensure the model is learning the right concepts (and not just memorizing background artifacts like the sky or ocean), we use **Gradient-weighted Class Activation Mapping (Grad-CAM)**[cite: 1].

*   **How it works:** We extract the gradients of the target class with respect to the last convolutional feature map (`out_relu`) in the MobileNetV2 backbone[cite: 1]. This produces a coarse localization heatmap[cite: 1].
*   **Result:** You can physically see the model looking at the wings of an airplane, the wheels of a car, or the hull of a ship to make its decision[cite: 1].

---

## 🚀 Edge Deployment: INT8 TFLite Export

To make the model viable for mobile apps and IoT devices, it is exported to TensorFlow Lite with **Full INT8 Quantization**[cite: 1].
*   A `representative_dataset_gen` function feeds 50 calibration batches of validation data into the converter[cite: 1].
*   This estimates the proper scale and zero-point for 8-bit integer quantization[cite: 1].
*   **Result:** A highly compressed model (`model_int8.tflite`) that takes up only **~3.08 MB** of space, maintaining float32 I/O for deployment flexibility while achieving ~98.8% sanity-check accuracy natively on CPU[cite: 1, 2].

---

## 📊 Performance Results (Near-Perfect)

The fine-tuned model achieves state-of-the-art results on both validation and fully held-out test sets.

*   **Validation Accuracy:** **99.65%** (Loss: 0.0184)
*   **Test Accuracy:** **99.83%** (Loss: 0.0076)

### Classification Report — Validation Set
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **airplane** | 1.000 | 0.990 | 0.995 | 200 |
| **automobile** | 0.995 | 1.000 | 0.998 | 200 |
| **ship** | 0.994 | 1.000 | 0.997 | 176 |
| *Accuracy* | | | *0.997* | *576* |

### Classification Report — Test Set (Held-out)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **airplane** | 1.000 | 0.995 | 0.997 | 189 |
| **automobile** | 1.000 | 1.000 | 1.000 | 193 |
| **ship** | 0.995 | 1.000 | 0.998 | 200 |
| *Accuracy* | | | *0.998* | *582* |

---

## 🛠️ Local Installation & Usage

To run this project locally, clone the repository and install the dependencies outlined in `requirements.txt`.

1. **Clone the repository:**
   
```bash
   git clone [https://github.com/lovnishverma/Airplane-Car-Ship-Classifier-with-Explainability.git](https://github.com/lovnishverma/Airplane-Car-Ship-Classifier-with-Explainability.git)
   cd Airplane-Car-Ship-Classifier-with-Explainability
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Dependencies include: `tensorflow`, `numpy`, `gradio`, `opencv-python-headless`, `scikit-learn`, `pillow`)*

3. **Run the Gradio App:**
   ```bash
   python app.py
   ```

---

## 👨‍💻 Author

**Lovnish Verma**  
Project Engineer, NIELIT Ropar

⭐ *If you found this repository useful, informative, or learned something new about Transfer Learning and Grad-CAM, please consider giving it a star!*
