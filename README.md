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

# ⚡ Airplane-Car-Ship Classifier with Explainability

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://lovnishverma-airplane-car-ship-classifier.hf.space)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/princelv84/airplane-car-ship/data)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Training_Pipeline-blue.svg)](https://www.kaggle.com/code/princelv84/airplanes-cars-ships)
[![Gradio](https://img.shields.io/badge/UI-Gradio-pink.svg)](https://gradio.app/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange.svg)](https://tensorflow.org/)

A robust, fine-tuned deep learning classifier capable of distinguishing between airplanes, cars (automobiles), and ships. This project not only delivers high-accuracy predictions but also incorporates explainability techniques to visualize model decision-making.

**Live Demo:** [Hugging Face Space](https://lovnishverma-airplane-car-ship-classifier.hf.space)

**GitHub Repo:** https://github.com/lovnishverma/Airplane-Car-Ship-Classifier-with-Explainability

---

<img width="1918" height="967" alt="image" src="https://github.com/user-attachments/assets/84e473fb-b4db-4070-a33c-f4422039eaf0" />

---

## 🧠 Approach & Explainability

*   **Model Training:** The model is built using a fine-tuned architecture (`best_finetuned.keras`), trained on a specialized Kaggle dataset. 
*   **Web Interface:** The interactive user interface is powered by Gradio (`sdk_version: 6.14.0`) via `app.py`, allowing users to easily upload images and receive classifications.
*   **Explainability:** To ensure transparency, the classifier includes explainability features. Using techniques like Grad-CAM, the model generates attention maps that highlight the specific regions of an image (e.g., the wings of a plane or the hull of a ship) that contributed most to its final prediction. 

---

## 📊 Performance Metrics

The model demonstrates exceptional performance across both validation and held-out test sets, achieving near-perfect accuracy. 

*   **Validation Accuracy:** 99.58% (Loss: 0.0086)
*   **Test Accuracy:** 99.66% (Loss: 0.0094)

### Classification Report — Validation Set
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **airplane** | 1.000 | 0.988 | 0.994 | 160 |
| **automobile** | 0.988 | 1.000 | 0.994 | 160 |
| **ship** | 1.000 | 1.000 | 1.000 | 160 |
| *Accuracy* | | | *0.996* | *480* |
| *Macro Avg* | 0.996 | 0.996 | 0.996 | 480 |
| *Weighted Avg* | 0.996 | 0.996 | 0.996 | 480 |

**

### Classification Report — Test Set (Held-out)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **airplane** | 0.995 | 0.995 | 0.995 | 189 |
| **automobile** | 1.000 | 1.000 | 1.000 | 193 |
| **ship** | 0.995 | 0.995 | 0.995 | 200 |
| *Accuracy* | | | *0.997* | *582* |
| *Macro Avg* | 0.997 | 0.997 | 0.997 | 582 |
| *Weighted Avg* | 0.997 | 0.997 | 0.997 | 582 |

**

---

## 🛠️ Local Installation & Usage

To run this project locally, clone the repository and install the dependencies outlined in `requirements.txt`.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lovnishverma/Airplane-Car-Ship-Classifier-with-Explainability.git
   cd Airplane-Car-Ship-Classifier-with-Explainability
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Dependencies include: `tensorflow`, `numpy`, `gradio`, `opencv-python-headless`)*

3. **Run the Gradio App:**
   ```bash
   python app.py
   ```

---

## 👨‍💻 Author

**Lovnish Verma**  
Project Engineer, NIELIT Ropar

⭐ *If you found this repository useful or informative, please consider giving it a star!*

