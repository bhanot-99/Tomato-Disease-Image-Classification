# 🍅 Tomato Disease Image Classification

## 📌 Overview

This project focuses on **automated detection and classification of tomato leaf diseases** using deep learning. The goal is to assist in early disease identification, reducing reliance on manual inspection and enabling scalable agricultural monitoring systems.

The system is designed with a vision toward **real-world deployment**, particularly in **resource-constrained environments** such as edge devices and UAV-based monitoring systems.

---

## 🎯 Problem Statement

Manual detection of plant diseases is:

* Time-consuming
* Prone to human error
* Not scalable for large farms

This project aims to build a **robust image classification model** that can accurately identify different tomato leaf diseases from images.

---

## 🧠 Approach

### 1. Data Processing

* Image resizing and normalization
* Dataset splitting (train / validation / test)
* Optional data augmentation (if applied)

> 📌 Dataset Link: **https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data**

---

### 2. Model Architecture

* Base model: **MobileNetV2 (pretrained on ImageNet — Transfer Learning)**
* Framework: **TensorFlow 2.10 with Keras**
* Loss function: **Categorical Crossentropy**
* Optimizer: **Adam (learning rate = 0.001)**

---

### 3. Training Pipeline

* Batch size: **32**
* Epochs: **20 (Early Stopping triggered automatically)**
* Learning rate: **0.001 (initial) → reduced automatically via ReduceLROnPlateau**
* Hardware used: **Windows CPU (TensorFlow 2.10 — GPU support attempted but fell back to CPU due to Windows compatibility)**

---

## 📊 Results

| Metric    | Value       |
| --------- | ----------- |
| Accuracy  | **89.06%** |
| Precision | **0.897**   |
| Recall    | **0.892**   |
| F1 Score  | **0.893**   |

* Number of classes: **10**
* Dataset size: **9325**

---

## 📈 Sample Outputs

(Add a few prediction examples here)

```
/outputs/sample/
```

---

## 📊 Per-Class Performance

The model was evaluated across all classes using **Precision, Recall, and F1-score**, providing a detailed view of class-wise performance.

| Class                  | Precision | Recall | F1 Score |
| ---------------------- | --------- | ------ | -------- |
| Bacterial spot         | 0.861     | 0.993  | 0.923    |
| Early blight           | 0.848     | 0.820  | 0.834    |
| Late blight            | 0.937     | 0.893  | 0.915    |
| Leaf Mold              | 0.904     | 0.923  | 0.913    |
| Septoria leaf spot     | 0.914     | 0.853  | 0.883    |
| Spider mites           | 0.885     | 0.773  | 0.826    |
| Target Spot            | 0.750     | 0.860  | 0.801    |
| Yellow Leaf Curl Virus | 0.979     | 0.953  | 0.966    |
| Tomato mosaic virus    | 0.944     | 0.911  | 0.927    |
| Healthy                | 0.946     | 0.940  | 0.943    |

---

### 🔍 Observations

* Strong performance across most classes, with **F1-scores above 0.90** for several diseases.
* **Yellow Leaf Curl Virus** achieved the highest performance (F1: 0.966), indicating strong feature separability.
* Slightly lower recall observed in:

  * *Spider mites* (0.773)
  * *Target Spot* (precision: 0.750)
* These cases suggest **inter-class similarity or limited data representation**.

---

### 📌 Note

Performance may vary depending on dataset distribution and image quality. Further improvements can be achieved through:

* Data augmentation
* Class balancing
* Lightweight architecture optimization for edge deployment

---


## ⚙️ Project Structure

```
Tomato-Disease-Classification/
│
├── notebooks/          # Experimentation and prototyping
├── models/             # Model architecture / saved models (if small)
├── outputs/            # Predictions and sample results
├── dataset/            # (Not included in repo)
├── requirements.txt    # Dependencies
└── README.md
```
---

## 🌍 Applications

* Smart agriculture systems
* UAV-based crop monitoring
* Edge AI for on-field disease detection
* Decision support for farmers

---

## 🔮 Future Improvements

* Deploy lightweight models (e.g., MobileNet) for edge devices
* Real-time inference pipeline
* Integration with UAV imagery
* Web/mobile interface for live predictions
* Model optimization (quantization / pruning)

---

