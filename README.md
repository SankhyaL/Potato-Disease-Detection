# 🥔 Potato Leaf Disease Detection using Deep Learning

A deep learning-based web application that detects and classifies potato leaf diseases using CNN, ResNet18, and YOLOv8 models. This project assists farmers and researchers by enabling real-time disease identification through a simple web interface built with Streamlit.

---

## ✨ Features

* **Multi-model Implementation:** CNN, ResNet50, and YOLOv8 for robust real-time disease detection and localization.
* **Interactive Web Interface:** User-friendly Streamlit application for model testing and visualization.
* **Image Preprocessing:** Includes augmentation and preprocessing pipelines.
* **Visualizations:** Comprehensive visualization of model accuracy and predictions.

---

## 🚀 Models Overview

* **CNN (Custom):** Convolutional neural network for basic classification with **94.2% accuracy**.
* **ResNet50 (Transfer Learning):** Utilizes a pretrained ResNet50 for deeper feature extraction, achieving **96.8% accuracy**.
* **YOLOv8:** Object detection model for real-time leaf disease localization with **93.5% mAP**.

---

## 📊 Dataset

* **Source:** PlantVillage Dataset (Potato Leaf Subset).
* **Original Images:** 2,152.
* **After Augmentation:** Approximately ~6,000 images.
* **Classes:** Healthy, Early Blight, and Late Blight.

---

## 🛠️ Tech Stack & Frameworks

* **Languages:** Python
* **Core Libraries:** TensorFlow, Keras, PyTorch, OpenCV, scikit-learn, NumPy, Pandas, Matplotlib, TQDM
* **Web Framework:** Streamlit

---

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Potato-Disease-Detection.git](https://github.com/your-username/Potato-Disease-Detection.git)
    cd Potato-Disease-Detection
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app/streamlit_app.py
    ```
4.  **Usage:** Upload a potato leaf image via the web interface and view disease prediction results.

---

## ✅ Performance & Results

* **CNN:** 94.2% accuracy
* **ResNet50:** 96.8% accuracy
* **YOLOv8:** 93.5% mAP (mean Average Precision)

Demonstrates robust performance across all models with effective real-time deployment capabilities.

---

## 📈 Future Enhancements

* Add more disease categories.
* Deploy on cloud (Streamlit Cloud / AWS).
* Integrate with mobile camera for field detection.

---
