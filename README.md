🥔 Potato Leaf Disease Detection using Deep Learning

A deep learning–based web application that detects and classifies potato leaf diseases using CNN, ResNet50, and YOLOv8 models. The project assists farmers and researchers by enabling real-time disease identification through a simple web interface built with Streamlit.

🚀 Features

Multi-model implementation: CNN, ResNet50, and YOLOv8

Real-time detection and localization of diseased leaf regions

Interactive Streamlit web interface for model testing

Image augmentation and preprocessing pipeline for improved accuracy

Visualization of model accuracy and predictions

🧠 Models Overview
Model	Description	Accuracy / mAP
CNN	Custom convolutional neural network for basic classification	94.2%
ResNet50	Transfer learning with pretrained ResNet50 for deeper feature extraction	96.8%
YOLOv8	Object detection model for real-time leaf disease localization	93.5% (mAP)
📂 Dataset

Source: PlantVillage Dataset (Potato Leaf Subset)

Original Images: 2,152

After Augmentation: ~6,000 images

Classes include Healthy, Early Blight, and Late Blight.

🛠️ Tech Stack

Languages & Frameworks: Python, Streamlit
Libraries Used: TensorFlow, Keras, PyTorch, OpenCV, scikit-learn, NumPy, Pandas, Matplotlib, TQDM

⚙️ Setup Instructions

Clone the repository:

git clone https://github.com/<your-username>/Potato-Disease-Detection.git
cd Potato-Disease-Detection


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app/streamlit_app.py


Upload a potato leaf image and view disease prediction results.

📈 Results

CNN: 94.2%

ResNet50: 96.8%

YOLOv8: 93.5% (mAP)

Demonstrated robust performance across all models with effective real-time deployment.

📚 Future Enhancements

Add more disease categories

Deploy on cloud (Streamlit Cloud / AWS)

Integrate with mobile camera for field detection
