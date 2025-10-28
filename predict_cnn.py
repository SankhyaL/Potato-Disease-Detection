import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained CNN model once at import
cnn_model = tf.keras.models.load_model("models/potato_cnn_model.h5")

# Class labels (adjust if your dataset has different class names)
CLASSES = ['Early blight', 'Late blight', 'healthy']

def predict(image_file):
    """Predict class using the CNN model"""
    image = Image.open(image_file).convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = cnn_model.predict(img_array)
    class_index = np.argmax(preds)
    confidence = round(float(np.max(preds)) * 100, 2)

    return f"{CLASSES[class_index]} ({confidence}% confidence)"
