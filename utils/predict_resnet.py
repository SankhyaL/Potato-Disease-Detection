import torch
from torchvision import transforms
from PIL import Image

# Load model (you must define same architecture before loading state dict)
from torchvision import models

# Example: ResNet18 architecture
resnet_model = models.resnet18()
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(num_ftrs, 4)  # 4 classes (including desktop.ini)
resnet_model.load_state_dict(torch.load("models/Potato_Disease_CNNresnet_Classifier.pth", map_location=torch.device("cpu")))
resnet_model.eval()

CLASSES = ['desktop.ini', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']

def predict(image_file):
    """Predict class using the ResNet model"""
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = resnet_model(img_tensor)
        _, pred = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][pred].item() * 100

    return f"{CLASSES[pred.item()]} ({confidence:.2f}% confidence)"
