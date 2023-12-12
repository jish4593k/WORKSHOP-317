import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

# Load the pre-trained ResNet50 model
model = resnet50(pretrained=True)
model.eval()

# Define a function to preprocess images for the model
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Function to predict class labels using the pre-trained model
def predict_image(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Iterate through the downloaded images and make predictions
def classify_images():
    for product in get_product_name():
        product_directory = f"./images/{product}"
        for image_file in os.listdir(product_directory):
            image_path = os.path.join(product_directory, image_file)
            predicted_class = predict_image(image_path)
            print(f"Product: {product}, Image: {image_file}, Predicted Class: {predicted_class}")

# Call the classify_images function
classify_images()
