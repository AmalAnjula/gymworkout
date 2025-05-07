import os
import torch
from pathlib import Path

# Function to check if GPU is available
def check_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# Set paths for dataset and model configurations
dataset_path = 'dataset.yaml'  # Replace with your actual dataset.yaml path
model_weights = 'yolov5s.pt'  # Pre-trained weights for transfer learning, or start from scratch with 'yolov5s.yaml'

# Set the parameters for training
img_size = 640  # Image size for training
batch_size = 16  # Batch size
epochs = 100  # Number of epochs
device = check_device()

# Ensure that the necessary directories exist
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset YAML not found at {dataset_path}")

# Train the YOLOv5 model using the Ultralytics YOLOv5 framework
os.system(f'python yolov5/train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data {dataset_path} --weights {model_weights} --device {device}')
