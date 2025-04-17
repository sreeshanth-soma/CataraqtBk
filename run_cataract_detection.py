import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from quantum_eye_disease_classifier import QuantumEyeDiseaseClassifier
import matplotlib.pyplot as plt
import cv2
import os

def preprocess_image(image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

def visualize_prediction(image_path, prediction,  confidence):
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(10, 5))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot prediction
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f'Prediction: {"Cataract" if prediction == 1 else "Normal"}\nConfidence: {confidence:.2%}',
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.close()

def run_detection(image_path, model_path='best_model.pth'):
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return None, None
            
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            print("Please run 'python train_model.py' first to train a model.")
            return None, None
            
        # Initialize model
        model = QuantumEyeDiseaseClassifier()
        
        # Load trained weights with error handling
        try:
            checkpoint = torch.load(model_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("The model file may be incompatible with the current model architecture.")
            print("Try running 'python train_model.py' to train a new model.")
            return None, None
            
        model.eval()
        
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            prediction = model.predict(image_tensor)
            confidence = model.get_confidence()
        
        # Visualize results
        visualize_prediction(image_path, prediction, confidence)
        
        return prediction, confidence
        
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure the image path is correct")
        print("2. Check if the model file exists")
        print("3. Verify that all required packages are installed")
        print("4. Try running 'python train_model.py' first to train a new model")
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run cataract detection on an image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the trained model')
    
    args = parser.parse_args()
    
    prediction, confidence = run_detection(args.image_path, args.model_path)
    
    if prediction is not None:
        print(f"\nPrediction: {'Cataract' if prediction == 1 else 'Normal'}")
        print(f"Confidence for the prediction: {confidence:.2%}")
        print("\nVisualization saved as 'prediction_result.png'") 