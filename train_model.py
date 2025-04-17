import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from quantum_eye_disease_classifier import QuantumEyeDiseaseClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class EyeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

def load_dataset(data_dir):
    cataract_dir = os.path.join(data_dir, 'cataract')
    normal_dir = os.path.join(data_dir, 'normal')
    
    # Get all image paths and labels
    cataract_paths = [os.path.join(cataract_dir, f) for f in os.listdir(cataract_dir) if f.endswith('.jpg')]
    normal_paths = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.jpg')]
    
    image_paths = cataract_paths + normal_paths
    labels = [1] * len(cataract_paths) + [0] * len(normal_paths)
    
    return image_paths, labels

def train_model(model, train_loader, val_loader, num_epochs=100, device='cpu'):
    criterion = nn.BCELoss()
    
    # Adjusted learning rates
    feature_optimizer = optim.Adam(model.feature_extractor.parameters(), lr=1e-5)
    quantum_optimizer = optim.Adam(model.quantum_layer.parameters(), lr=1e-4)
    classifier_optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
    
    # Learning rate schedulers with more patience
    feature_scheduler = optim.lr_scheduler.ReduceLROnPlateau(feature_optimizer, mode='max', factor=0.5, patience=7, verbose=True)
    quantum_scheduler = optim.lr_scheduler.ReduceLROnPlateau(quantum_optimizer, mode='max', factor=0.5, patience=7, verbose=True)
    classifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, mode='max', factor=0.5, patience=7, verbose=True)
    
    train_losses = []
    val_accuracies = []
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Zero all optimizers
            feature_optimizer.zero_grad()
            quantum_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Reshape labels to match outputs
            labels = labels.view(-1, 1)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Clip gradients with a smaller threshold
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # Step all optimizers
            feature_optimizer.step()
            quantum_optimizer.step()
            classifier_optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                
                total += labels.size(0)
                correct += (predicted.view(-1) == labels).sum().item()
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'feature_optimizer_state_dict': feature_optimizer.state_dict(),
                'quantum_optimizer_state_dict': quantum_optimizer.state_dict(),
                'classifier_optimizer_state_dict': classifier_optimizer.state_dict(),
                'accuracy': accuracy,
            }, 'best_model.pth')
        
        # Step schedulers
        feature_scheduler.step(accuracy)
        quantum_scheduler.step(accuracy)
        classifier_scheduler.step(accuracy)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    return train_losses, val_accuracies

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Enhanced data augmentation and preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    data_dir = 'Datasets/eye_diseases_classification/Proc/Datasets/eye_diseases_classification'
    image_paths, labels = load_dataset(data_dir)
    
    # Split dataset with stratification
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = EyeDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = EyeDataset(val_paths, val_labels, transform=val_transform)
    
    # Create data loaders with more workers
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = QuantumEyeDiseaseClassifier().to(device)
    
    # Train model with more epochs
    print("Starting training...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=100, device=device
    )
    
    print(f"Best validation accuracy: {max(val_accuracies):.2f}%")
    print("Training completed! Model saved as 'best_model.pth'")
    print("Training curves saved as 'training_curves.png'")

if __name__ == "__main__":
    main() 