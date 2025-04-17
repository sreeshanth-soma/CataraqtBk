import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
import numpy as np
from PIL import Image

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=8, n_layers=3):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Parameters for each layer
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 3))  # 3 parameters per qubit per layer
        self.phi = nn.Parameter(torch.randn(n_layers, n_qubits, n_qubits))  # Entanglement parameters
        
    def quantum_circuit(self, x):
        # Create quantum circuit
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Take first n_qubits elements and ensure they're scalar values
        x_reduced = x[:self.n_qubits].flatten()
        
        # Initial state preparation with more sophisticated encoding
        for i in range(self.n_qubits):
            # Apply Hadamard gate for superposition
            qc.h(i)
            # Encode classical data with rotation
            qc.rx(float(x_reduced[i]), i)
            qc.ry(float(x_reduced[i]), i)
        
        # Add variational layers with enhanced entanglement
        for layer in range(self.n_layers):
            # Single qubit rotations
            for i in range(self.n_qubits):
                theta = self.theta[layer, i].detach().numpy()
                qc.rx(float(theta[0]), i)
                qc.ry(float(theta[1]), i)
                qc.rz(float(theta[2]), i)
            
            # Enhanced entangling layer with parameterized gates
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    phi = self.phi[layer, i, j].detach().numpy()
                    qc.cx(i, j)
                    qc.rz(float(phi), j)
                    qc.cx(i, j)
        
        return qc
    
    def forward(self, x):
        # Handle batched input
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            # Process each sample in the batch
            x_i = x[i]
            if len(x_i.shape) > 1:
                x_i = x_i.flatten()
            
            # Create and execute quantum circuit
            qc = self.quantum_circuit(x_i.detach().numpy())
            backend = Aer.get_backend('statevector_simulator')
            job = backend.run(qc)
            result = job.result()
            
            # Get statevector and convert to torch tensor
            statevector = result.get_statevector()
            output = torch.from_numpy(np.abs(statevector)).float()
            outputs.append(output)
        
        # Stack outputs into a batch
        return torch.stack(outputs)

class QuantumEyeDiseaseClassifier(nn.Module):
    def __init__(self):
        super(QuantumEyeDiseaseClassifier, self).__init__()
        
        # Enhanced CNN for feature extraction with more capacity
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Feature compression before quantum layer
        self.feature_compression = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # Quantum layer with more qubits and layers
        self.quantum_layer = QuantumLayer(n_qubits=8, n_layers=3)
        
        # Enhanced classifier with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.confidence = 0.0
        
    def forward(self, x):
        # Extract features using CNN
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # Compress features before quantum layer
        compressed_features = self.feature_compression(features)
        
        # Process through quantum layer
        quantum_features = self.quantum_layer(compressed_features)
        
        # Final classification
        output = self.classifier(quantum_features)
        
        # Ensure output has shape [batch_size, 1]
        if len(output.shape) == 1:
            output = output.unsqueeze(1)
            
        return output
        
    def preprocess_image(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image.unsqueeze(0)
    
    def predict(self, image):
        self.eval()
        with torch.no_grad():
            output = self.forward(image)
            self.confidence = float(output.item())
            return 1 if self.confidence > 0.5 else 0
    
    def get_confidence(self):
        return self.confidence
    
    def get_activation_map(self, x):
        self.eval()
        with torch.no_grad():
            # Get feature maps from the last convolutional layer
            for i, layer in enumerate(self.feature_extractor):
                x = layer(x)
                if isinstance(layer, nn.Conv2d) and i > 20:  # Get activation from last conv layer
                    features = x
            
            # Global average pooling
            weights = torch.mean(features, dim=(2, 3))
            
            # Weight the channels by their average values
            batch_size, n_channels, h, w = features.shape
            weighted_features = torch.zeros((batch_size, h, w))
            
            for i in range(n_channels):
                weighted_features += weights[:, i].view(-1, 1, 1) * features[:, i, :, :]
            
            # Normalize to [0, 1]
            activation_map = F.relu(weighted_features)
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
            
            return activation_map[0].numpy()  # Return the first item in batch
    
    def load_pretrained(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = QuantumEyeDiseaseClassifier()
    
    # Example image path
    image_path = "path_to_your_image.jpg"
    
    # Preprocess and predict
    image = classifier.preprocess_image(image_path)
    prediction = classifier.predict(image)
    
    # Print results
    print(f"Prediction: {'Cataract' if prediction == 1 else 'Normal'}")
    print(f"Confidence: {classifier.get_confidence():.2%}") 