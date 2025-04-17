import numpy as np

# Create sample training history data
epochs = 20
history = {
    'accuracy': np.linspace(0.6, 0.95, epochs) + np.random.normal(0, 0.02, epochs),
    'val_accuracy': np.linspace(0.55, 0.92, epochs) + np.random.normal(0, 0.03, epochs),
    'loss': np.exp(-np.linspace(0, 2, epochs)) + np.random.normal(0, 0.05, epochs),
    'val_loss': np.exp(-np.linspace(0, 1.8, epochs)) + np.random.normal(0, 0.06, epochs)
}

# Ensure values are between 0 and 1
history['accuracy'] = np.clip(history['accuracy'], 0, 1)
history['val_accuracy'] = np.clip(history['val_accuracy'], 0, 1)
history['loss'] = np.clip(history['loss'], 0, 1)
history['val_loss'] = np.clip(history['val_loss'], 0, 1)

# Save to file
np.save('training_history.npy', history)
print("Training history data created successfully!") 