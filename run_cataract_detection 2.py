import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os, glob, cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_custom_objects

# Set random seed for reproducibility
SEED = 53
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Set image directories
IMG_ROOT = 'Datasets/eye_diseases_classification/Proc/Datasets/eye_diseases_classification/'
IMG_DIR = [
    os.path.join(IMG_ROOT, 'normal'),
    os.path.join(IMG_ROOT, 'cataract')
]

def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

def create_dataframe():
    # First, get all image files
    filepaths = glob.glob(IMG_ROOT + '*/*.jpg')  # Only look for .jpg files
    print(f"Found {len(filepaths)} files")
    print("Sample file paths:")
    for path in filepaths[:5]:
        print(path)
    
    # Create DataFrame with the correct size
    df = pd.DataFrame({
        'paths': filepaths,
        'cataract': [''] * len(filepaths)
    })
    
    # Set labels based on directory names
    df['cataract'] = df['paths'].apply(lambda x: 'cataract' if 'cataract' in x else 'normal')
    
    return df

def main():
    # Set random seed
    seed_everything(SEED)
    
    # Create dataframe
    print("Creating dataset dataframe...")
    df = create_dataframe()
    print("\nDataset statistics:")
    print(df['cataract'].value_counts())
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    
    print("\nTraining set size:", len(train_df))
    print("Test set size:", len(test_df))
    
    # Create data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='paths',
        y_col='cataract',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col='paths',
        y_col='cataract',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='categorical'
    )
    
    print("\nStarting model training...")
    # Create and train model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator
    )
    
    # Save the model and training history
    model.save('cataract_detection_model.h5')
    np.save('training_history.npy', history.history)
    
    print("\nEvaluating model...")
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nTraining history plot saved as 'training_history.png'")

if __name__ == "__main__":
    main() 