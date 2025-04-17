#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p Datasets/eye_diseases_classification/Proc

# Extract dataset if RAR file exists
if [ -f "Datasets/Datasets.rar" ]; then
    unar -o Datasets/eye_diseases_classification/Proc Datasets/Datasets.rar
fi

# Set permissions
chmod +x run_cataract_detection.py
chmod +x train_model.py

echo "Setup completed successfully!"
echo "To activate the environment, run: source venv/bin/activate" 