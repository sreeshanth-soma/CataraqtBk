# Quantum Eye Disease Classifier

A web application that uses quantum-classical hybrid machine learning to detect cataracts in eye images.

## Features

- Modern, responsive web interface
- Drag-and-drop image upload
- Real-time image preview
- Quantum-classical hybrid model for cataract detection
- Confidence score for predictions

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://localhost:5000
```

3. Upload an eye image by either:

   - Dragging and dropping an image onto the upload area
   - Clicking the "Browse Files" button to select an image

4. The application will process the image and display:
   - The prediction (Cataract or Normal)
   - The confidence score of the prediction

## Project Structure

```
.
├── app.py                 # Flask application
├── templates/
│   └── index.html        # Web interface template
├── static/
│   └── uploads/          # Temporary storage for uploaded images
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Technical Details

The application uses:

- Flask for the web server
- Qiskit for quantum computing components
- PyTorch for classical neural network components
- TailwindCSS for styling

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
