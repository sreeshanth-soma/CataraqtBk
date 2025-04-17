import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import torch
from quantum_eye_disease_classifier import QuantumEyeDiseaseClassifier
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from functools import wraps
from datetime import timedelta
import cv2
import base64
import sys

app = Flask(__name__)

# --- Define Upload Folder using Absolute Path --- #
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'static/uploads')
# --- End Absolute Path --- #

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.urandom(24)  # For session management
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session lifetime
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Ensure upload folder exists --- #
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
# --- End folder check --- #

db = SQLAlchemy(app)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    analyses = db.relationship('Analysis', backref='user', lazy=True)

    def __init__(self, username, password, name):
        self.username = username
        self.password = generate_password_hash(password)
        self.name = name

# Analysis model
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

# Create database tables
with app.app_context():
    db.create_all()

# Initialize model
model = QuantumEyeDiseaseClassifier()
checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# --- Corrected PyTorch Transform (Matching Model's Preprocessing) --- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor() # Converts to [0, 1] range, matching /255.0 implicitly
])
# --- End Corrected Transform --- #

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_recommendations(prediction, confidence):
    if prediction == "Cataract Detected":
        return [
            "Schedule an appointment with an ophthalmologist",
            "Avoid rubbing your eyes",
            "Protect your eyes from UV radiation",
            "Consider wearing sunglasses outdoors",
            "Monitor any changes in vision"
        ]
    else:
        return [
            "Continue regular eye check-ups",
            "Maintain good eye hygiene",
            "Take regular breaks when using screens",
            "Stay hydrated for good eye health",
            "Consider using blue light filters on devices"
        ]

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session.permanent = True
            session['user_id'] = user.id
            session['username'] = user.username
            session['user_name'] = user.name
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name', username)
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
        else:
            new_user = User(username=username, password=password, name=name)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('auth/register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get only the 5 most recent analyses for the dashboard
    user_id = session.get('user_id')
    recent_analyses = Analysis.query.filter_by(user_id=user_id)\
        .order_by(Analysis.created_at.desc())\
        .limit(5)\
        .all()
    
    # Calculate some basic statistics
    total_analyses = Analysis.query.filter_by(user_id=user_id).count()
    cataract_detected = Analysis.query.filter_by(
        user_id=user_id, 
        prediction="Cataract Detected"
    ).count()
    
    stats = {
        'total_analyses': total_analyses,
        'cataract_detected': cataract_detected,
        'no_cataract': total_analyses - cataract_detected,
        'cataract_percentage': (cataract_detected / total_analyses * 100) if total_analyses > 0 else 0
    }
    
    return render_template('dashboard.html', recent_analyses=recent_analyses, stats=stats)

@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # --- Preprocessing uses corrected transform --- #
                image_pil = Image.open(filepath).convert('RGB') 
                img_tensor = transform(image_pil)
                img_tensor = img_tensor.unsqueeze(0)
                # --- End Preprocessing --- #

                # Make prediction
                with torch.no_grad():
                    prediction_tensor = model(img_tensor) 
                    raw_confidence = float(prediction_tensor[0][0]) # Model's raw output (0..1)
                    
                    # --- Final Interpretation Attempt: Assume 0=Cataract, 1=Normal --- #
                    result = "Cataract Detected" if raw_confidence < 0.5 else "No Cataract Detected"
                    # Calculate a score where higher means more likely cataract for storage/display
                    cataract_likelihood = 1.0 - raw_confidence 
                    # --- End Interpretation --- #
                
                # Save analysis to history - Store the calculated cataract likelihood
                analysis = Analysis(
                    user_id=session['user_id'],
                    image_path=filename, 
                    prediction=result,
                    confidence=cataract_likelihood # Store likelihood of cataract
                )
                db.session.add(analysis)
                db.session.commit()
                
                # Get recommendations based on prediction and cataract likelihood
                recommendations = get_recommendations(result, cataract_likelihood)
                
                # --- Restore Visualizations (Attempt) --- # 
                # Need original image as numpy array for cv2 operations
                # Convert PIL image used for transform back to numpy (or reload with cv2)
                img_cv_rgb = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                img_display_resized = cv2.resize(img_cv_rgb, (224, 224)) # Ensure consistent size

                # Edge detection on the resized display image
                img_gray = cv2.cvtColor(img_display_resized, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(img_gray, 100, 200)
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                
                # --- Try getting activation map again --- #
                # This assumes 'get_activation_map' exists on your PyTorch model
                # and works with the 3-channel input. Might need adjustment.
                activation_map = None
                overlay = None
                try:
                    if hasattr(model, 'get_activation_map'):
                        activation_map_raw = model.get_activation_map(img_tensor) # Pass the tensor
                        # Post-process activation map (resize, colormap)
                        activation_map_resized = cv2.resize(activation_map_raw, (224, 224))
                        heatmap_vis = np.uint8(255 * activation_map_resized)
                        heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                        heatmap_vis_rgb = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB)
                        activation_map = heatmap_vis_rgb # Store for output
                        
                        # Create overlay
                        overlay_vis = cv2.addWeighted(cv2.cvtColor(img_display_resized, cv2.COLOR_BGR2RGB), 0.6, heatmap_vis_rgb, 0.4, 0)
                        overlay = overlay_vis # Store for output
                    else:
                        print("Model does not have get_activation_map method.", file=sys.stderr)
                except Exception as viz_error:
                     print(f"Error generating activation map: {viz_error}", file=sys.stderr)
                # --- End Try Activation Map --- #
                
                def image_to_base64(img_array_rgb):
                    if img_array_rgb is None: return None
                    # Convert RGB (numpy) to BGR for cv2.imencode
                    img_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR) 
                    _, buffer = cv2.imencode('.png', img_bgr)
                    return base64.b64encode(buffer).decode('utf-8')
                
                visualizations = {
                    'original': image_to_base64(cv2.cvtColor(img_display_resized, cv2.COLOR_BGR2RGB)), # Ensure original is RGB for display
                    'heatmap': image_to_base64(activation_map), 
                    'overlay': image_to_base64(overlay), 
                    'edges': image_to_base64(edges_rgb)
                }
                # --- End Restore Visualizations --- #

                return jsonify({
                    'prediction': result,
                    'confidence': cataract_likelihood, # Return likelihood of cataract
                    'recommendations': recommendations,
                    'visualizations': visualizations
                })
                
            except Exception as e:
                print(f"Error during analysis: {e}", file=sys.stderr) # Add logging
                return jsonify({'error': str(e)}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    return render_template('analyze.html')

@app.route('/history')
@login_required
def history():
    # Get all analyses with pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    user_id = session.get('user_id')
    
    # Get paginated analyses
    analyses = Analysis.query.filter_by(user_id=user_id)\
        .order_by(Analysis.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('history.html', analyses=analyses)

@app.route('/profile', methods=['GET'])
@login_required
def profile():
    user = User.query.get(session['user_id'])
    
    # Calculate analysis statistics for the user
    total_analyses = Analysis.query.filter_by(user_id=user.id).count()
    cataract_detected = Analysis.query.filter_by(
        user_id=user.id, 
        prediction="Cataract Detected"
    ).count()
    
    stats = {
        'total_analyses': total_analyses,
        'cataract_detected': cataract_detected,
        'no_cataract': total_analyses - cataract_detected,
        'cataract_percentage': (cataract_detected / total_analyses * 100) if total_analyses > 0 else 0
    }
    
    return render_template('profile.html', user=user, stats=stats)

@app.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        name = request.form.get('name')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        
        if not check_password_hash(user.password, current_password):
            flash('Current password is incorrect', 'error')
            return redirect(url_for('edit_profile'))
        
        user.name = name
        if new_password:
            user.password = generate_password_hash(new_password)
        
        db.session.commit()
        session['user_name'] = name
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile_edit.html', user=user)

@app.route('/analysis/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Security check: Ensure the analysis belongs to the logged-in user
    if analysis.user_id != session.get('user_id'):
        flash('You do not have permission to view this analysis.', 'error')
        return redirect(url_for('history'))

    # Get recommendations based on the stored prediction and confidence
    recommendations = get_recommendations(analysis.prediction, analysis.confidence)

    # Note: Image preview is currently unavailable because original images are deleted after analysis.
    return render_template('analysis_detail.html', analysis=analysis, recommendations=recommendations)

@app.route('/uploads/<path:filename>')
@login_required
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # REMOVED folder check from here 
    # if not os.path.exists(app.config['UPLOAD_FOLDER']):
    #     os.makedirs(app.config['UPLOAD_FOLDER'])
        
    with app.app_context():
        db.create_all()
        
    app.run(debug=True, host='0.0.0.0', port=5001) 