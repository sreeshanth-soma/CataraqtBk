import os
import logging # Add logging import
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
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.logger.info("Flask app initialized")

# --- Define Upload Folder using Absolute Path --- #
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'static/uploads')
app.logger.info(f"Upload folder set to: {app.config['UPLOAD_FOLDER']}")
# --- End Absolute Path --- #

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))  # Use env var if available
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session lifetime

# Database configuration - use PostgreSQL on Render, SQLite locally
app.logger.info("Configuring database...")
if os.environ.get('DATABASE_URL'):
    # PostgreSQL database URL from Render
    db_uri = os.environ.get('DATABASE_URL')
    # Fix for Heroku/Render postgres URLs if needed
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    app.logger.info("Using PostgreSQL database")
else:
    # SQLite for local development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
    app.logger.info("Using SQLite database")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Ensure upload folder exists --- #
app.logger.info("Checking upload folder existence...")
try:
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        app.logger.info("Upload folder created.")
except Exception as e:
    app.logger.error(f"Error creating upload folder: {e}")
# --- End folder check --- #

db = SQLAlchemy(app)
app.logger.info("SQLAlchemy initialized")

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
    email = db.Column(db.String(120), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    location = db.Column(db.String(100), nullable=True)
    profile_picture = db.Column(db.String(255), nullable=True, default='default_profile.png')
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    analyses = db.relationship('Analysis', backref='user', lazy=True)

    def __init__(self, username, password, name, email=None, bio=None, location=None):
        self.username = username
        self.password = generate_password_hash(password)
        self.name = name
        self.email = email
        self.bio = bio
        self.location = location

# Analysis model
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

# Create database tables
app.logger.info("Creating database tables (within app context)...")
try:
    with app.app_context():
        db.create_all()
        app.logger.info("Database tables created (or already exist).")
except Exception as e:
    app.logger.error(f"Error creating database tables: {e}")

# Initialize model
app.logger.info("Initializing ML model...")
try:
    model = QuantumEyeDiseaseClassifier()
    model_path = os.path.join(basedir, 'best_model.pth')
    app.logger.info(f"Attempting to load model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    app.logger.info("ML model loaded and set to eval mode successfully.")
except Exception as e:
    app.logger.error(f"Error loading ML model: {e}")
    model = None

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
    hospital_data_file = os.path.join(basedir, 'static/data/hospitals.json')
    try:
        with open(hospital_data_file, 'r') as f:
            hospitals = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        hospitals = []

    # Get top 3 hospitals for recommended consultation
    recommended_hospitals = hospitals[:3] if hospitals else []
    
    if prediction == "Cataract Detected":
        return [
            "Schedule an appointment with an ophthalmologist",
            "Avoid rubbing your eyes",
            "Protect your eyes from UV radiation",
            "Consider wearing sunglasses outdoors",
            "Monitor any changes in vision"
        ], recommended_hospitals
    else:
        return [
            "Continue regular eye check-ups",
            "Maintain good eye hygiene",
            "Take regular breaks when using screens",
            "Stay hydrated for good eye health",
            "Consider using blue light filters on devices"
        ], []

@app.route('/')
def index():
    # Keep this route simple for health checks
    return render_template('home.html', active_page='index')

@app.route('/healthz') # Add a dedicated health check route
def healthz():
    try:
        # Optional: Check DB connection
        # db.session.execute('SELECT 1')
        return "OK", 200
    except Exception as e:
        app.logger.error(f"Health check failed: {e}")
        return "Error", 500

@app.route('/technology-details')
def technology_details():
    return render_template('technology_details.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Check if already logged in first
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try: # This try block will handle form processing errors
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not username or not password:
                flash('Please provide both username and password', 'error')
                return render_template('auth/login.html', active_page='login')

            user = User.query.filter_by(username=username).first()

            if user and check_password_hash(user.password, password):
                try: # Inner try for session logic
                    # Set up permanent session
                    session.permanent = True
                    session['user_id'] = user.id
                    session['username'] = user.username
                    session['user_name'] = user.name
                    if user.profile_picture:
                        session['user_profile_picture'] = user.profile_picture

                    app.logger.info(f"User {username} logged in successfully")
                    flash('Logged in successfully!', 'success')
                    return redirect(url_for('dashboard'))
                except Exception as session_error: # Except for inner try
                    app.logger.error(f"Session error during login: {str(session_error)}")
                    flash('Session error during login. Please try again.', 'error')
                    # Fall through to render login page again
            else:
                app.logger.warning(f"Failed login attempt for user: {username}")
                flash('Invalid username or password', 'error')
                # Fall through to render login page again

        except Exception as form_error: # Except for outer try
            app.logger.error(f"Form processing error: {str(form_error)}")
            flash('Error processing login. Please try again.', 'error')
            # Fall through to render login page again

        # If any error occurred or login failed, render the login page again
        return render_template('auth/login.html', active_page='login') # Unindent to be outside the 'if user...' block but inside 'if POST'

    # For GET requests
    return render_template('auth/login.html', active_page='login') # This should be the final return for GET

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name', username)
        email = request.form.get('email')
        bio = request.form.get('bio')
        location = request.form.get('location')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
        else:
            new_user = User(
                username=username, 
                password=password, 
                name=name,
                email=email,
                bio=bio,
                location=location
            )
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('auth/register.html', active_page='register')

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
        try: # Outer try for the whole POST request handling
            if 'file' not in request.files: # Indent this block
                flash('No file uploaded', 'error')
                return redirect(request.url)
    
            file = request.files['file']
            if file.filename == '': # Indent this block
                flash('No file selected', 'error')
                return redirect(request.url)

            if file and allowed_file(file.filename): # Indent this block
                try: # Inner try for file saving and analysis
                    # Create upload folder if it doesn't exist
                    if not os.path.exists(app.config['UPLOAD_FOLDER']):
                        os.makedirs(app.config['UPLOAD_FOLDER'])

                    filename = secure_filename(file.filename) # Indent correctly
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) # Indent correctly
                    file.save(filepath) # Indent correctly

                    app.logger.info(f"File saved to {filepath}") # Indent correctly

                    # --- Start analysis code --- #
                    try: # Innermost try for the actual ML analysis and DB save
                        # --- Preprocessing uses corrected transform --- #
                        image_pil = Image.open(filepath).convert('RGB')
                        img_tensor = transform(image_pil)
                        img_tensor = img_tensor.unsqueeze(0)
                        # --- End Preprocessing --- #

                        # Make prediction
                        if model is None: # Check if model loaded successfully
                             raise ValueError("Model not loaded, cannot perform analysis.")

                        with torch.no_grad(): # Indent correctly
                            prediction_tensor = model(img_tensor)
                            raw_confidence = float(prediction_tensor[0][0]) # Model's raw output (0..1)

                            # Print debug info about confidence levels
                            app.logger.info(f"Raw confidence value for cataract: {raw_confidence}")

                            # Set threshold to 40% (0.4) for cataract detection
                            cataract_threshold = 0.4
                            result = "Cataract Detected" if raw_confidence > cataract_threshold else "No Cataract Detected"
                            app.logger.info(f"Final prediction: {result} (threshold={cataract_threshold})")

                            # Store raw confidence - higher means more likely cataract
                            cataract_likelihood = raw_confidence

                        # Save analysis to history - Store the calculated cataract likelihood
                        try: # Try for database save
                            analysis = Analysis(
                                user_id=session['user_id'],
                                image_path=filename,
                                prediction=result,
                                confidence=cataract_likelihood # Store likelihood of cataract
                            )
                            db.session.add(analysis)
                            db.session.commit()
                            app.logger.info("Analysis saved to database")
                        except Exception as db_error: # Except for database save
                            app.logger.error(f"Database error: {str(db_error)}")
                            db.session.rollback()
                            # Continue even if database save fails, but maybe flash a warning?
                            flash('Analysis complete, but failed to save to history.', 'warning')

                        # Get recommendations based on prediction and cataract likelihood
                        recommendations, recommended_hospitals = get_recommendations(result, cataract_likelihood) # Indent correctly

                        # --- Rest of visualization code --- #
                        # ...
                        # --- End visualization code --- #

                        # --- Prepare results dictionary for template --- #
                        results_data = { # Indent correctly
                            'prediction': result,
                            'confidence': cataract_likelihood,
                            'recommendations': recommendations,
                            'recommended_hospitals': recommended_hospitals,
                            'visualizations': {} # Initialize empty in case visualizations fail
                        }

                        # Try to generate visualizations, but don't fail if they can't be generated
                        try: # Try for visualizations
                            # Rest of visualization code as before
                            img_cv_rgb = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                            img_display_resized = cv2.resize(img_cv_rgb, (224, 224))

                            # Edge detection
                            img_gray = cv2.cvtColor(img_display_resized, cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(img_gray, 100, 200)
                            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

                            # Activation map
                            activation_map = None
                            overlay = None
                            # Check if model has the method AND is not None
                            if model is not None and hasattr(model, 'get_activation_map'):
                                activation_map_raw = model.get_activation_map(img_tensor)
                                activation_map_resized = cv2.resize(activation_map_raw, (224, 224))
                                heatmap_vis = np.uint8(255 * activation_map_resized)
                                heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                                heatmap_vis_rgb = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB)
                                activation_map = heatmap_vis_rgb

                                # Create overlay
                                overlay_vis = cv2.addWeighted(cv2.cvtColor(img_display_resized, cv2.COLOR_BGR2RGB), 0.6, heatmap_vis_rgb, 0.4, 0)
                                overlay = overlay_vis

                            def image_to_base64(img_array_rgb):
                                if img_array_rgb is None: return None
                                img_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
                                _, buffer = cv2.imencode('.png', img_bgr)
                                return base64.b64encode(buffer).decode('utf-8')

                            visualizations = {
                                'original': image_to_base64(cv2.cvtColor(img_display_resized, cv2.COLOR_BGR2RGB)),
                                'heatmap': image_to_base64(activation_map),
                                'overlay': image_to_base64(overlay),
                                'edges': image_to_base64(edges_rgb)
                            }

                            results_data['visualizations'] = visualizations
                        except Exception as viz_error: # Except for visualizations
                            app.logger.error(f"Visualization error: {str(viz_error)}")
                            # Continue without visualizations if they fail

                        # --- Render HTML template --- #
                        return render_template('analyze_results.html', results=results_data, active_page='analyze')

                    except Exception as analysis_error: # Except for ML analysis/DB save
                        app.logger.error(f"Analysis error: {str(analysis_error)}")
                        flash(f"Error analyzing image: {str(analysis_error)}", "error")
                        return redirect(request.url) # Redirect back to upload page on analysis error

                except Exception as file_error: # Except for file saving/analysis (inner try)
                    app.logger.error(f"File handling error: {str(file_error)}")
                    flash(f"Error processing file: {str(file_error)}", "error")
                    return redirect(request.url) # Redirect back to upload page on file error
            else:
                 # Handle invalid file type (this block was indented incorrectly before)
                 flash("Invalid file type. Please upload JPG, PNG, or JPEG files.", "error")
                 return redirect(request.url)

        except Exception as e: # Except for the whole POST request (outer try)
            app.logger.error(f"Unexpected error in analyze POST: {str(e)}")
            flash(f"An unexpected error occurred. Please try again.", "error")
            return redirect(request.url) # Redirect back to upload page on general error

    # For GET request, just render the upload form (This line was incorrectly indented before)
    return render_template('analyze.html', active_page='analyze')

@app.route('/history')
@login_required
def history():
    # Get all analyses with pagination
    page = request.args.get('page', 1, type=int)
    per_page = 9 # Display 9 analyses per page
    user_analyses = Analysis.query.filter_by(user_id=session['user_id']).order_by(Analysis.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('history.html', analyses=user_analyses, active_page='history')

@app.route('/profile', methods=['GET'])
@login_required
def profile():
    user = User.query.get(session['user_id'])
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    # Calculate stats
    total_analyses = Analysis.query.filter_by(user_id=user.id).count()
    cataract_detected = Analysis.query.filter_by(user_id=user.id, prediction='Cataract Detected').count()
    healthy_detected = Analysis.query.filter_by(user_id=user.id, prediction='No Cataract Detected').count()
    
    # Handle missing profile picture more gracefully
    profile_pic_url = url_for('serve_upload', filename=user.profile_picture if user.profile_picture else 'default_profile.png')
    
    stats = {
        'total_analyses': total_analyses,
        'cataract_detected': cataract_detected,
        'no_cataract': healthy_detected,
        'cataract_percentage': (cataract_detected / total_analyses * 100) if total_analyses > 0 else 0
    }
    return render_template('profile.html', user=user, stats=stats, profile_pic_url=profile_pic_url, active_page='profile')

@app.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        bio = request.form.get('bio')
        location = request.form.get('location')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        
        if not check_password_hash(user.password, current_password):
            flash('Current password is incorrect', 'error')
            return redirect(url_for('edit_profile'))
        
        # Update user information
        user.name = name
        user.email = email
        user.bio = bio
        user.location = location
        
        # Handle profile picture upload
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(f"profile_{user.id}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Delete previous profile picture if it's not the default
                if user.profile_picture and user.profile_picture != 'default_profile.png':
                    old_filepath = os.path.join(app.config['UPLOAD_FOLDER'], user.profile_picture)
                    if os.path.exists(old_filepath):
                        os.remove(old_filepath)
                
                # Update user's profile picture
                user.profile_picture = filename
                
                # Update session
                session['user_profile_picture'] = filename
        
        # Update password if provided
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
    recommendations, recommended_hospitals = get_recommendations(analysis.prediction, analysis.confidence)

    # Note: Image preview is currently unavailable because original images are deleted after analysis.
    return render_template('analysis_detail.html', analysis=analysis, recommendations=recommendations, recommended_hospitals=recommended_hospitals, active_page='history')

@app.route('/uploads/<path:filename>')
@login_required
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/hospitals')
def hospitals():
    """Display a list of eye hospitals that treat cataracts."""
    # Load hospital data from JSON file
    hospital_data_file = os.path.join(basedir, 'static/data/hospitals.json')
    try:
        with open(hospital_data_file, 'r') as f:
            hospitals_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        hospitals_list = []  # Fallback if file doesn't exist or is invalid
    
    return render_template('hospitals.html', hospitals=hospitals_list, active_page='hospitals')

@app.route('/set-theme', methods=['POST'])
def set_theme():
    try:
        # Try to get JSON data
        if request.is_json:
            theme = request.json.get('theme')
        else:
            # Try to get form data if not JSON
            theme = request.form.get('theme')
            
        if not theme:
            # Try to get from query params as last resort
            theme = request.args.get('theme')
            
        if theme in ['light', 'dark']:
            # Store in session if available
            try:
                session['theme'] = theme
            except Exception as e:
                app.logger.error(f"Session error: {str(e)}")
                
            # Return response with cookie
            response = jsonify({'success': True, 'theme': theme})
            response.set_cookie('theme', theme, max_age=31536000, samesite='Lax', secure=False)
            return response
            
        return jsonify({'success': False, 'error': 'Invalid theme'}), 400
    except Exception as e:
        app.logger.error(f"Theme setting error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Create default profile picture if it doesn't exist
def create_default_profile_picture():
    default_profile_path = os.path.join(app.config['UPLOAD_FOLDER'], 'default_profile.png')
    if not os.path.exists(default_profile_path):
        # Create a simple default profile picture
        img = np.ones((200, 200, 3), dtype=np.uint8) * 200  # Light gray background
        cv2.circle(img, (100, 100), 80, (70, 130, 180), -1)  # Blue circle
        cv2.putText(img, "User", (65, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(default_profile_path, img)

# Setup function to run initialization tasks
def setup_app():
    app.logger.info("Running setup_app function...")
    try:
        # Ensure upload folder exists (already checked above, but good to be safe)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            try:
                os.makedirs(app.config['UPLOAD_FOLDER'])
                app.logger.info(f"Created upload folder in setup_app: {app.config['UPLOAD_FOLDER']}")
            except Exception as folder_error:
                app.logger.error(f"Error creating upload folder in setup_app: {str(folder_error)}")
        
        # Create default profile picture
        try:
            create_default_profile_picture()
            app.logger.info("Default profile picture check/creation complete in setup_app.")
        except Exception as pic_error:
            app.logger.error(f"Error creating default profile picture in setup_app: {str(pic_error)}")
    except Exception as e:
        app.logger.error(f"Error during setup_app execution: {str(e)}")
    app.logger.info("setup_app function finished.")

@app.before_request
def check_theme():
    if 'theme' not in session and 'theme' in request.cookies:
        session['theme'] = request.cookies.get('theme')

if __name__ == '__main__':
    try:
        with app.app_context():
            print("Attempting to create database tables...")
            db.create_all()
            print("Database tables check/creation complete.")
            print("Running setup_app...")
            setup_app()  # Run setup tasks
            print("setup_app finished.")
            
        print("Starting Flask development server...")
        # Use debug=False for testing potential startup hangs, True for more error info
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"\\n{'*'*20} ERROR DURING STARTUP {'*'*20}", file=sys.stderr)
        print(f"An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"{'*'*58}", file=sys.stderr)
