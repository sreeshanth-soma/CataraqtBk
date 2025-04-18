import os
import shutil
import time
import importlib
import sys

print("=== COMPLETE APPLICATION RESET ===")

# 1. Delete all SQLAlchemy cache files
print("Removing __pycache__ directories...")
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        pycache_dir = os.path.join(root, '__pycache__')
        print(f"Removing {pycache_dir}")
        shutil.rmtree(pycache_dir)

# 2. Delete the database file
print("Removing database file...")
db_path = 'users.db'
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Removed {db_path}")
else:
    print(f"{db_path} not found")

# 3. Creating directory for profile pictures
print("Ensuring uploads directory exists...")
uploads_dir = os.path.join('static', 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
    print(f"Created {uploads_dir}")

# 4. Import app without instantiating the database
print("Setting up new database with correct schema...")
# Force reload of app module
if 'app' in sys.modules:
    del sys.modules['app']

# Import and create tables
from app import app, db, create_default_profile_picture

# 5. Create tables
with app.app_context():
    db.create_all()
    print("Created database tables with current schema")
    
    # Create default profile picture
    create_default_profile_picture()
    print("Created default profile picture")

print("\nReset complete! Your database has been rebuilt with the correct schema.")
print("You will need to register new user accounts.")
print("\nPlease restart your Flask application now.") 