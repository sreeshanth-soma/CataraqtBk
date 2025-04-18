import os
from app import app, db, User, Analysis

with app.app_context():
    # Confirm we're working with the right database
    db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
    print(f"Working with database at: {db_path}")
    
    # Drop all existing tables
    print("Dropping all tables...")
    db.drop_all()
    
    # Recreate all tables from the models
    print("Creating new tables from SQLAlchemy models...")
    db.create_all()
    
    # Verify the user table schema using raw SQL to confirm
    from sqlalchemy import text
    result = db.session.execute(text("PRAGMA table_info(user)"))
    columns = result.fetchall()
    column_names = [col[1] for col in columns]
    print(f"Columns in user table: {column_names}")
    
    print("Database has been reset! You will need to register new user accounts.")

# No reset_database() call needed 