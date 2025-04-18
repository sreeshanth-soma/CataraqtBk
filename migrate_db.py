import os
import sqlite3
from app import app, db, User, create_default_profile_picture

def migrate_user_table():
    """
    Migration script to add profile picture, email, bio, and location columns to the User table.
    """
    # Get the database path
    with app.app_context():
        db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
    
    # Check if the database exists, if not create tables first
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        print(f"Database not found or empty. Creating new database at {db_path}")
        with app.app_context():
            db.create_all()
            print("Database tables created.")
    
    print(f"Upgrading database at {db_path}")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the user table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
    if not cursor.fetchone():
        print("Table 'user' not found. Creating tables...")
        conn.close()
        with app.app_context():
            db.create_all()
            print("Database tables created.")
        # Reconnect after table creation
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    
    # Check if the columns already exist
    cursor.execute("PRAGMA table_info(user)")
    columns = [column[1] for column in cursor.fetchall()]
    print(f"Existing columns: {columns}")
    
    # Add columns if they don't exist
    try:
        if 'email' not in columns:
            print("Adding email column to user table...")
            cursor.execute("ALTER TABLE user ADD COLUMN email TEXT;")
        
        if 'bio' not in columns:
            print("Adding bio column to user table...")
            cursor.execute("ALTER TABLE user ADD COLUMN bio TEXT;")
        
        if 'location' not in columns:
            print("Adding location column to user table...")
            cursor.execute("ALTER TABLE user ADD COLUMN location TEXT;")
        
        if 'profile_picture' not in columns:
            print("Adding profile_picture column to user table...")
            cursor.execute("ALTER TABLE user ADD COLUMN profile_picture TEXT DEFAULT 'default_profile.png';")
        
        # Commit the changes
        conn.commit()
        print("Migration completed successfully!")
    
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    
    finally:
        # Close the connection
        conn.close()
    
    # Ensure the uploads directory exists
    uploads_dir = os.path.join(app.root_path, 'static/uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print(f"Created uploads directory at {uploads_dir}")
    
    # Create default profile picture if it doesn't exist
    create_default_profile_picture()
    print("Default profile picture created")

if __name__ == "__main__":
    with app.app_context():
        migrate_user_table()
        print("Database migration complete!") 