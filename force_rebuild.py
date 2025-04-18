import os
import sqlite3
from app import app

# Get database path
with app.app_context():
    db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')

print(f"Working with database at {db_path}")

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if database exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(f"Existing tables: {tables}")

if ('user',) in tables:
    # Check current columns
    cursor.execute("PRAGMA table_info(user)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    print(f"Current columns in user table: {column_names}")
    
    # Drop the user table to recreate it
    print("Dropping user table to recreate with new schema...")
    cursor.execute("DROP TABLE user")
    conn.commit()

# Recreate user table with all required columns
print("Creating new user table with all required columns...")
cursor.execute('''
CREATE TABLE user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(80) UNIQUE NOT NULL,
    password VARCHAR(120) NOT NULL,
    name VARCHAR(80) NOT NULL,
    email VARCHAR(120),
    bio TEXT,
    location VARCHAR(100),
    profile_picture VARCHAR(255) DEFAULT 'default_profile.png',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Check if analysis table exists, if so recreate the relationship
if ('analysis',) in tables:
    print("Recreating foreign key relationship for analysis table...")
    cursor.execute("PRAGMA foreign_keys=OFF")
    cursor.execute("CREATE TABLE temp_analysis AS SELECT * FROM analysis")
    cursor.execute("DROP TABLE analysis")
    cursor.execute('''
    CREATE TABLE analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        prediction VARCHAR(20) NOT NULL,
        confidence FLOAT NOT NULL,
        image_path VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user (id)
    )
    ''')
    cursor.execute("INSERT INTO analysis SELECT * FROM temp_analysis")
    cursor.execute("DROP TABLE temp_analysis")
    cursor.execute("PRAGMA foreign_keys=ON")

conn.commit()
conn.close()

print("Database structure rebuilt successfully!")
print("You will need to register new user accounts.") 