import os
import sqlite3

# 1. Delete the database file
print("Deleting database file...")
db_path = 'users.db'
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Removed {db_path}")

# 2. Create a new database with the correct schema
print("Creating new database with direct SQL...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create user table with all required fields
cursor.execute('''
CREATE TABLE user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(80) NOT NULL UNIQUE,
    password VARCHAR(120) NOT NULL,
    name VARCHAR(80) NOT NULL,
    email VARCHAR(120),
    bio TEXT,
    location VARCHAR(100),
    profile_picture VARCHAR(255) DEFAULT 'default_profile.png',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
print("Created user table with all fields")

# Create analysis table with foreign key to user
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
print("Created analysis table")

# Commit changes and close connection
conn.commit()
conn.close()

# 3. Ensure uploads directory exists
uploads_dir = os.path.join('static', 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
    print(f"Created uploads directory at {uploads_dir}")

print("\nDatabase has been directly created with the correct schema.")
print("You need to restart your Flask application and create new user accounts.")
print("Make sure to run the app with the virtual environment activated.") 