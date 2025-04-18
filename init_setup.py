from app import app, create_default_profile_picture

print("Creating default profile picture...")
with app.app_context():
    create_default_profile_picture()
print("Default profile picture created!") 