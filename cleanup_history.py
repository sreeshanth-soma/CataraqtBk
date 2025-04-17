import os
import sys

# Add project directory to Python path to import app
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from app import app, db, Analysis

def cleanup_missing_image_analyses():
    """Iterates through Analysis records and deletes those with missing image files."""
    deleted_count = 0
    kept_count = 0
    
    with app.app_context():
        print("Starting cleanup of analysis history...")
        upload_folder = app.config['UPLOAD_FOLDER']
        if not os.path.isdir(upload_folder):
            print(f"Error: Upload folder '{upload_folder}' does not exist. Aborting.")
            return

        analyses_to_check = Analysis.query.all()
        total_records = len(analyses_to_check)
        print(f"Found {total_records} analysis records to check.")

        for i, analysis in enumerate(analyses_to_check):
            if (i + 1) % 50 == 0:
                print(f"Checked {i + 1}/{total_records} records...")

            if not analysis.image_path:
                print(f"Deleting analysis ID {analysis.id} - image_path is empty.")
                db.session.delete(analysis)
                deleted_count += 1
                continue
            
            expected_image_path = os.path.join(upload_folder, analysis.image_path)
            
            if not os.path.exists(expected_image_path):
                print(f"Deleting analysis ID {analysis.id} - image not found at '{expected_image_path}'.")
                db.session.delete(analysis)
                deleted_count += 1
            else:
                kept_count += 1
        
        if deleted_count > 0:
            print(f"Committing deletions ({deleted_count} records)...")
            db.session.commit()
            print("Deletions committed.")
        else:
            print("No records needed deletion.")
            
        print(f"Cleanup complete. Total checked: {total_records}, Deleted: {deleted_count}, Kept: {kept_count}")

if __name__ == "__main__":
    cleanup_missing_image_analyses() 