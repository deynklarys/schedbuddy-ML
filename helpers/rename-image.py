import os
import shutil
from pathlib import Path

def rename_images():
    """
    Renames images from images/cropped directory to sequential numbering (1-100)
    and saves them to helpers/images directory.
    """
    source_dir = Path("../images/cropped")
    dest_dir = Path("./images")
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(exist_ok=True)
    
    # Get all image files from source directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = sorted([
        f for f in source_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    # Rename and copy images with sequential numbering
    for idx, image_file in enumerate(image_files, start=1):
        new_filename = f"{idx}{image_file.suffix}"
        dest_path = dest_dir / new_filename
        
        shutil.copy2(image_file, dest_path)
        print(f"Copied: {image_file.name} -> {new_filename}")
    
    print(f"\nTotal images processed: {len(image_files)}")
    print(f"Images saved to: {dest_dir.absolute()}")

if __name__ == "__main__":
    rename_images()
