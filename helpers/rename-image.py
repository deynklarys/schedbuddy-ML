import os
import shutil
from pathlib import Path

def rename_images():
    """
    Renames images from images/cropped directory to sequential numbering (1-100)
    and saves them to helpers/images directory.
    """
    # source_dir = Path("../images/form-COR")
    source_dir = Path("../images/original")
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
    for idx, image_file in enumerate(image_files, start=46): # Modify as needed
        new_filename = f"{idx}{image_file.suffix}"
        dest_path = dest_dir / new_filename
        
        shutil.copy2(image_file, dest_path)
        print(f"Copied: {image_file.name} -> {new_filename}")
    
    print(f"\nTotal images processed: {len(image_files)}")
    print(f"Images saved to: {dest_dir.absolute()}")


def convert_from_pdf():
    """
    Converts PDF files from images/form-COR directory to PNG format
    and saves them to converted-pdf directory.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("Error: pdf2image module not found. Install it with: pip install pdf2image")
        return
    
    source_dir = Path("../images/form-COR")
    dest_dir = Path("../images/form-COR")
    poppler_path = r"C:\Applications ni Deyn\Poppler\poppler-25.12.0\Library\bin"

    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files from source directory
    pdf_files = sorted([
        f for f in source_dir.iterdir() 
        if f.is_file() and f.suffix.lower() == '.pdf'
    ])
    
    # Convert PDFs to PNG
    for pdf_file in pdf_files:
        try:
            images = convert_from_path(str(pdf_file), poppler_path=poppler_path)
            for page_num, image in enumerate(images, start=1):
                output_filename = f"{pdf_file.stem}.png"
                output_path = dest_dir / output_filename
                image.save(str(output_path), "PNG")
                print(f"Converted: {pdf_file.name} (page {page_num}) -> {output_filename}")
        except Exception as e:
            print(f"Error converting {pdf_file.name}: {e}")
    
    print(f"\nTotal PDFs processed: {len(pdf_files)}")
    print(f"Images saved to: {dest_dir.absolute()}")

if __name__ == "__main__":
    # convert_from_pdf()
    rename_images()
