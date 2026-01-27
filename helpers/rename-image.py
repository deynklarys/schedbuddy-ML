import os
import shutil
from pathlib import Path
from pdf2image import convert_from_path
from config import POPPLER_PATH

# Rename images in sequential order. If file is pdf, convert first.
def rename_images(source_dir, dest_dir):
    # Create new dest_dir
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files from source_dir
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = sorted ([
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

# Convert all PDFs to PNG
def convert_from_pdf(source_dir, dest_dir,  poppler_path):
    # Get all PDF files from source_dir
    pdf_files = sorted([
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() == '.pdf'
    ])

    # Convert PDF to PNG
    for pdf_file in pdf_files:
        try:
            images = convert_from_path(str(pdf_file), poppler_path=poppler_path)
            for page_num, image in enumerate(images,start=1):
                output_filename = f"{pdf_file.stem}-{page_num}.png"
                output_path = dest_dir / output_filename
                image.save(str(output_path), "png")
                print(f"Converted: {pdf_file.name} (p. {page_num})-> {output_filename}")
        except Exception as e:
            print(f"Error converting {pdf_file.name}: {e}")
    
    print(f"\nTotal PDFs processed: {len(pdf_files)}")
    print(f"Images saved to: {dest_dir.absolute()}")

if __name__ == "__main__":
    source_dir = Path("../images")
    dest_dir = Path("./images")

    # Convert .pdf to .png if there are.
    convert_from_pdf(source_dir, dest_dir=Path("../images"), 
                     poppler_path=POPPLER_PATH)
    rename_images(source_dir, dest_dir)