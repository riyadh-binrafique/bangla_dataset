import os
import cv2
import numpy as np
from PIL import Image
import shutil

def is_corrupted(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # verify that it is, in fact, an image
        return False
    except (IOError, SyntaxError) as e:
        return True

def is_blank(image):
    # Calculate standard deviation of pixel values
    std_dev = np.std(image)
    return std_dev < 5  # Threshold for blank image (almost no variation in pixels)

def is_blurry(image, threshold=50):
    # Compute Variance of Laplacian
    # Lower variance means more blurry
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance < threshold

def main():
    base_dir = "/Users/riyadhmac/Downloads/Bangla_OCR/Ekush_Dataset/dataset-2"
    removed_dir = "/Users/riyadhmac/Downloads/Bangla_OCR/Ekush_Dataset/removed_data"
    
    if not os.path.exists(removed_dir):
        os.makedirs(removed_dir)

    corrupted_count = 0
    blank_count = 0
    blurry_count = 0

    print(f"Starting data cleaning in: {base_dir}")
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # We are assuming image files (skip hidden files like .DS_Store)
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            
            # 1. Check for corruption
            if is_corrupted(file_path):
                print(f"Corrupted: {file_path}")
                shutil.move(file_path, os.path.join(removed_dir, f"corrupt_{file}"))
                corrupted_count += 1
                continue
            
            # Read image using OpenCV for blank and blur checks
            # Read in grayscale
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                # Could not read the image properly
                print(f"Unreadable by OpenCV: {file_path}")
                shutil.move(file_path, os.path.join(removed_dir, f"unreadable_{file}"))
                corrupted_count += 1
                continue
                
            # 2. Check for blank image
            if is_blank(image):
                print(f"Blank: {file_path}")
                shutil.move(file_path, os.path.join(removed_dir, f"blank_{file}"))
                blank_count += 1
                continue
                
            # 3. Check for blurriness
            # For characters, threshold might need to be tuned. 50 is a reasonable start.
            if is_blurry(image, threshold=50):
                print(f"Blurry: {file_path}")
                shutil.move(file_path, os.path.join(removed_dir, f"blurry_{file}"))
                blurry_count += 1
                continue

    print("--- Cleaning Complete ---")
    print(f"Total Corrupted removed: {corrupted_count}")
    print(f"Total Blank removed:     {blank_count}")
    print(f"Total Blurry removed:    {blurry_count}")
    print(f"Check {removed_dir} to verify removed images.")

if __name__ == "__main__":
    main()
