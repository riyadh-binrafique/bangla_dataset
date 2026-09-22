import os
import cv2
import numpy as np
import random

def augment_image(image):
    """Applies random rotation and translation to an image."""
    rows, cols = image.shape[:2]

    # Random rotation angle between -10 and 10 degrees
    angle = random.uniform(-10, 10)
    
    # Random translation between -2 and 2 pixels
    tx = random.uniform(-2, 2)
    ty = random.uniform(-2, 2)

    # 1. Rotation
    # Calculate the rotation matrix
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    
    # Perform the rotation, use border mode to fill with white/black depending on bg.
    # Since we did Otsu thresholding, bg is typically white (255) for Ekush. 
    # Actually wait, let's just use BORDER_REPLICATE or BORDER_CONSTANT with white.
    # We will use BORDER_REPLICATE to be safe.
    rotated = cv2.warpAffine(image, M_rot, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    # 2. Translation
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(rotated, M_trans, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    return shifted

def main():
    base_dir = "/Users/riyadhmac/Downloads/Bangla_OCR/Ekush_Dataset/dataset-2_preprocessed"
    
    print(f"Starting data augmentation in {base_dir}")
    
    # Step 1: Collect all original files first to avoid augmenting the augmented images
    original_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith('.') or file.endswith('_aug.jpg'):
                continue
            original_files.append(os.path.join(root, file))

    total_files = len(original_files)
    print(f"Found {total_files} original images to augment.")

    # Step 2: Augment
    processed_count = 0
    for file_path in original_files:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        aug_img = augment_image(img)
        
        # Determine new filename
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        new_file_path = os.path.join(dir_name, f"{name}_aug{ext}")
        
        cv2.imwrite(new_file_path, aug_img)
        
        processed_count += 1
        if processed_count % 5000 == 0:
            print(f"Augmented {processed_count}/{total_files} images...")

    print(f"--- Data Augmentation Complete ---")
    print(f"Total images augmented: {processed_count}")

if __name__ == "__main__":
    main()
