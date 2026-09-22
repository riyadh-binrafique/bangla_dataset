import os
import cv2
import shutil

def preprocess_image(image_path, output_path):
    """Reads an image, converts to grayscale, applies Otsu's thresholding, and saves it."""
    # Read in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {image_path}")
        return False
        
    # Apply Otsu's thresholding
    # This automatically finds the best threshold to separate the foreground from the background
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save the processed image
    cv2.imwrite(output_path, thresh)
    return True

def main():
    base_dir = "/Users/riyadhmac/Downloads/Bangla_OCR/Ekush_Dataset/dataset-2"
    output_dir = "/Users/riyadhmac/Downloads/Bangla_OCR/Ekush_Dataset/dataset-2_preprocessed"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting preprocessing from {base_dir} to {output_dir}")
    processed_count = 0
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith('.'):
                continue
                
            input_path = os.path.join(root, file)
            
            # Construct corresponding output path
            rel_path = os.path.relpath(input_path, base_dir)
            output_path = os.path.join(output_dir, rel_path)
            
            # Ensure output subdirectory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if preprocess_image(input_path, output_path):
                processed_count += 1
                if processed_count % 5000 == 0:
                    print(f"Processed {processed_count} images...")

    print(f"--- Preprocessing Complete ---")
    print(f"Total images processed: {processed_count}")
    print(f"Preprocessed images saved in: {output_dir}")

if __name__ == "__main__":
    main()
