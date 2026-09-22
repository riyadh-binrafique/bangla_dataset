import os
import hashlib
import shutil

def get_file_hash(file_path):
    """Computes the MD5 hash of a file."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    base_dir = "/Users/riyadhmac/Downloads/Bangla_OCR/Ekush_Dataset/dataset-2"
    removed_dir = "/Users/riyadhmac/Downloads/Bangla_OCR/Ekush_Dataset/removed_data"
    
    if not os.path.exists(removed_dir):
        os.makedirs(removed_dir)

    print(f"Starting duplicate detection in: {base_dir}")
    
    seen_hashes = set()
    duplicate_count = 0
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            file_hash = get_file_hash(file_path)
            
            if file_hash is None:
                continue
                
            if file_hash in seen_hashes:
                # We found a duplicate
                print(f"Duplicate found: {file_path}")
                # Ensure the destination filename is unique if there are multiple duplicates with the same name
                dest_filename = f"duplicate_{duplicate_count}_{file}"
                shutil.move(file_path, os.path.join(removed_dir, dest_filename))
                duplicate_count += 1
            else:
                # First time seeing this file content
                seen_hashes.add(file_hash)

    print("--- Duplicate Removal Complete ---")
    print(f"Total Duplicates removed: {duplicate_count}")

if __name__ == "__main__":
    main()
