import os
import shutil
import pandas as pd
from pathlib import Path

def create_unified_dataset(base_dir):
    unified_dir = base_dir / 'unified_dataset'
    unified_dir.mkdir(exist_ok=True)
    
    data = []
    seen_filenames = set()
    total_files = 0
    
    print("Consolidating ALL images correctly into 'unified_dataset' folder...")
    
    for dataset_name in ['Images', 'male', 'female']:
        dataset_dir = base_dir / dataset_name
        if not dataset_dir.exists():
            continue
            
        print(f"Processing {dataset_name}...")
        for class_dir in dataset_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            label = class_dir.name
            target_class_dir = unified_dir / label
            target_class_dir.mkdir(exist_ok=True)
            
            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    original_filename = file_path.name
                    # FIX: Append the class label to the filename to guarantee global uniqueness
                    # because the source dataset reuses identical filenames across different class folders!
                    new_filename = f"{dataset_name}_class{label}_{original_filename}"
                    target_file = target_class_dir / new_filename
                    
                    if new_filename not in seen_filenames:
                        seen_filenames.add(new_filename)
                        
                        # Link or copy the file
                        if not target_file.exists():
                            try:
                                os.link(file_path, target_file)
                            except OSError:
                                shutil.copy2(file_path, target_file)
                        
                        # Parse original metadata from the stem
                        parts = file_path.stem.split('_')
                        if len(parts) >= 7:
                            district = parts[0]
                            institution = parts[1]
                            gender_code = parts[2]
                            gender = 'Male' if gender_code == '0' else ('Female' if gender_code == '1' else gender_code)
                            age = parts[3]
                            date = parts[4]
                            serial = parts[5]
                            char_class = parts[6]
                        elif len(parts) == 5:
                            gender_code = parts[0]
                            gender = 'Male' if gender_code == '0' else ('Female' if gender_code == '1' else gender_code)
                            district = parts[1]
                            institution = ''
                            age = parts[2]
                            char_class = parts[3]
                            serial = parts[4]
                            date = ''
                        else:
                            district = institution = gender = age = date = serial = char_class = ''
                        
                        # Save the new relative path
                        rel_path = f"{label}/{new_filename}"
                        data.append([rel_path, label, district, institution, gender, age, date, serial, char_class])
                        total_files += 1

    # Save the new unified CSV
    unified_csv_path = base_dir / 'labels_unified.csv'
    df = pd.DataFrame(data, columns=['filepath', 'label', 'district', 'institution', 'gender', 'age', 'date', 'serial_number', 'character_class'])
    
    # Fill any broken NaNs
    df['character_class'] = df['character_class'].fillna(0)
    df.to_csv(unified_csv_path, index=False)
    
    print(f"Successfully consolidated {total_files} images into {unified_dir}")
    print(f"Created unified raw dataset: {unified_csv_path}")
    
    # Balance the unified dataset
    print("\nBalancing unified dataset...")
    class_counts = df['label'].value_counts()
    max_samples = class_counts.max()
    
    # Oversample
    balanced_df = df.groupby('label').sample(n=max_samples, replace=True, random_state=42)
    
    # Sort
    balanced_df['label_int'] = pd.to_numeric(balanced_df['label'], errors='coerce')
    balanced_df = balanced_df.sort_values(by=['label_int', 'filepath']).drop(columns=['label_int'])
    
    balanced_csv_path = base_dir / 'labels_unified_balanced.csv'
    balanced_df.to_csv(balanced_csv_path, index=False)
    print(f"Saved perfectly balanced dataset to {balanced_csv_path}")
    print(f"All {len(class_counts)} classes now have exactly {max_samples} samples.")

if __name__ == '__main__':
    base_dir = Path(__file__).parent
    create_unified_dataset(base_dir)
