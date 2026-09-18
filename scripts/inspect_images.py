import os
import csv
from pathlib import Path

def create_labels_csv(images_dir, output_csv):
    images_path = Path(images_dir)
    data = []
    
    seen_filenames = set()
    
    # Iterate through all subdirectories in the Images folder
    for subdir in images_path.iterdir():
        if subdir.is_dir():
            label = subdir.name
            # Iterate through all files in the subdirectory
            for file_path in subdir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    filename = file_path.name
                    if filename not in seen_filenames:
                        seen_filenames.add(filename)
                        # Parse filename: 01_0001_0_08_0916_1990_1.png or 0_BAR_11_1_12.jpg
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
                            
                            rel_path = file_path.relative_to(images_path.parent)
                            data.append([str(rel_path), label, district, institution, gender, age, date, serial, char_class])
                        elif len(parts) == 5:
                            # e.g., 0_BAR_11_1_12.jpg
                            gender_code = parts[0]
                            gender = 'Male' if gender_code == '0' else ('Female' if gender_code == '1' else gender_code)
                            district = parts[1]  # or institution
                            age = parts[2]
                            char_class = parts[3]
                            serial = parts[4]
                            
                            rel_path = file_path.relative_to(images_path.parent)
                            data.append([str(rel_path), label, district, '', gender, age, '', serial, char_class])
                        else:
                            rel_path = file_path.relative_to(images_path.parent)
                            data.append([str(rel_path), label, '', '', '', '', '', '', ''])
                    
    # Sort data by label (numerically) and then by filepath
    data.sort(key=lambda x: (int(x[1]) if x[1].isdigit() else x[1], x[0]))
    
    # Write to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'label', 'district', 'institution', 'gender', 'age', 'date', 'serial_number', 'character_class'])
        writer.writerows(data)
        
    print(f"Successfully created {output_csv} with {len(data)} labeled images.")

if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    
    # Process Images folder
    create_labels_csv(base_dir / 'Images', base_dir / 'labels.csv')
    
    # Process male folder
    create_labels_csv(base_dir / 'male', base_dir / 'labels_male.csv')
    
    # Process female folder
    create_labels_csv(base_dir / 'female', base_dir / 'labels_female.csv')
