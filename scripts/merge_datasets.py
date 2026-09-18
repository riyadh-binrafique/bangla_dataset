import pandas as pd
from pathlib import Path

def merge_and_balance(base_dir):
    print("Loading datasets...")
    df1 = pd.read_csv(base_dir / 'labels.csv')
    df2 = pd.read_csv(base_dir / 'labels_male.csv')
    df3 = pd.read_csv(base_dir / 'labels_female.csv')
    
    # Merge datasets
    merged_df = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"Total merged samples: {len(merged_df)}")
    
    # Fill missing values
    merged_df['institution'] = merged_df['institution'].fillna('')
    merged_df['date'] = merged_df['date'].fillna('')
    # For character_class, there are a couple of empty strings causing NaNs (e.g. 0_BAR_14__492.jpg)
    merged_df['character_class'] = merged_df['character_class'].fillna(0)
    
    # Save un-balanced merged dataset
    merged_csv_path = base_dir / 'labels_merged.csv'
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Saved merged un-balanced dataset to {merged_csv_path}")
    
    # Balance the merged dataset using oversampling
    print("\nBalancing merged dataset...")
    class_counts = merged_df['label'].value_counts()
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    
    print(f"Minimum samples in a class: {min_samples} (Class {class_counts.idxmin()})")
    print(f"Maximum samples in a class: {max_samples} (Class {class_counts.idxmax()})")
    
    # Oversample to max_samples
    balanced_df = merged_df.groupby('label').sample(n=max_samples, replace=True, random_state=42)
    
    # Sort
    balanced_df['label_int'] = pd.to_numeric(balanced_df['label'], errors='coerce')
    balanced_df = balanced_df.sort_values(by=['label_int', 'filepath']).drop(columns=['label_int'])
    
    balanced_csv_path = base_dir / 'labels_merged_balanced.csv'
    balanced_df.to_csv(balanced_csv_path, index=False)
    print(f"Saved balanced merged dataset to {balanced_csv_path}")
    print(f"All classes now have {max_samples} samples.")

if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    merge_and_balance(base_dir)
