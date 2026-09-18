import pandas as pd
from pathlib import Path

def balance_dataset(input_csv, output_csv):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if len(df) == 0:
        print(f"Warning: {input_csv} is empty! Skipping balancing.")
        return
        
    # Calculate the maximum number of samples across all classes
    class_counts = df['label'].value_counts()
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    
    print(f"Dataset currently has {len(df)} total samples.")
    print(f"Minimum samples in a class: {min_samples} (Class {class_counts.idxmin()})")
    print(f"Maximum samples in a class: {max_samples} (Class {class_counts.idxmax()})")
    
    # Sample 'max_samples' from each class (oversampling)
    # We use random_state for reproducibility so we get the same balanced set on subsequent runs
    balanced_df = df.groupby('label').sample(n=max_samples, replace=True, random_state=42)
    
    # Sort the balanced dataset by label and then by filepath
    # Convert label to integer for sorting if possible
    balanced_df['label_int'] = pd.to_numeric(balanced_df['label'], errors='coerce')
    balanced_df = balanced_df.sort_values(by=['label_int', 'filepath']).drop(columns=['label_int'])
    
    print(f"Writing balanced dataset with {len(balanced_df)} samples to {output_csv}...")
    balanced_df.to_csv(output_csv, index=False)
    
    print("Done! All classes now have exactly", max_samples, "samples.")

if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    
    print("--- Balancing main Images ---")
    balance_dataset(base_dir / 'labels.csv', base_dir / 'labels_balanced.csv')
    
    print("\n--- Balancing Male Images ---")
    balance_dataset(base_dir / 'labels_male.csv', base_dir / 'labels_male_balanced.csv')
    
    print("\n--- Balancing Female Images ---")
    balance_dataset(base_dir / 'labels_female.csv', base_dir / 'labels_female_balanced.csv')
