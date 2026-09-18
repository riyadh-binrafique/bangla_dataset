import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BanglaOCRDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct full image path
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['filepath'])
        
        # Open image and convert to grayscale (since handwriting is usually 1 channel)
        image = Image.open(img_name).convert('L')
        
        # Get label (the merged dataset has labels 0 to 119)
        label = int(self.data_frame.iloc[idx]['label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(csv_path, root_dir, batch_size=32, img_size=(64, 64)):
    """
    Helper function to create DataLoaders with appropriate augmentations.
    """
    
    # Define the augmentations for the training set
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        # Randomly rotate the image by up to 10 degrees
        transforms.RandomRotation(degrees=10),
        # Apply slight random affine transformations (shear and translation)
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        # Convert to tensor and scale pixels between 0 and 1
        transforms.ToTensor(),
        # Normalize with mean and standard deviation (using typical 0.5 for 1-channel)
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Define simpler transforms for validation/testing (no augmentation)
    test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create the dataset
    dataset = BanglaOCRDataset(csv_file=csv_path, root_dir=root_dir, transform=train_transforms)
    
    # For demonstration, we'll just return a single dataloader. 
    # In practice, you would split your CSV into train/val/test first.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader

if __name__ == '__main__':
    from pathlib import Path
    
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / 'labels_merged_balanced.csv'
    
    # The root_dir is the base_dir because 'filepath' in the CSV includes 'Images/...'
    root_dir = base_dir
    
    print("Initializing PyTorch DataLoader with Augmentations...")
    dataloader = get_data_loaders(csv_path, root_dir, batch_size=16)
    
    # Fetch a single batch to test
    for images, labels in dataloader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Sample labels: {labels}")
        break
    
    print("DataLoader is working perfectly! You can now import this into your training script.")
