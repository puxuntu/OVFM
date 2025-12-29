import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, Normalize, ToTensor, ToPILImage
from PIL import Image


class NucleusDataset(Dataset):
    def __init__(self, cfg, mode):
        """
        Dataset for single frame target localization
        
        Args:
            cfg: Configuration object with data parameters
            mode: 'train' or 'val'
            frames_dir: Directory containing the image frames
        """
        self.cfg = cfg
        self.frames_dir = "data/downstream/necleus_tracking/frames_with_json_labels"
        self.mode = mode
        
        # Set the path to the annotation file
        if mode == "train":
            self.data_file = "data/downstream/necleus_tracking/train.txt"
        elif mode == "val" or mode == "test":
            self.data_file = "data/downstream/necleus_tracking/test.txt"
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'.")
        
        # Read data file
        self.image_paths = []
        self.centers = []
        self.dimensions = []
        self._read_data_file()
        
        # Define transforms
        if mode == "train":
            # Training transforms with data augmentation
            self.transform = Compose([
                ToPILImage(),
                Resize((cfg.DATA.CROP_SIZE + 32, cfg.DATA.CROP_SIZE + 32)),
                RandomCrop((cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
                ToTensor(),
                Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
            ])
            
            # Transform for bounding boxes during random crop
            # This is more complex and requires custom handling in __getitem__
        else:
            # Validation/test transforms without random augmentations
            self.transform = Compose([
                ToPILImage(),
                Resize((cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
                ToTensor(),
                Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
            ])
    
    def _read_data_file(self):
        """Read annotation data from text file"""
        with open(self.data_file, 'r') as file:
            # Skip header line
            header = file.readline()

            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    image_name = parts[0]
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Store full path to image
                    image_path = os.path.join(self.frames_dir, image_name)
                    
                    # Store data
                    self.image_paths.append(image_path)
                    self.centers.append((center_x, center_y))
                    self.dimensions.append((width, height))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get item at index idx
        
        Returns:
            image: Tensor of shape [C, T, H, W] where T=1 (single frame)
            target: Dict containing 'center', 'dimensions', and 'normalized_center'
        """
        # Get image path and label information
        image_path = self.image_paths[idx]
        center = self.centers[idx]
        dimensions = self.dimensions[idx]
        
        # Load the image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
        
        # Get original image dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Apply transformations to the image
        if self.transform:
            # Processing remains the same...
            transformed_image = self.transform(image)
            
            # Calculate normalized coordinates as before...
            normalized_center_x = center[0] / orig_width
            normalized_center_y = center[1] / orig_height
            
            # Calculate normalized dimensions
            normalized_width = dimensions[0] / orig_width
            normalized_height = dimensions[1] / orig_height
        
        # Create target dictionary
        target = {
            'center': torch.tensor(center, dtype=torch.float),
            'dimensions': torch.tensor(dimensions, dtype=torch.float),
            'normalized_center': torch.tensor([normalized_center_x, normalized_center_y], dtype=torch.float),
            'normalized_dimensions': torch.tensor([normalized_width, normalized_height], dtype=torch.float),
            'original_size': torch.tensor([orig_width, orig_height], dtype=torch.int)
        }
        
        # Add temporal dimension to make shape [C, T, H, W] where T=1
        transformed_image = transformed_image.unsqueeze(1)  # This is the key change
        
        return transformed_image, target, idx, {}

def create_data_loaders(cfg):
    """
    Create data loaders for training and validation
    
    Args:
        cfg: Configuration object
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    # Create datasets
    train_dataset = NucleusDataset(cfg, mode="train")
    val_dataset = NucleusDataset(cfg, mode="val")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


# Example configuration class
class Config:
    def __init__(self):
        self.DATA = type('', (), {})()
        self.DATA.MEAN = [0.485, 0.456, 0.406]
        self.DATA.STD = [0.229, 0.224, 0.225]
        self.DATA.CROP_SIZE = 224
        self.DATA.NUM_WORKERS = 4
        
        self.TRAIN = type('', (), {})()
        self.TRAIN.BATCH_SIZE = 32
        
        self.VAL = type('', (), {})()
        self.VAL.BATCH_SIZE = 64


# Example usage
if __name__ == "__main__":
    cfg = Config()
    
    # Create dataset
    train_dataset = NucleusDataset(cfg, mode="train")
    
    # Get an item
    image, target, idx, _ = train_dataset[0]
    
    print(f"Image shape: {image.shape}")
    print(f"Center: {target['center']}")
    print(f"Dimensions: {target['dimensions']}")
    print(f"Normalized center: {target['normalized_center']}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(cfg)
    
    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")