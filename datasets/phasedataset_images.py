import os
import torch
from torchvision.io import read_video
from torchvision import transforms
from torchvision.transforms import Resize, RandomCrop, RandomHorizontalFlip, RandomRotation, ToTensor, Normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2


class PhaseDatasetImages(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        if mode == "train":
            self.data_file = "../OVFM/data/downstream/step_recognition/Cataract 101/" + "train_4_phases_images.txt"
            self.transform = transforms.Compose([
            Resize((224 + 32, 224 + 32)),
            RandomCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
            ])
        if mode == "val":
            self.data_file = "../OVFM/data/downstream/step_recognition/Cataract 101/" + "val_4_phases_images.txt"
            self.transform = transforms.Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
            ])
        self.img_paths = []
        self.labels = []
        self._read_data_file()
        

        self.total_frames = 8

    def _read_data_file(self):
        with open(self.data_file, 'r') as file:
            data = file.readlines()
            for line in data:
                img_path, label = line.strip().split(',')
                self.img_paths.append(img_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        if self.transform:
            transform = self.transform

        # Extract the folder index from the idx-th image path
        current_folder_index = self.img_paths[idx].split('/')[-2]

        # Load the series of images
        img_stack = []
        start_idx = max(0, idx - (self.total_frames-1))  # Ensure start index is not negative
        # print("idx:",idx)
        for i in range(start_idx, idx+1):
            # Get the folder index for the i-th path
            folder_index = self.img_paths[i].split('/')[-2]
            # make sure all images from the same video
            if folder_index != current_folder_index:
                # if not from the same video, copy from the first frame
                img_path = '/'.join(self.img_paths[idx].split('/')[:-1]) + '/' + '_'.join(['0' if i.isdigit() else i for i in self.img_paths[idx].split('/')[-1].split('_')[:-1]]) + '_0.jpg'
            else:
                img_path = self.img_paths[i]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"cannot read image: {img_path}")
            img = img.astype("int16").astype("float32")
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = Image.fromarray(np.uint8(img * 255)).convert('RGB')
            
            img = transform(img) # Apply the same transformation to all images

            img_stack.append(img)

        #  the frames are padded by repeating the first frame until there are 5 frames in total
        while len(img_stack) < self.total_frames:
            img_stack.insert(0, img_stack[0])

        # Stack images along the channel dimension
        img_stack = torch.stack(img_stack, dim=0)

        # Reshape the stacked tensor
        img_stack = img_stack.permute(1, 0, 2, 3)  # (channel, num_frames, height, width)

        label = self.labels[idx]

        return img_stack, label, idx, {}


class Config:
    def __init__(self):
        # Fill in your data mean and standard deviation here
        class DataConfig:
            MEAN = [0.485, 0.456, 0.406]
            STD = [0.229, 0.224, 0.225]
        self.DATA = DataConfig()

# Example usage:
if __name__ == "__main__":
    cfg = Config()

    # Initialize training and validation datasets
    train_dataset = PhaseDatasetImages(cfg, mode="train")
    val_dataset = PhaseDatasetImages(cfg, mode="val")

    # Use DataLoader to load data
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Print the length of training and validation datasets
    print("Train Dataset Length:", len(train_dataset))
    print("Val Dataset Length:", len(val_dataset))

    # Print the first training batch
    for batch in train_loader:
        print("Train Batch Sample Shape:", batch[0].shape)
        print("Train Batch Label Shape:", batch[1].shape)
        break

    # # Print the first validation batch
    # for batch in val_loader:
    #     print("Val Batch Sample Shape:", batch[0].shape)
    #     print("Val Batch Label Shape:", batch[1].shape)
    #     break
