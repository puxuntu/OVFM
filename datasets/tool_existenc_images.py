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
import re
import pandas as pd

class ToolExistenceImages(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        dataset_base_path = "../OVFM/data/downstream/surgical_tool_existence_recognition/"

        if mode == "train":
            self.data_file = dataset_base_path + "train.txt"
            self.transform = transforms.Compose([
            Resize((224 + 32, 224 + 32)),
            RandomCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
            ])
        if mode == "val":
            self.data_file = dataset_base_path + "test.txt"
            self.transform = transforms.Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
            ])
        self.img_paths = []
        self.labels = []
        self.total_frames = 8

        label_dfs = {}

        if mode == "train":
            for i in range(1, 26):
                path = os.path.join(dataset_base_path, f"label/train_gt/train{i:02}.csv")
                label_dfs[i] = pd.read_csv(path)

        if mode == "val":
            for i in range(1, 26):
                path = os.path.join(dataset_base_path, f"label/test_gt/test{i:02}.csv")
                label_dfs[i] = pd.read_csv(path)

        with open(self.data_file, 'r') as file:
            data = file.readlines()

            for line in data:
                img_path = line.strip()
                match = re.search(r'/(\d+)/video(\d+)_(\d+)', img_path)

                video_number = int(match.group(2))
                frame_number = int(match.group(3)) * 10 + 1

                if mode == "train":
                    csv_index = video_number - 25  
                else:
                    csv_index = video_number

                df = label_dfs.get(csv_index)
 
                if df is None:
                    print(f"CSV for video {csv_index} not loaded.")
                    continue
                
                row = df[df.iloc[:, 0] == frame_number]

                if row.empty:
                    print(video_number)
                    print(frame_number)
                    continue
                
                labels = row.iloc[0, 1:].values
                labels_tensor = torch.tensor(labels, dtype=torch.float32)
                self.img_paths.append(img_path)
                self.labels.append(labels_tensor)
        

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        if self.transform:
            transform = self.transform

        current_folder_index = self.img_paths[idx].split('/')[-2]

        img_stack = []
        start_idx = max(0, idx - (self.total_frames-1))  
        for i in range(start_idx, idx+1):
            folder_index = self.img_paths[i].split('/')[-2]
            if folder_index != current_folder_index:
                img_path = '/'.join(self.img_paths[idx].split('/')[:-1]) + '/' + '_'.join(['0' if i.isdigit() else i for i in self.img_paths[idx].split('/')[-1].split('_')[:-1]]) + '_0.jpg'
            else:
                img_path = self.img_paths[i]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = Image.fromarray(np.uint8(img * 255)).convert('RGB')
            
            img = transform(img) 

            img_stack.append(img)

        while len(img_stack) < self.total_frames:
            img_stack.insert(0, img_stack[0])

        img_stack = torch.stack(img_stack, dim=0)

        img_stack = img_stack.permute(1, 0, 2, 3)  

        label = self.labels[idx]

        return img_stack, label, idx, self.img_paths[idx], {}


class Config:
    def __init__(self):
        class DataConfig:
            MEAN = [0.485, 0.456, 0.406]
            STD = [0.229, 0.224, 0.225]
        self.DATA = DataConfig()

if __name__ == "__main__":
    cfg = Config()

    train_dataset = ToolExistenceImages(cfg, mode="train")
    val_dataset = ToolExistenceImages(cfg, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    print("Train Dataset Length:", len(train_dataset))
    print("Val Dataset Length:", len(val_dataset))

    for batch in train_loader:
        print("Train Batch Sample Shape:", batch[0].shape)
        print("Train Batch Label Shape:", batch[1].shape)
        break

    for batch in val_loader:
        print("Val Batch Sample Shape:", batch[0].shape)
        print("Val Batch Label Shape:", batch[1].shape)
        break
