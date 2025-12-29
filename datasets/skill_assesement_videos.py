import os
import torch
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, RandomCrop, Normalize, ToTensor, ToPILImage
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2


class SkillAssesementDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        if mode == "train":
            self.data_file = "../OVFM/data/downstream/surgical_skill_assessment/" + "train.txt"
            # self.data_file = "../OVFM/data/downstream/complication_detection/" + "train.txt"
        if mode == "val":
            self.data_file = "../OVFM/data/downstream/surgical_skill_assessment/" + "test.txt"
            # self.data_file = "../OVFM/data/downstream/complication_detection/" + "train.txt"
        self.video_paths = []
        self.labels = []
        self._read_data_file()
        self.transform = Compose([
            ToPILImage(),
            Resize((224 + 32, 224 + 32)),
            RandomCrop((224, 224)),
            ToTensor(),
            Normalize(mean=self.cfg.DATA.MEAN, std=self.cfg.DATA.STD)
        ])

    def _read_data_file(self):
        with open(self.data_file, 'r') as file:
            data = file.readlines()
            for line in data:
                video_path, label = line.strip().split(',')
                self.video_paths.append(video_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num=16, dtype=int)

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        seed = np.random.randint(2147483647)  
        if self.transform:
            torch.manual_seed(seed)  
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)

        frames = frames.permute(1,0,2,3)

        return frames, label,  idx, {}


class Config:
    def __init__(self):
        self.datasetpath = "data/downstream/phase recognition/"
        class DataConfig:
            MEAN = [0.485, 0.456, 0.406]
            STD = [0.229, 0.224, 0.225]
        self.DATA = DataConfig()

if __name__ == "__main__":
    cfg = Config()

    train_dataset = SkillAssesementDataset(cfg, mode="train")
    val_dataset = SkillAssesementDataset(cfg, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

    print("Train Dataset Length:", len(train_dataset))
    print("Val Dataset Length:", len(val_dataset))

    for batch in train_loader:
        print("Train Batch Sample Shape:", batch[0].shape)
        print("Train Batch Label Shape:", batch[1])
        break

    for batch in val_loader:
        print("Val Batch Sample Shape:", batch[0].shape)
        print("Val Batch Label Shape:", batch[1])
        break
