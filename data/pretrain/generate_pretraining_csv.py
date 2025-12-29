import os
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import csv
import glob

datadirs = [
    'data/pretrain/OVFM-pretrain_data',
]

videos = []

for datadir in datadirs:
    videolist = glob.glob(f'{datadir}/**/*.mp4', recursive=True)
    for video in tqdm(videolist, desc=f'Scanning {datadir}'):
        videos.append([os.path.abspath(video), -1])

print(f'Total videos found: {len(videos)}')

csv_file_path = os.path.join('data/pretrain', 'train.csv')
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

with open(csv_file_path, "w", newline='') as csvfile:  
    writer = csv.writer(csvfile)
    writer.writerows(videos)

print(f'Video paths written to {csv_file_path}')

