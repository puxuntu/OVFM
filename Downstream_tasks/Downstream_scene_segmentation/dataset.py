import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os


class SegCTDataset(Dataset):
    def __init__(self, dataroot, transforms=None, mode='train'):
        self.mode = mode
        self.img_transforms = transforms  # These will be applied only to the image
        
        # Determine which txt file to use based on mode
        if self.mode == 'train':
            txt_file = os.path.join(dataroot, 'train_dataset.txt')
        elif self.mode == 'test':
            txt_file = os.path.join(dataroot, 'test_dataset.txt')
        else:
            raise ValueError("Invalid mode. Mode should be 'train' or 'test'.")

        # Read paths from the txt file
        try:
            with open(txt_file, 'r') as f:
                self.paths = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {txt_file}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Parse the file paths correctly
        full_line = self.paths[idx].strip()
        split_index = full_line.find('.png ') + 4  # Include '.png' length
        
        if split_index < 4:  # Not found or wrong format
            raise ValueError(f"Invalid line format in dataset file: {full_line}")
            
        image_path = full_line[:split_index]
        label_path = full_line[split_index + 1:]  # +1 to skip the space

        # Load and preprocess image
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
        except Exception as e:
            raise ValueError(f"Error reading image {image_path}: {str(e)}")
            
        # Convert to float and normalize before converting to PIL
        img = img.astype("float32")
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = Image.fromarray(np.uint8(img * 255)).convert('RGB')
        
        # Load label image - ensure labels remain as integers (0-15)
        try:
            label = Image.open(label_path)
            if label is None:
                raise ValueError(f"Failed to read label: {label_path}")
        except Exception as e:
            raise ValueError(f"Error reading label {label_path}: {str(e)}")
            
        # Apply transforms - with separate handling for image and label
        if self.img_transforms:
            # Handle geometric transforms that need to be applied to both image and label
            if isinstance(self.img_transforms, transforms.Compose):
                # First, apply geometric transforms to both
                img_copy = img.copy()
                label_copy = label.copy()
                
                # Apply geometric transforms that need to be applied identically to both
                for t in self.img_transforms.transforms:
                    if isinstance(t, (RandomCrop, RandomFlip, RandomRotation, Resize, CenterCrop)):
                        img_copy, label_copy = t((img_copy, label_copy))
                
                # Convert both to tensors
                for t in self.img_transforms.transforms:
                    if isinstance(t, ToTensor):
                        img_tensor, label_tensor = t((img_copy, label_copy))
                        break
                else:
                    # If no ToTensor found, do the conversion manually
                    img_array = np.array(img_copy)
                    label_array = np.array(label_copy)
                    img_tensor = torch.from_numpy(img_array.transpose((2, 0, 1))).float()
                    label_tensor = torch.from_numpy(label_array).long()
                
                # Apply remaining transforms (like normalization) only to the image
                for t in self.img_transforms.transforms:
                    if isinstance(t, Normalize):
                        img_tensor = t.transform(img_tensor)
            else:
                # If not a Compose transform, just apply to image and convert label manually
                img_tensor = self.img_transforms(img)
                label_array = np.array(label)
                label_tensor = torch.from_numpy(label_array).long()
        else:
            # If no transforms, convert manually
            img_array = np.array(img)
            label_array = np.array(label)
            img_tensor = torch.from_numpy(img_array.transpose((2, 0, 1))).float()
            label_tensor = torch.from_numpy(label_array).long()
            
        # Create sample dictionary
        if self.mode == 'train':
            sample = {'image': img_tensor, 'label': label_tensor}
        else:
            sample = {'image': img_tensor, 'label': label_tensor, 'case_name': os.path.basename(image_path)}

        return sample


# Utility transformation classes

class CenterCrop(object):
    def __init__(self, arg):
        self.transform = transforms.CenterCrop(arg)

    def __call__(self, sample):
        img, label = sample
        return self.transform(img), self.transform(label)


class Resize(object):
    def __init__(self, arg):
        self.transform_img = transforms.Resize(arg, Image.BILINEAR)
        self.transform_label = transforms.Resize(arg, Image.NEAREST)

    def __call__(self, sample):
        img, label = sample
        return self.transform_img(img), self.transform_label(label)


class Normalize(object):
    def __init__(self, mean, std):
        self.transform = transforms.Normalize(mean, std)
        
    def __call__(self, sample):
        img, label = sample
        # Only normalize image, NOT the label
        return self.transform(img), label


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        img, label = sample
        # Convert to numpy arrays
        label = np.array(label)
        img = np.array(img)

        # Convert to tensors with correct dimensions
        # Image: [C, H, W]
        # Label: [H, W]
        return torch.from_numpy(img.transpose((2, 0, 1))).float(), torch.from_numpy(label.copy()).long()


class RandomRescale(object):
    def __init__(self, min_ratio=0.5, max_ratio=1.0):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, sample):
        img, label = sample
        width, height = img.size
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        new_width, new_height = int(ratio * width), int(ratio * height)
        return img.resize((new_width, new_height)), label.resize((new_width, new_height))


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, label = sample
        if random.uniform(0, 1) > self.p:
            return transforms.functional.hflip(img), transforms.functional.hflip(label)
        else:
            return img, label


class RandomColor(object):
    def __init__(self, brightness=0, contrast=0.2, saturation=0, hue=0):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        img, label = sample
        # Only apply color jitter to the image, not the label
        return self.transform(img), label


class RandomRotation(object):
    def __init__(self, degree=[-5, 5]):
        self.degree = degree

    def __call__(self, sample):
        img, label = sample
        angle = transforms.RandomRotation.get_params(self.degree)
        img = transforms.functional.rotate(img, angle)
        label = transforms.functional.rotate(label, angle)
        return img, label


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img, label = sample
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=self.output_size)
        img = transforms.functional.crop(img, i, j, h, w)
        label = transforms.functional.crop(label, i, j, h, w)
        return img, label


def read_txt(file):
    tmp = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            tmp.append(line)
    return tmp