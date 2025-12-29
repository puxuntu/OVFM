import os
import re

image_folder = "./data/downstream/segmentation/Xinhua_Cata/data_resize_feature_extractor"
label_folder = "./data/downstream/segmentation/Xinhua_Cata/segmentation_labels_feature_extractor"

train_image_paths = []
train_label_paths = []
for i in range(1, 28):
    image_subfolder = os.path.join(image_folder, f"{i:02d}")
    label_subfolder = os.path.join(label_folder, f"{i:02d}")

    if not os.path.exists(image_subfolder) or not os.path.exists(label_subfolder):
        continue

    for image_file, label_file in zip(os.listdir(image_subfolder), os.listdir(label_subfolder)):
        if image_file.endswith('.jpg'):
            train_image_paths.append(os.path.join(image_subfolder, image_file))
            train_label_paths.append(os.path.join(label_subfolder, image_file.replace(".jpg", ".png")))

with open("./data/downstream/segmentation/Xinhua_Cata/train_dataset.txt", "w") as f:
    for image_path, label_path in zip(train_image_paths, train_label_paths):
        f.write(f"{image_path} {label_path}\n")

test_image_paths = []
test_label_paths = []
for i in range(28, 51):
    image_subfolder = os.path.join(image_folder, f"{i:02d}")
    label_subfolder = os.path.join(label_folder, f"{i:02d}")
    if not os.path.exists(image_subfolder) or not os.path.exists(label_subfolder):
        continue
    for image_file, label_file in zip(os.listdir(image_subfolder), os.listdir(label_subfolder)):
        if image_file.endswith('.jpg'):
            test_image_paths.append(os.path.join(image_subfolder, image_file))
            test_label_paths.append(os.path.join(label_subfolder, image_file.replace(".jpg", ".png")))

with open("./data/downstream/segmentation/Xinhua_Cata/test_dataset.txt", "w") as f:
    for image_path, label_path in zip(test_image_paths, test_label_paths):
        f.write(f"{image_path} {label_path}\n")
