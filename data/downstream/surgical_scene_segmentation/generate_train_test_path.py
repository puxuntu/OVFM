import os

def write_image_paths_train(data_folder, output_file):
    sub_folders = sorted(os.listdir(data_folder))[10:30]

    with open(output_file, 'w') as file:
        for sub_folder in sub_folders:
            img_folder = os.path.join(data_folder, sub_folder, 'img')
            mask_folder = os.path.join(data_folder, sub_folder, 'label')

            if not os.path.exists(img_folder) or not os.path.exists(mask_folder):
                continue

            img_files = os.listdir(img_folder)
            for img_file in img_files:
                img_path = os.path.abspath(os.path.join(img_folder, img_file))
                mask_path = os.path.abspath(os.path.join(mask_folder, img_file))
 
                if os.path.exists(mask_path):
                    file.write(f"{img_path} {mask_path}\n")
                else:
                    print(f"Warning: corresponding mask not found for {img_path}")

data_folder = 'data/downstream/surgical_scene_segmentation/Images-and-Supervisely-Annotations'
output_file = 'data/downstream/surgical_scene_segmentation/train_dataset.txt'

write_image_paths_train(data_folder, output_file)


def write_image_paths_val(data_folder, output_file):
    sub_folders = sorted(os.listdir(data_folder))[0:10]

    with open(output_file, 'w') as file:
        for sub_folder in sub_folders:
            img_folder = os.path.join(data_folder, sub_folder, 'img')
            mask_folder = os.path.join(data_folder, sub_folder, 'label')

            if not os.path.exists(img_folder) or not os.path.exists(mask_folder):
                continue  

            img_files = os.listdir(img_folder)
            for img_file in img_files:
                img_path = os.path.abspath(os.path.join(img_folder, img_file))
                mask_path = os.path.abspath(os.path.join(mask_folder, img_file))

                if os.path.exists(mask_path):
                    file.write(f"{img_path} {mask_path}\n")
                else:
                    print(f"Warning: corresponding mask not found for {img_path}")

data_folder = 'data/downstream/surgical_scene_segmentation/Images-and-Supervisely-Annotations'
output_file = 'data/downstream/surgical_scene_segmentation/test_dataset.txt'

write_image_paths_val(data_folder, output_file)
