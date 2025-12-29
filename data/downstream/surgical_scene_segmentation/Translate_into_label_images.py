import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

def create_label_images(root_dir):
    """
    Convert Supervisely JSON annotations to label images and colored visualizations.
    Only processes anatomical structures (Cornea, Pupil, Lens).
    
    Args:
        root_dir: Root directory containing case folders
    """

    anatomical_names = ['Cornea', 'Pupil', 'Lens']
    
    class_mapping = {
        'Pupil': 1,
        'Cornea': 2,
        'Lens': 3
    }
    
    color_map = {
        0: [0, 0, 0],       # Background (black)
        1: [255, 0, 0],     # Pupil (blue)
        2: [0, 255, 0],     # Cornea (green)
        3: [255, 0, 255]    # Lens (magenta)
    }
    
    processed_files = 0
    error_files = 0
    
    case_folders = [d for d in Path(root_dir).iterdir() if d.is_dir()]
    print(f"Found {len(case_folders)} case folders")
    
    for case_folder in tqdm(case_folders, desc="Processing case folders"):
        ann_folder = case_folder / "ann"
        img_folder = case_folder / "img"
        
        label_folder = case_folder / "label"
        label_folder.mkdir(exist_ok=True)
        
        vis_folder = case_folder / "visualization"
        vis_folder.mkdir(exist_ok=True)
        
        if not ann_folder.exists() or not img_folder.exists():
            print(f"Skipping {case_folder.name}: Missing ann or img folder")
            continue
        
        json_files = list(ann_folder.glob("*.json"))
        
        for json_path in json_files:
            try:
                base_name = json_path.stem
                if base_name.endswith('.png'):
                    base_name = base_name[:-4]
                
                img_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    possible_path = img_folder / f"{base_name}{ext}"
                    if possible_path.exists():
                        img_path = possible_path
                        break
                
                if img_path is None:
                    print(f"Warning: No matching image found for {json_path}")
                
                with open(json_path, 'r') as f:
                    ann_data = json.load(f)
                
                height = ann_data['size']['height']
                width = ann_data['size']['width']
                
                label_img = np.zeros((height, width), dtype=np.uint8)
                
                anatomical_objects = []
                
                for obj in ann_data['objects']:
                    class_title = obj['classTitle']
                    
                    if class_title in anatomical_names:
                        anatomical_objects.append(obj)
                    else:
                        continue
                
                def get_polygon_area(obj):
                    points = obj['points']['exterior']
                    if len(points) < 3:
                        return 0
                    points_array = np.array(points, dtype=np.int32)
                    return cv2.contourArea(points_array)
                
                anatomical_objects_sorted = sorted(anatomical_objects, 
                                                 key=get_polygon_area, 
                                                 reverse=True)
                
                for obj in anatomical_objects_sorted:
                    class_title = obj['classTitle']
                    pixel_value = class_mapping[class_title]
                    
                    points = obj['points']['exterior']
                    if len(points) < 3:  
                        continue
                    
                    points = np.array(points, dtype=np.int32)
                    
                    cv2.fillPoly(label_img, [points], pixel_value)
                
                label_path = label_folder / f"{base_name}.png"
                cv2.imwrite(str(label_path), label_img)
                
                colored_img = np.zeros((height, width, 3), dtype=np.uint8)
                for class_id, color in color_map.items():
                    mask = (label_img == class_id)
                    colored_img[mask] = color
                
                vis_path = vis_folder / f"{base_name}_colored.png"
                cv2.imwrite(str(vis_path), colored_img)
                
                processed_files += 1
                
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
                error_files += 1
    
    print(f"Completed: Processed {processed_files} files successfully ({error_files} errors)")
    print(f"Original label images saved to 'label' folders")
    print(f"Colored visualization images saved to 'visualization' folders")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Supervisely JSON annotations to label images with colored visualizations (anatomical structures only)")
    parser.add_argument("--root", default="data/downstream/surgical_scene_segmentation/Images-and-Supervisely-Annotations", 
                        help="Root directory containing the case folders")
    args = parser.parse_args()
    
    create_label_images(args.root)