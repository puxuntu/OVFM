import os
import json
import cv2
import numpy as np
import shutil

def process_annotations():
    input_folder = "data/downstream/necleus_tracking/frames_with_json_labels"
    output_folder = "data/downstream/necleus_tracking/frames_label_visualization"
    output_file = "data/downstream/necleus_tracking/annotation_data.txt"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(output_file, 'w') as f_out:
        f_out.write("Image_Name,Center_X,Center_Y,Width,Height\n")
        
        json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
        
        for json_file in json_files:
            try:
                with open(os.path.join(input_folder, json_file), 'r') as f:
                    data = json.load(f)
                
                if not data.get('shapes'):
                    print(f"No shapes found in {json_file}. Skipping...")
                    continue
                
                image_name = data.get('imagePath')
                image_path = os.path.join(input_folder, image_name)
                
                if not os.path.exists(image_path):
                    print(f"Image {image_name} not found. Skipping...")
                    continue
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load {image_name}. Skipping...")
                    continue
                
                vis_image = image.copy()
                
                max_area = 0
                selected_shape = None
                
                for shape in data['shapes']:
                    if shape['shape_type'] == 'rectangle' and len(shape['points']) == 2:
                        top_left = shape['points'][0]
                        bottom_right = shape['points'][1]
                        
                        width = abs(bottom_right[0] - top_left[0])
                        height = abs(bottom_right[1] - top_left[1])
                        
                        area = width * height
                        
                        if area > max_area:
                            max_area = area
                            selected_shape = shape
                
                if selected_shape is None:
                    print(f"No valid rectangle found in {json_file}. Skipping...")
                    continue
                
                top_left = selected_shape['points'][0]
                bottom_right = selected_shape['points'][1]
                
                x1, y1 = min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])
                x2, y2 = max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                f_out.write(f"{image_name},{center_x:.2f},{center_y:.2f},{width:.2f},{height:.2f}\n")
                
                cv2.rectangle(vis_image, 
                              (int(x1), int(y1)), 
                              (int(x2), int(y2)), 
                              (0, 255, 0), 2)  
                
                cv2.circle(vis_image, 
                           (int(center_x), int(center_y)), 
                           5, (0, 0, 255), -1)  
                
                vis_path = os.path.join(output_folder, image_name)
                cv2.imwrite(vis_path, vis_image)
                
                print(f"Processed {image_name}")
                
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
    
    print(f"Processing complete. Results saved to {output_file} and {output_folder}")

if __name__ == "__main__":
    process_annotations()