import os
import re
from collections import defaultdict

def split_dataset_by_video_number():
    input_file = "data/downstream/necleus_tracking/annotation_data.txt"
    train_file = "data/downstream/necleus_tracking/train.txt"
    test_file = "data/downstream/necleus_tracking/test.txt"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run the annotation processor first.")
        return
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    header = lines[0]
    annotation_lines = lines[1:]
    
    train_annotations = []
    test_annotations = []
    
    found_videos = defaultdict(int)
    
    video_pattern = re.compile(r'video_(\d+)_\d+\.jpg', re.IGNORECASE)
    
    for line in annotation_lines:
        parts = line.strip().split(',')
        if len(parts) >= 5:  
            image_name = parts[0]
            
            match = video_pattern.match(image_name)
            if match:
                try:
                    video_number = int(match.group(1))
                    found_videos[video_number] += 1
                    
                    if 1 <= video_number <= 50:
                        train_annotations.append(line)
                    if 51 <= video_number <= 70:
                        test_annotations.append(line)
                    else:
                        print(f"Skipping video {video_number} (outside of ranges 1-50 and 51-70)")
                except ValueError:
                    print(f"Warning: Could not convert '{match.group(1)}' to integer for {image_name}")
            else:
                print(f"Warning: Could not extract video number from {image_name}")
    
    with open(train_file, 'w') as f:
        f.write(header)  
        for line in train_annotations:
            f.write(line)
    
    with open(test_file, 'w') as f:
        f.write(header)  
        for line in test_annotations:
            f.write(line)
    
    train_video_count = sum(1 for video_num in found_videos if 1 <= video_num <= 50)
    test_video_count = sum(1 for video_num in found_videos if 51 <= video_num <= 70)
    
    print(f"Dataset split complete:")
    print(f"  - Training set (videos 1-50): {train_video_count} videos, {len(train_annotations)} annotations, saved to {train_file}")
    print(f"  - Testing set (videos 51-70): {test_video_count} videos, {len(test_annotations)} annotations, saved to {test_file}")
    
    print("\nVideos found in training range (1-50):")
    train_videos = sorted([v for v in found_videos.keys() if 1 <= v <= 50])
    for i, video_num in enumerate(train_videos):
        print(f"  Video {video_num}: {found_videos[video_num]} frames")
    
    print("\nVideos found in testing range (51-70):")
    test_videos = sorted([v for v in found_videos.keys() if 51 <= v <= 70])
    for i, video_num in enumerate(test_videos):
        print(f"  Video {video_num}: {found_videos[video_num]} frames")
    
    missing_train = [v for v in range(1, 51) if v not in found_videos]
    missing_test = [v for v in range(51, 71) if v not in found_videos]
    
    if missing_train:
        print("\nWarning: The following training videos (1-50) were not found:")
        print(f"  {', '.join(map(str, missing_train))}")
    
    if missing_test:
        print("\nWarning: The following testing videos (51-70) were not found:")
        print(f"  {', '.join(map(str, missing_test))}")

if __name__ == "__main__":
    split_dataset_by_video_number()