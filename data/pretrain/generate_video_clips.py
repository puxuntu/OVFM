import cv2
import os

# List of input folders
input_folders = [
    "Your_video_list"
]

# Output folder where processed videos will be saved
output_folder = "data/pretrain/OVFM-pretrain_data/"

os.makedirs(output_folder, exist_ok=True)

def process_video(input_path):
    try:
        # Open the video using OpenCV
        video = cv2.VideoCapture(input_path)
        if not video.isOpened():
            print(f"Error opening video file {input_path}")
            return
        
        # Get the total number of frames and the frame rate
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        num_segments = int(total_frames // (fps * 25))  # Number of segments based on 25-second intervals

        folder_name = os.path.basename(os.path.dirname(input_path))

        for i in range(num_segments):
            start_frame = int(i * fps * 25)  # Start frame for the 25-second interval
            end_frame = start_frame + int(fps * 5)  # End frame for the 5-second clip
            
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Move to start frame
            
            frames = []
            for frame_num in range(start_frame, end_frame):
                ret, frame = video.read()  # Read a frame
                if not ret:
                    print(f"Error reading frame {frame_num} in video {input_path}")
                    break
                frames.append(frame)
            
            if frames:
                # Resize the frames to 480x270
                resized_frames = [cv2.resize(frame, (480, 270)) for frame in frames]

                # Write the frames to an output video
                output_filename = f"{folder_name}_{os.path.splitext(os.path.basename(input_path))[0]}_segment_{i+1}.mp4"
                output_path = os.path.join(output_folder, output_filename)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for writing video
                out_video = cv2.VideoWriter(output_path, fourcc, fps, (480, 270))
                
                for resized_frame in resized_frames:
                    out_video.write(resized_frame)

                out_video.release()  # Release the output video writer
            
        video.release()  # Release the input video file

    except Exception as e:
        print(f"Error processing video {input_path}: {e}")

def process_all_folders(input_folders):
    for input_folder in input_folders:
        for filename in os.listdir(input_folder):
            if filename.endswith(".mp4"):
                input_path = os.path.join(input_folder, filename)
                print(f"Processing video: {input_path}")
                process_video(input_path)

process_all_folders(input_folders)

print("Finished!")
