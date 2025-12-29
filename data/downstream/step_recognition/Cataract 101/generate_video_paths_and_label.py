import os
import csv

folder_a_path = 'data/downstream/step_recognition/Cataract 101/video_clips'
folder_b_path = 'data/downstream/step_recognition/Cataract 101/phase_annotations_phase_recognition'

train_output_file_path = 'data/downstream/step_recognition/Cataract 101/train.txt'
val_output_file_path = 'data/downstream/step_recognition/Cataract 101/val.txt'

train_folders = range(1, 74)
val_folders = range(74, 102)

with open(train_output_file_path, 'w') as train_output_file:
    for folder_number in train_folders:
        folder_name = f"{folder_number:03d}"
        folder_path = os.path.join(folder_a_path, folder_name)
        if os.path.isdir(folder_path):
            csv_file_path = os.path.join(folder_b_path, 'video' + folder_name + '.csv')
            if os.path.exists(csv_file_path):
                with open(csv_file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    next(csv_reader)  
                    row_counter = 0  
                    for row in csv_reader:
                        row_counter += 1
                        frame_name = row[0] + ".mp4"
                        phase_name = row[1]
                        video_path = os.path.join(folder_path, frame_name).replace("\\", "/")
                        if os.path.exists(video_path):
                            if phase_name in ['0', '2', '7', '8']:
                                train_output_file.write(f'{video_path},{phase_name}\n')
                            elif row_counter % 5 == 0:
                                train_output_file.write(f'{video_path},{phase_name}\n')

with open(val_output_file_path, 'w') as val_output_file:
    for folder_number in val_folders:
        folder_name = f"{folder_number:03d}"
        folder_path = os.path.join(folder_a_path, folder_name)
        if os.path.isdir(folder_path):
            csv_file_path = os.path.join(folder_b_path, 'video' + folder_name + '.csv')
            if os.path.exists(csv_file_path):
                with open(csv_file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    next(csv_reader)  
                    row_counter = 0
                    for row in csv_reader:
                        row_counter += 1
                        frame_name = row[0] + ".mp4"
                        phase_name = row[1]
                        video_path = os.path.join(folder_path, frame_name).replace("\\", "/")
                        if os.path.exists(video_path):
                            if phase_name in ['0', '2', '7', '8']:
                                val_output_file.write(f'{video_path},{phase_name}\n')
                            elif row_counter % 5 == 0:
                                val_output_file.write(f'{video_path},{phase_name}\n')
