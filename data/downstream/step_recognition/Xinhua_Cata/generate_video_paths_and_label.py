import os
import csv

train_file = open('data/downstream/step_recognition/Xinhua_Cata/train_4_phases_images.txt', 'w')
val_file = open('data/downstream/step_recognition/Xinhua_Cata/val_4_phases_images.txt', 'w')

for i in range(1, 68):
    file_num = f"{i:02d}"
    
    csv_path = f"data/downstream/step_recognition/Xinhua_Cata/phase_annotations_phase_recognition/video{file_num}.csv"
    
    try:
        with open(csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            header = next(csv_reader)

            for row_idx, row in enumerate(csv_reader, 1):  
                frame_name = row[0]
                phase_name = row[1]
                
                string_a = f"data/downstream/step_recognition/Xinhua_Cata/data_resize_phase_recognition/{file_num}/{frame_name}.jpg"
                string_b = phase_name
                
                if string_b in ["0", "1"]:
                    should_write = True  
                elif string_b in ["7"]:
                    should_write = True 
                    string_b = "2"
                else:
                    should_write = row_idx % 5 == 0  
                    string_b = "3"
                
                if i < 46 and should_write:
                    train_file.write(f"{string_a}, {string_b}\n")
                
                if i >= 46 and should_write:
                    val_file.write(f"{string_a}, {string_b}\n")
    
    except FileNotFoundError:
        print(f"File {csv_path} not found, skipping...")
        continue

# Close output files
train_file.close()
val_file.close()

print("Processing completed!")
