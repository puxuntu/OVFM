import pandas as pd

csv_file_path = 'data/downstream/complication_detection/train.csv'
df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

output_txt_file = 'data/downstream/complication_detection/train.txt'
with open(output_txt_file, 'w') as txt_file:
    for index, row in df.iterrows():
        video_index = int(row[0])  
        label = int(row[1])        

        print(video_index)
        
        if label in [0]:
            label = 0
        elif label in [1]:
            label = 1
        
        line = f"data/downstream/complication_detection/videos/{video_index}.mp4,{label}"
        
        txt_file.write(line + '\n')

print("File written successfully!")

import pandas as pd

csv_file_path = 'data/downstream/complication_detection/test.csv'
df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')  

output_txt_file = 'data/downstream/complication_detection/test.txt'
with open(output_txt_file, 'w') as txt_file:
    for index, row in df.iterrows():
        video_index = int(row[0])  
        label = int(row[1])        
        
        if label in [0]:
            label = 0
        elif label in [1]:
            label = 1
        
        line = f"data/downstream/complications_detection/videos/{video_index}.mp4,{label}"
        
        txt_file.write(line + '\n')

print("File written successfully!")

