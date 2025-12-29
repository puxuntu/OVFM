import pandas as pd

csv_file_path = 'data/downstream/surgical_skill_assessment/train.csv'
df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')  

output_txt_file = 'data/downstream/surgical_skill_assessment/train.txt'
with open(output_txt_file, 'w') as txt_file:
    for index, row in df.iterrows():
        video_index = int(row[0])  
        label = int(row[1])        
        
        if label in [1, 2, 3]:
            label = 0
        elif label in [4, 5]:
            label = 1
        
        line = f"data/downstream/surgical_skill_assessment/videos/{video_index}.mp4,{label}"
        
        txt_file.write(line + '\n')

print("File written successfully!")

import pandas as pd

csv_file_path = 'data/downstream/surgical_skill_assessment/test.csv'
df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')  

output_txt_file = 'data/downstream/surgical_skill_assessment/test.txt'
with open(output_txt_file, 'w') as txt_file:
    for index, row in df.iterrows():
        video_index = int(row[0])  
        label = int(row[1])        
        
        if label in [1, 2, 3]:
            label = 0
        elif label in [4, 5]:
            label = 1
        
        line = f"data/downstream/surgical_skill_assessment/videos/{video_index}.mp4,{label}"
        
        txt_file.write(line + '\n')

print("File written successfully!")

