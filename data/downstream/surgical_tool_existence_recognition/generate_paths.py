import os
from natsort import natsorted

root_folder = 'data/downstream/surgical_tool_existence_recognition/data_resize'
test_file = 'data/downstream/surgical_tool_existence_recognition/test.txt'
train_file = 'data/downstream/surgical_tool_existence_recognition/train.txt'

with open(test_file, 'w') as test_f, open(train_file, 'w') as train_f:
    for i in range(1, 26):
        subfolder = os.path.join(root_folder, f'{i:02d}')
        if os.path.exists(subfolder):
            for root, _, files in os.walk(subfolder):
                for file in natsorted(files):  
                    if file.endswith(('jpg', 'png', 'jpeg')):  
                        file_path = os.path.abspath(os.path.join(root, file))
                        file_path = file_path.replace('\\', '/')  
                        test_f.write(file_path + '\n')
    
    for i in range(26, 51):
        subfolder = os.path.join(root_folder, f'{i:02d}')
        if os.path.exists(subfolder):
            for root, _, files in os.walk(subfolder):
                for file in natsorted(files):  
                    if file.endswith(('jpg', 'png', 'jpeg')):  
                        file_path = os.path.abspath(os.path.join(root, file))
                        file_path = file_path.replace('\\', '/')
                        train_f.write(file_path + '\n')

print('Finished!')
