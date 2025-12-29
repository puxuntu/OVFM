import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def copy_file(src, dst):
    try:
        shutil.copy2(src, dst)
    except shutil.SameFileError:
        pass
    except Exception as e:
        print(f"Error copying file from {src} to {dst}: {e}")

def process_video(src, dst):
    try:
        dst_mp4 = os.path.splitext(dst)[0] + '.mp4'
        if os.path.exists(dst_mp4):
            print(f"Target file {dst_mp4} already exists, skipping processing")
            return

        ffmpeg_path = r'Your ffmpeg.exe path'
        command = [
            ffmpeg_path, '-i', src,
            '-vf', 'scale=iw/4:ih/4',
            '-c:v', 'h264_nvenc',
            '-preset', 'fast',
            '-crf', '23',
            dst_mp4
        ]
        subprocess.run(command, check=True, text=True, encoding='utf-8')
    except Exception as e:
        print(f"Error processing video {src}: {e}")

def process_folder(src_folder, dst_folder):
    tasks = []
    total_files = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for root, _, files in os.walk(src_folder):
            relative_path = os.path.relpath(root, src_folder)
            target_dir = os.path.join(dst_folder, relative_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, file)

                # Check if the file already exists in the destination folder
                if not os.path.exists(dst_file):
                    if file.lower().endswith('.mpg') or file.lower().endswith('.mp4'):
                        tasks.append(executor.submit(process_video, src_file, dst_file))
                    else:
                        tasks.append(executor.submit(copy_file, src_file, dst_file))
                    total_files += 1

        with tqdm(total=total_files, desc="Processing files") as pbar:
            for task in as_completed(tasks):
                task.result()
                pbar.update(1)

if __name__ == "__main__":
    src_main_folder = "Your_scr_folder"
    dst_main_folder = "Your_sat_folder"
    process_folder(src_main_folder, dst_main_folder)
