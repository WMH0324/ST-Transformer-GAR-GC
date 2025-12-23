import os
import shutil

def delete_files_in_subdirectories(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for directory in dirs:
            # 构建子目录的完整路径
            current_dir = os.path.join(root, directory)

            # 删除子目录中的 "frame" 文件夹
            frame_folder = os.path.join(current_dir, "frame")
            if os.path.exists(frame_folder):
                shutil.rmtree(frame_folder)
                print(f"Deleted {frame_folder}")

            # 删除子目录中的 "another_file.txt" 文件
            another_file = os.path.join(current_dir, "MOT16-03-results.mp4")
            if os.path.exists(another_file):
                os.remove(another_file)
                print(f"Deleted {another_file}")

# 指定要操作的根文件夹
root_folder_path = "E://VLBNewLabel"

# 调用函数删除文件夹中子目录中的文件
delete_files_in_subdirectories(root_folder_path)
