import os
import shutil

target_dir = "/raid/home/dong.wang/data/yolov5_oxford_pet/"
images_sub_dir = [
    "images/train",
    "images/val"
]
labels_sub_dir = [
    "labels/train",
    "labels/val"
]

source_dir = "/raid/home/dong.wang/data/yolov5_oxford_pet_annotations/labels/annotations"

def copy_annotations():
    for image_sub, label_sub in zip(images_sub_dir, labels_sub_dir):
        image_path = os.path.join(target_dir, image_sub)
        label_path = os.path.join(target_dir, label_sub)
        
        # Ensure the label directory exists
        os.makedirs(label_path, exist_ok=True)
        
        for image_file in os.listdir(image_path):
            if image_file.endswith('.jpg'):  # Assuming image files are .jpg
                base_name = os.path.splitext(image_file)[0]
                source_file = os.path.join(source_dir, f"{base_name}.txt")
                target_file = os.path.join(label_path, f"{base_name}.txt")
                
                if os.path.exists(source_file):
                    shutil.copy(source_file, target_file)
                    print(f"Copied {source_file} to {target_file}")
                else:
                    print(f"Annotation for {image_file} not found.")

copy_annotations()

