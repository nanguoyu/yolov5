# import json

# def check_annotations(coco_annotation_path):
#     with open(coco_annotation_path, 'r') as file:
#         data = json.load(file)

#     images = {image['id']: image for image in data['images']}
#     annotations = data['annotations']

#     annotated_image_ids = {annotation['image_id'] for annotation in annotations}

#     unannotated_images = [image for image_id, image in images.items() if image_id not in annotated_image_ids]

#     if unannotated_images:
#         print("The following images do not have annotations:")
#         for image in unannotated_images:
#             print(f"Image ID: {image['id']}, File Name: {image['file_name']}")
#     else:
#         print("All images have annotations.")

# # Example usage
# coco_annotation_path = "/raid/home/dong.wang/data/yolov5_oxford_pet/annotations/annotations.json"
# check_annotations(coco_annotation_path)


# # Image ID: 1404, File Name: Saint_Bernard_8.jpg
# # Image ID: 2407, File Name: Egyptian_Mau_61.jpg
# # Image ID: 2410, File Name: Egyptian_Mau_64.jpg
# # Image ID: 2437, File Name: Egyptian_Mau_91.jpg
# # Image ID: 2833, File Name: Leonberger_98.jpg
# # Image ID: 2889, File Name: Miniature_Pinscher_54.jpg
# # Image ID: 3246, File Name: Saint_Bernard_65.jpg


import os

def check_labels(base_path):
    for subset in ['train', 'valid']:
        images_path = os.path.join(base_path, subset, 'images')
        labels_path = os.path.join(base_path, subset, 'labels')

        image_files = {os.path.splitext(f)[0] for f in os.listdir(images_path) if f.endswith('.jpg')}
        label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_path) if f.endswith('.txt')}

        missing_labels = image_files - label_files

        if missing_labels:
            print(f"The following images in {subset} do not have corresponding labels:")
            for image in missing_labels:
                print(f"Image: {image}.jpg")
        else:
            print(f"All images in {subset} have corresponding labels.")

# Example usage
base_path = "/raid/home/dong.wang/data/oxford_pet_roboflow"
check_labels(base_path)

