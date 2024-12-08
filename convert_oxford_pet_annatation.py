from ultralytics.data.converter import convert_coco

path = "/raid/home/dong.wang/data/yolov5_oxford_pet/annotations"
save_dir = "/raid/home/dong.wang/data/yolov5_oxford_pet_annotations/"
convert_coco(labels_dir=path, save_dir=save_dir, use_segments=True, use_keypoints=False, cls91to80=False)
