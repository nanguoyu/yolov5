import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import numpy as np
import os
from PIL import ImageOps
from shapely.geometry import Polygon

dataset_root = "/raid/home/dong.wang/data/oxford_iiit_pet"
output_root = "/raid/home/dong.wang/data/yolov5_oxford_pet"
images_dir = os.path.join(output_root, "images")
labels_dir = os.path.join(output_root, "labels")
segmentation_dir = os.path.join(output_root, "segmentation")

coco_json_path = os.path.join(output_root, "coco_annotations.json")
# Load COCO JSON
with open(coco_json_path, "r") as f:
    coco_data = json.load(f)

# Randomly select an image and its annotations
# image_info = random.choice(coco_data["images"])
image_info = coco_data["images"][95]
image_id = image_info["id"]
print("image_id: ", image_id)
annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

# Load the image
image_path = os.path.join(images_dir, "train", image_info["file_name"])
image = Image.open(image_path)

# Load the original segmentation mask
segmentation_path = os.path.join(segmentation_dir, "train", f"{image_id}.png")
segmentation_mask = Image.open(segmentation_path).convert("L")
segmentation_array = np.array(segmentation_mask)

# Plot the image
fig, ax = plt.subplots(1)
ax.imshow(image)

# Overlay the original segmentation mask
ax.imshow(segmentation_array, cmap='jet', alpha=0.5)

# Plot each annotation
for ann in annotations:
    # Plot bbox
    x_min, y_min, width, height = ann["bbox"]
    print("x_min: ", x_min, "y_min: ", y_min, "width: ", width, "height: ", height)
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Plot simplified segmentation
    for seg in ann["segmentation"]:
        poly = np.array(seg).reshape((len(seg) // 2, 2))
        polygon = patches.Polygon(poly, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(polygon)

# Save the plot
plt.axis('off')
plt.savefig("test_coco_format.jpg", bbox_inches='tight', pad_inches=0)
plt.close()

print("Image with annotations saved as test_coco_format.jpg")

# New code: Randomly select a rotation angle from (0,180)
rotation_angle = random.randint(0, 180)
print("Rotation angle: ", rotation_angle)

# Rotate the original image
rotated_image = image.rotate(rotation_angle, expand=False)

# Rotate the original segmentation mask
rotated_segmentation_mask = segmentation_mask.rotate(rotation_angle, expand=True)
rotated_segmentation_array = np.array(rotated_segmentation_mask)

# Define a function to rotate points
def rotate_points(points, angle, image_size):
    angle_rad = np.deg2rad(-angle)  # Negating the angle to match image rotation
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    cx, cy = image_size[0] / 2, image_size[1] / 2
    rotated_points = []
    for x, y in points:
        x_new = cos_angle * (x - cx) - sin_angle * (y - cy) + cx
        y_new = sin_angle * (x - cx) + cos_angle * (y - cy) + cy
        rotated_points.append((x_new, y_new))
    return rotated_points

# Clip the bounding box to the image boundaries
def clip_bbox(bbox, image_size):
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    x_min = max(0, min(x_min, image_size[0] - 1))
    y_min = max(0, min(y_min, image_size[1] - 1))
    x_max = max(0, min(x_max, image_size[0] - 1))
    y_max = max(0, min(y_max, image_size[1] - 1))

    return [x_min, y_min, x_max - x_min, y_max - y_min]

# Calculate the rotated coco segmentation and generate a new bbox
fig, ax = plt.subplots(1)
ax.imshow(rotated_image)

for ann in annotations:
    # Rotate segmentation
    new_segmentation = []
    for seg in ann["segmentation"]:
        poly = np.array(seg).reshape((len(seg) // 2, 2))
        rotated_poly = rotate_points(poly, rotation_angle, rotated_image.size)
        new_segmentation.append(np.array(rotated_poly).flatten().tolist())

    # Calculate new bbox from the rotated segmentation
    polygon = Polygon(rotated_poly)
    min_x, min_y, max_x, max_y = polygon.bounds
    new_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

    # Clip the new bbox to the image boundaries
    new_bbox = clip_bbox(new_bbox, rotated_image.size)
    print("Clipped new bbox: ", new_bbox)

    # Draw the clipped new bbox
    rect = patches.Rectangle((new_bbox[0], new_bbox[1]), new_bbox[2], new_bbox[3], linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # Draw the rotated segmentation
    for seg in new_segmentation:
        poly = np.array(seg).reshape((len(seg) // 2, 2))
        polygon = patches.Polygon(poly, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(polygon)

# Save the rotated image
plt.axis('off')
plt.savefig("rotated_test_coco_format.jpg", bbox_inches='tight', pad_inches=0)
plt.close()

print("Rotated image with annotations saved as rotated_test_coco_format.jpg")