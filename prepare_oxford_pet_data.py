import os
import shutil
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import json
from pycocotools import mask as maskUtils
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform
import cv2
# Set paths
dataset_root = "/raid/home/dong.wang/data/oxford_iiit_pet"
output_root = "/raid/home/dong.wang/data/yolov5_oxford_pet"
images_dir = os.path.join(output_root, "images")
labels_dir = os.path.join(output_root, "labels")
segmentation_dir = os.path.join(output_root, "segmentation")

# Create target directories
os.makedirs(os.path.join(images_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(images_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(labels_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(labels_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(segmentation_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(segmentation_dir, "val"), exist_ok=True)

# Load OxfordIIITPet dataset
dataset = OxfordIIITPet(
    root=dataset_root,
    split="trainval",
    target_types=["category", "segmentation"],
    download=True,
)

# Class names
classes = dataset.classes  # Class names
class_to_index = {cls: idx for idx, cls in enumerate(classes)}

def generate_bbox_from_mask(mask):
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return x_min, y_min, x_max, y_max

def plot_bbox_on_mask(mask_array, bbox):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_array, cmap='gray')
    x, y, w, h = bbox
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
    plt.gca().add_patch(rect)
    plt.title('Mask with Bounding Box')
    plt.axis('off')
    plt.savefig('mask_with_bbox.png')
    plt.close()

def simplify_polygon(polygon, tolerance=2.0):
    """Simplify the polygon using Ramer-Douglas-Peucker algorithm."""
    poly = Polygon(polygon)
    simplified = poly.simplify(tolerance, preserve_topology=True)
    return list(simplified.exterior.coords)

def mask_to_polygon(mask):
    # Use a more stable contour extraction method
    contours = measure.find_contours(mask, 0.5)
    
    # Merge all contours into a single polygon
    if len(contours) > 1:
        # Sort by area, choose the largest contour as the main contour
        contours = sorted(contours, key=len, reverse=True)
    elif len(contours) == 0:
        return False
    contour = contours[0]
    contour = np.flip(contour, axis=1)
    
    # Use a more conservative simplification parameter
    poly = Polygon(contour)
    
    # Ensure the polygon is valid
    if not poly.is_valid:
        poly = poly.buffer(0)
    
    # Simplify the polygon but retain more details
    simplified = poly.simplify(tolerance=0.2, preserve_topology=True)
    
    # Handle possible MultiPolygon cases
    if isinstance(simplified, MultiPolygon):
        # Choose the largest polygon by area
        largest_poly = max(simplified.geoms, key=lambda p: p.area)
        return [list(largest_poly.exterior.coords)]
    else:
        return [list(simplified.exterior.coords)]
        
# Initialize COCO JSON structure
coco_json = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Add categories to COCO JSON
for idx, cls in enumerate(classes):
    coco_json["categories"].append({
        "id": idx,
        "name": cls,
        "supercategory": "pet"
    })

annotation_id = 0

# Organize dataset
for idx in tqdm(range(len(dataset)), desc="Processing Dataset"):
    image, (category, segmentation) = dataset[idx]
    class_idx = int(category)
    split = "train" if random.random() > 0.2 else "val"
    image_path = os.path.join(images_dir, split, f"{idx}.jpg")
    label_path = os.path.join(labels_dir, split, f"{idx}.txt")
    segmentation_path = os.path.join(segmentation_dir, split, f"{idx}.png")

    # Save image
    image.save(image_path)

    # Convert segmentation to a PIL image if necessary
    if not isinstance(segmentation, Image.Image):
        segmentation = Image.fromarray(segmentation)

    # Save segmentation mask
    segmentation.save(segmentation_path)

    # Add image info to COCO JSON
    image_info = {
        "id": int(idx),
        "file_name": f"{idx}.jpg",
        "height": int(image.height),
        "width": int(image.width)
    }
    coco_json["images"].append(image_info)

    # Choose the foreground value for the mask
    foreground_value = 1
    unsure_value = 3
    mask_array = (np.array(segmentation) == foreground_value) | (np.array(segmentation) == unsure_value)
    # Set the border pixels to False
    mask_array[0, :] = False
    mask_array[-1, :] = False
    mask_array[:, 0] = False
    mask_array[:, -1] = False
    encoded_mask = maskUtils.encode(np.asfortranarray(mask_array.astype(np.uint8)))
    bbox = maskUtils.toBbox(encoded_mask)
    # bbox = generate_bbox_from_mask(mask_array)
    # plot_bbox_on_mask(mask_array, bbox)
    x_min, y_min, width, height = map(float, bbox)
    # Convert mask to polygon format
    segmentation_polygons = mask_to_polygon(mask_array)
    if not segmentation_polygons:
        print("No segmentation polygons found for image: ", idx)
        continue

    # Add annotation info to COCO JSON
    annotation_info = {
        "id": int(annotation_id),
        "image_id": int(idx),
        "category_id": int(class_idx),
        "bbox": [x_min, y_min, width, height],
        "area": float(width * height),
        "segmentation": [list(map(float, np.array(poly).ravel())) for poly in segmentation_polygons],
        "iscrowd": 0
    }
    coco_json["annotations"].append(annotation_info)
    annotation_id += 1
    # if annotation_id>100:
    #     break


# Save COCO JSON
coco_json_path = os.path.join(output_root, "annotations.json")
with open(coco_json_path, "w") as f:
    json.dump(coco_json, f)

print(f"COCO FormatJSON saved at {coco_json_path}")

