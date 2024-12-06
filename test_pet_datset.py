import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import random

def generate_bbox_from_mask(mask, foreground_value):
    print("mask: ", mask)
    # Ensure the mask is single-channel
    print("mask shape: ", mask.shape)
    if mask.ndim == 3:
        print("mask is multi-channel")
        mask = mask[:, :, 0]  # Use the first channel if it's a multi-channel mask

    foreground = (mask == foreground_value)
    
    coords = np.column_stack(np.where(foreground))
    if coords.size == 0:  # 如果没有目标，返回 None
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    # Plot the mask and bounding box
    # plt.figure()
    # plt.imshow(mask, cmap='gray')
    # plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
    #                                   edgecolor='red', facecolor='none', linewidth=2))
    # plt.title('Bounding Box on Mask')
    # plt.axis('off')

    # # Save the plot
    # plt.savefig("test1.jpg", bbox_inches='tight')
    # plt.close()
    return x_min, y_min, x_max, y_max

def rotate_image(image, angle):
    # Get the image size
    (h, w) = image.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)
    # Perform the rotation with border replication
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated, M

def rotate_bbox(bbox, M):
    # Unpack the bounding box
    x_min, y_min, x_max, y_max = bbox
    # Create a list of points
    points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    # Apply the rotation matrix
    rotated_points = cv2.transform(np.array([points]), M)[0]
    # Get the new bounding box
    x_min, y_min = rotated_points.min(axis=0)
    x_max, y_max = rotated_points.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)

image_path = '/raid/home/dong.wang/data/yolov5_oxford_pet/images/train/56.jpg'
mask_path = '/raid/home/dong.wang/data/yolov5_oxford_pet/segmentation/train/56.png'
annotation_path = '/raid/home/dong.wang/data/yolov5_oxford_pet/labels/train/56.txt'

mask = cv2.imread(mask_path)  # 读取为三通道图像

# 检查是否成功读取
if mask is None:
    raise FileNotFoundError(f"无法读取掩码：{mask_path}")

# 将掩码展平为一维数组，并找到所有唯一值
unique_values, counts = np.unique(mask.reshape(-1, 3), axis=0, return_counts=True)

# 输出结果
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")

output_path = 'visualized_trimap.png'


visualized_mask = np.zeros_like(mask)

visualized_mask[np.all(mask == [1, 1, 1], axis=-1)] = [255, 0, 0]  # 类别 [1,1,1] -> 蓝色
visualized_mask[np.all(mask == [2, 2, 2], axis=-1)] = [0, 255, 0]  # 类别 [2,2,2] -> 绿色
visualized_mask[np.all(mask == [3, 3, 3], axis=-1)] = [0, 0, 255]  # 类别 [3,3,3] -> 红色

# 保存伪彩色图像
cv2.imwrite(output_path, visualized_mask)
print(f"伪彩色掩码已保存到: {output_path}")

# [1,1,1] -> 蓝色 [255, 0, 0]。前景
# [2,2,2] -> 绿色 [0, 255, 0]。背景
# [3,3,3] -> 红色 [0, 0, 255]。未知区域

# 根据掩码中前景值区域来抠出原图并保存

# 读取原始图像
image = cv2.imread(image_path)

# 检查是否成功读取
if image is None:
    raise FileNotFoundError(f"无法读取图像：{image_path}")

# 创建一个二值掩码，仅保留前景区域
foreground_mask = np.all(mask == [1, 1, 1], axis=-1).astype(np.uint8) * 255
foreground_unknown_mask = (np.all(mask == [1, 1, 1], axis=-1) | np.all(mask == [3, 3, 3], axis=-1)).astype(np.uint8) * 255

# 应用掩码到原始图像
foreground = cv2.bitwise_and(image, image, mask=foreground_mask)

# 保存抠出的前景图像
foreground_output_path = 'extracted_foreground.png'
cv2.imwrite(foreground_output_path, foreground)
print(f"前景图像已保存到: {foreground_output_path}")

# Read bounding box from annotation file
with open(annotation_path, 'r') as file:
    lines = file.readlines()

# Draw bounding boxes on the extracted foreground image
for line in lines:
    class_idx, x_center, y_center, width, height = map(float, line.strip().split())
    image_width, image_height = image.shape[1], image.shape[0]
    
    # Convert YOLO format to bounding box coordinates
    x_min = int((x_center - width / 2) * image_width)
    y_min = int((y_center - height / 2) * image_height)
    x_max = int((x_center + width / 2) * image_width)
    y_max = int((y_center + height / 2) * image_height)
    
    # Draw rectangle on the foreground image
    cv2.rectangle(foreground, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)  # Yellow color for bbox

# Save the image with bounding boxes
foreground_with_bbox_output_path = 'extracted_foreground_with_bbox.png'
cv2.imwrite(foreground_with_bbox_output_path, foreground)
print(f"前景图像与边界框已保存到: {foreground_with_bbox_output_path}")

# Rotate the original image and mask
angle = 45  # Example angle
rotated_image, M = rotate_image(image, angle)
print(mask)
print("-----")

original_foreground = (mask == [1, 1, 1])
cv2.imwrite("original_foreground.png", original_foreground[:, :, 0].astype(np.uint8) * 255)

# Rotate the mask and convert to grayscale
rotated_mask, _ = rotate_image(mask, angle)
rotated_foreground = (rotated_mask == [1, 1, 1])
cv2.imwrite("rotated_foreground.png", rotated_foreground[:, :, 0].astype(np.uint8) * 255)


unique_values, counts = np.unique(rotated_mask.reshape(-1, 3), axis=0, return_counts=True)

# 输出结果
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")

# Convert rotated mask to single-channel if necessary
rotated_mask_single_channel = rotated_mask[:, :, 0]  # Use the first channel directly

# Calculate the new bounding box from the rotated mask
new_bbox = generate_bbox_from_mask(np.array(rotated_mask_single_channel), foreground_value=1)
print("new_bbox: ", new_bbox)
if new_bbox:
    # Rotate the bounding box coordinates
    # Draw the new bounding box on the rotated image
    cv2.rectangle(rotated_image, (new_bbox[0], new_bbox[1]), (new_bbox[2], new_bbox[3]), (0, 255, 255), 2)

# Save the rotated image with the new bounding box
rotated_output_path = 'rotated_image_with_bbox.png'
cv2.imwrite(rotated_output_path, rotated_image)
print(f"旋转后的图像与边界框已保存到: {rotated_output_path}")

