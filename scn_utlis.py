import torch
import numpy as np
from torch import nn
import torchvision.transforms.functional as TF 

from tqdm import tqdm
import torch.nn.functional as F
import math
import os
import cv2
import numpy as np
import random

from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)


def print_trainable_modules(model: nn.Module):
    """
    Print the full name and type of each module in a PyTorch model that contains trainable parameters.
    
    Args:
        model (nn.Module): The PyTorch model to analyze.
    """
    for name, module in model.named_modules():
        if isinstance(module, (C3, C3x, C3Ghost, C3TR, SPP, SPPF)):
            print(f"Module: {name}, Type: {type(module).__name__}")
        # Check if the module has any trainable parameters
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            print(f"Module: {name}, Type: {type(module).__name__}")




def xywhn2xyxy(x, w=640, h=640):
    """Convert normalized xywh to xyxy format"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2)  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2)  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2)  # bottom right y
    return y

def xyxy2xywhn(x, w=640, h=640):
    """Convert xyxy to normalized xywh format"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w        # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h        # height
    return y

def rotate_box(bbox, angle, width, height):
    """Rotate bbox using polar coordinates"""
    width_bias = width // 2
    height_bias = height // 2
    
    # 将bbox转换为四个角点坐标
    x_coords = torch.tensor([bbox[0], bbox[2], bbox[0], bbox[2]]) - width_bias
    y_coords = torch.tensor([bbox[1], bbox[1], bbox[3], bbox[3]]) - height_bias
    
    # 转换为极坐标
    r = torch.sqrt(x_coords**2 + y_coords**2)
    theta = torch.atan2(y_coords, x_coords)
    
    # 旋转
    angle_rad = math.radians(angle)
    theta += angle_rad
    
    # 转回笛卡尔坐标
    x_cartesian = r * torch.cos(theta) + width_bias
    y_cartesian = r * torch.sin(theta) + height_bias
    
    # 裁剪到图像边界
    x_cartesian = torch.clamp(x_cartesian, 0, width)
    y_cartesian = torch.clamp(y_cartesian, 0, height)
    
    # 计算新的bbox
    x_min, _ = torch.min(x_cartesian, dim=0)
    x_max, _ = torch.max(x_cartesian, dim=0)
    y_min, _ = torch.min(y_cartesian, dim=0)
    y_max, _ = torch.max(y_cartesian, dim=0)
    
    return torch.tensor([x_min, y_min, x_max, y_max])

def plot_one_box(x, img, color, label=None, line_thickness=None):
    """Plot one bounding box"""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color.tolist(), thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color.tolist(), -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def save_comparison_images(original_imgs, rotated_imgs, original_targets, rotated_targets, angle, save_dir='tmp_img'):
    """Save comparison images with bboxes"""
    os.makedirs(save_dir, exist_ok=True)
    
    original_imgs = original_imgs.cpu().numpy()
    rotated_imgs = rotated_imgs.cpu().numpy()
    original_targets = original_targets.cpu()
    rotated_targets = rotated_targets.cpu()
    
    batch_size = original_imgs.shape[0]
    height, width = original_imgs.shape[2:]
    
    # 生成颜色
    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(100, 3))
    
    for i in range(batch_size):
        # 准备图片
        orig_img = original_imgs[i].transpose(1, 2, 0)
        rot_img = rotated_imgs[i].transpose(1, 2, 0)
        
        # 转换到uint8
        orig_img = (orig_img * 255).astype(np.uint8)
        rot_img = (rot_img * 255).astype(np.uint8)
        
        # RGB to BGR
        if orig_img.shape[2] == 3:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            rot_img = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
        
        # 绘制原始图片的bbox
        img_targets = original_targets[original_targets[:, 0] == i]
        for target in img_targets:
            cls_id = int(target[1])
            bbox = xywhn2xyxy(target[2:6].unsqueeze(0), width, height).squeeze()
            color = colors[cls_id % len(colors)]
            label = f'class {cls_id}'
            plot_one_box(bbox, orig_img, color, label)
        
        # 绘制旋转后图片的bbox
        img_targets = rotated_targets[rotated_targets[:, 0] == i]
        for target in img_targets:
            cls_id = int(target[1])
            bbox = xywhn2xyxy(target[2:6].unsqueeze(0), width, height).squeeze()
            color = colors[cls_id % len(colors)]
            label = f'class {cls_id}'
            plot_one_box(bbox, rot_img, color, label)
        
        # 拼接图片
        comparison = np.hstack((orig_img, rot_img))
        
        # 添加说明文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, f'Original', (50, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, f'Rotated {angle}', (width + 50, 30), font, 1, (255, 255, 255), 2)
        
        # 保存图片
        save_path = os.path.join(save_dir, f'comparison_{i}.jpg')
        cv2.imwrite(save_path, comparison)

def rotate_images_and_boxes(imgs, targets, angle):
    """Rotate images and bboxes"""
    # 旋转图像
    rotated_imgs = TF.rotate(imgs, -angle)
    
    # 处理bbox
    rotated_targets = targets.clone()
    if len(targets):
        height, width = imgs.shape[2:]
        for i in range(len(targets)):
            # 转换到xyxy格式
            bbox = xywhn2xyxy(targets[i, 2:6].unsqueeze(0), width, height).squeeze()
            # 旋转bbox
            rotated_bbox = rotate_box(bbox, angle, width, height)
            # 转回xywhn格式
            rotated_targets[i, 2:6] = xyxy2xywhn(rotated_bbox.unsqueeze(0), width, height).squeeze()
    
    return rotated_imgs, rotated_targets


def apply_rotation_augmentation(imgs, targets, paths, angle):
    """
    将指定角度的旋转应用到训练批次
    
    Args:
        imgs: 批量图像张量
        targets: 边界框目标张量
        paths: 图像路径
        angle: 旋转角度(度)，整个batch使用相同的角度
    """
    rotated_imgs, rotated_targets = rotate_images_and_boxes(imgs, targets, angle)
    return rotated_imgs, rotated_targets, paths
