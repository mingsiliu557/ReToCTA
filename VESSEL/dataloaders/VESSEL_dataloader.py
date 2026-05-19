import os
import math
from torch.utils import data
import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *
from dataloaders.normalize import normalize_image_to_0_1
import cv2
import torch
import random
import cv2
import numpy as np
from skimage.morphology import skeletonize
from pathlib import Path


def resize_mask_with_topology(mask, target_size):
    """
    mask: 原始灰度 mask，范围 0~255
    target_size: (H, W)
    按附录中的方法进行 resize：
        - Area resize
        - binarize@0.5 + binarize@0
        - skeleton from binarize@0
        - final = OR(skeleton, binarize@0.5)
    """

    # --- 1. Area interpolation resize ---
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)

    # --- 2. Normalized to 0~1 ---
    mask_norm = mask_resized.astype(np.float32) / 255.0

    # --- 3. Binarize ---
    mask_bin_05 = (mask_norm > 0.5).astype(np.uint8)
    mask_bin_0  = (mask_norm > 0.0).astype(np.uint8)

    # --- 4. Skeleton from mask_bin_0 ---
    # skimage 输入必须是 bool
    skeleton = skeletonize(mask_bin_0.astype(bool))
    skeleton = skeleton.astype(np.uint8)

    # --- 5. Combine (pixel-wise OR) ---
    final_mask = np.logical_or(skeleton, mask_bin_05).astype(np.uint8)

    return final_mask  # 值域为 0/1



class VESSEL_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list,
                 target_size=512,
                 augmentation=False,
                #  batch_size=None,
                #  img_normalize=True
                 ):
        # """
        # root        : 数据根目录
        # img_list    : 图像相对路径列表
        # label_list  : 标签相对路径列表
        # target_size : (H, W)，统一 resize 尺寸
        # img_normalize: 是否做 0-1 归一化
        # """
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)
        self.augmentation = augmentation
        # self.img_normalize = img_normalize

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_file = os.path.join(self.root, self.img_list[item])
        label_file = os.path.join(self.root, self.label_list[item])

        # 使用 cv2 读取图像，与标准格式一致
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        mask = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.target_size, cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, self.target_size, cv2.INTER_NEAREST)
        mask = resize_mask_with_topology(mask, self.target_size)

        if self.augmentation:
            if random.random() < 0.5:  
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() < 0.5:  
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)

        image = torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1)).astype(np.float32))
        # mask = torch.from_numpy(np.expand_dims(mask / 255.0, 0).astype(np.float32))
        mask = torch.from_numpy(np.expand_dims(mask.astype(np.float32), 0))

        return image, mask, self.img_list[item], self.label_list[item]

class VESSEL_inpainted_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list,
                 target_size=512,
                 augmentation=False,
                 inpainted_folder_name="inpainted",   # 生成的文件夹名
                 image_folder_name="image",           # 原图文件夹名
                 ):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = (target_size, target_size)
        self.augmentation = augmentation
        self.inpainted_folder_name = inpainted_folder_name
        self.image_folder_name = image_folder_name

    def __len__(self):
        return self.len

    def _get_inpainted_path(self, img_file: str) -> str:
        """
        规则：.../image/xxx.png  ->  .../Inpainted/xxx.png
        若你的文件夹名不是 image / Inpainted，可在 init 里改参数。
        """
        # 统一用 Path 更稳，也兼容 linux 路径分隔符
        img_path = Path(img_file)

        # 找到路径中名为 image_folder_name 的那一层并替换
        parts = list(img_path.parts)
        if self.image_folder_name in parts:
            idx = parts.index(self.image_folder_name)
            parts[idx] = self.inpainted_folder_name
            return str(Path(*parts))
        else:
            # 如果 img_list 里没有显式包含 /image/，就做一个兜底替换
            return img_file.replace(f"/{self.image_folder_name}/", f"/{self.inpainted_folder_name}/")

    def __getitem__(self, item):
        img_file = os.path.join(self.root, self.img_list[item])
        label_file = os.path.join(self.root, self.label_list[item])

        # ===== 新增：推导 inpainted 路径并读取 =====
        inpainted_file = self._get_inpainted_path(img_file)

        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        mask  = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        inpainted = cv2.imread(inpainted_file, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"image not found: {img_file}")
        if mask is None:
            raise FileNotFoundError(f"mask not found: {label_file}")
        if inpainted is None:
            raise FileNotFoundError(f"inpainted not found: {inpainted_file}")

        # resize（保持三者对齐）
        image = cv2.resize(image, self.target_size, cv2.INTER_LINEAR)
        inpainted = cv2.resize(inpainted, self.target_size, cv2.INTER_LINEAR)
        mask = resize_mask_with_topology(mask, self.target_size)

        # 同步增强（必须三者一致）
        if self.augmentation:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
                inpainted = cv2.flip(inpainted, 1)
            if random.random() < 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
                inpainted = cv2.flip(inpainted, 0)

        # to tensor
        image = torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1)).astype(np.float32))
        inpainted = torch.from_numpy(np.transpose(inpainted / 255.0, (2, 0, 1)).astype(np.float32))
        mask = torch.from_numpy(np.expand_dims(mask.astype(np.float32), 0))

        return image, mask, inpainted, self.img_list[item], self.label_list[item]
