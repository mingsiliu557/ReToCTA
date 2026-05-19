from __future__ import annotations

from typing import Any, List
from collections.abc import Hashable, Mapping
from monai.config import NdarrayOrTensor
from monai.transforms import RandomizableTransform, MapTransform
from monai.utils.type_conversion import convert_to_tensor
from monai.data.meta_obj import get_track_meta
import random, torch
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import cv2, time
from torch.utils.data import DataLoader

# from dataloaders.VESSEL_dataloader import VESSEL_inpainted_dataset

from torch.utils import data

from skimage.morphology import skeletonize
from pathlib import Path
import os

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

        return image, mask, inpainted


def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel

def getRandomCoordinates(labelmap: NdarrayOrTensor, num: int,
        size: List[int]) -> List[tuple]:
    
    if isinstance(labelmap, torch.Tensor):
        labelmap = labelmap.cpu().numpy()  # 先移到 CPU，然后转换为 numpy 数组

    allcoors = np.where(labelmap)
    if len(allcoors[0]) == 0:
        return []

    min_size = np.array(size)/2
    max_size = np.array(labelmap.shape) - min_size - 1

    coors = []
    while len(coors) < num:
        idx = random.randint(0, len(allcoors[0])-1)
        arr_coor = np.array([allcoors[i][idx] for i in range(len(allcoors))])

        if (arr_coor >= min_size).all() and (arr_coor <= max_size).all():
            coors.append( tuple(arr_coor) )

    return coors




def getNumberOfHoles(holes):
    if isinstance(holes, int):
        return holes
    return random.randint(min(holes), max(holes))

def getSize(size, dims):
    if isinstance(size, list):
        if len(size) == dims:
            return size
        msg = f"size specified to be {size} but the data has {dims} dimensions."
        raise ValueError(msg)
    else:
        return [size for _ in range(dims)]

def coor2slices(coors, size: List[int]):
    slices = []
    isodd = [s%2 for s in size]

    for coor in coors:
        sl = [slice(max(c-s//2, 0), c+s//2+o) for c,s,o in zip(coor, size, isodd)]
        # Adding the channel dimension
        slices.append(tuple([slice(0, None)] + sl))
    return slices

class CoLeTra(RandomizableTransform):
    def __init__(self, prob: float, mix_ratio: List, ws: int) -> None:
        RandomizableTransform.__init__(self, prob)
        self.mix_ratio = mix_ratio
        self.ws = ws

    def __call__(self, img: NdarrayOrTensor, inpainted: NdarrayOrTensor,
            slices: List) -> NdarrayOrTensor:
        device = img.device
        if self.mix_ratio[0] == -1 or self.mix_ratio[1] == -1:
            mx1 = np.random.random()
            mx2 = 1-mx1
        elif self.mix_ratio[0] == -2 or self.mix_ratio[1] == -2:
            mx2 = torch.tensor(gkern(l=self.ws, sig=3), device=device)
            mx1 = 1 - mx2
        else:
            mx1 = self.mix_ratio[0]
            mx2 = self.mix_ratio[1]


        for sl in slices:
            img[sl] = img[sl]*mx1 + inpainted[sl]*mx2

        return img
        #from IPython import embed; embed(); asd

class MyNormalizeIntensityd(MapTransform):
    # It is important to normalize the inpainted image in the same way as
    # the original image
    def __init__(self, keys: List[str], ref: str|None=None) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.ref = ref

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        if self.ref:
            mean = d[self.ref].mean()
            std = d[self.ref].std()

        for key in self.key_iterator(d):
            if self.ref:
                d[key] = (d[key]-mean)/(std)
            else:
                d[key] = (d[key]-d[key].mean())/(d[key].std())
        return d


class CoLeTraTransformd(RandomizableTransform, MapTransform):

    def __init__(self, key_images: List[str], key_label: str,
            key_label_class: int,
            holes: List[int] | int, size: List[int],
            fill: str="", fill_type: str="",
            prob: float=1) -> None:

        MapTransform.__init__(self, key_images)
        RandomizableTransform.__init__(self, prob=prob)

        if not isinstance(key_label, str):
            raise ValueError("`key_label` must be a string")
        if not isinstance(key_label_class, int):
            raise ValueError("`key_label_class` must be an int")
        if isinstance(holes, list):
            if len(holes) != 2:
                raise ValueError("`holes` must be of size 2 (min,max)")
            for holep in holes:
                if not isinstance(holep, int):
                    raise ValueError("`holes` must be an int or a list of ints")
        elif not isinstance(holes, int):
            raise ValueError("`holes` must be an int or a list of ints")
        if isinstance(size, list):
            if not len(size) in [2, 3]:
                msg = (f"`size` must have two or three elements, i.e.,"
                        "one per dimension")
                raise ValueError(msg)
            for i, el in enumerate(size):
                if not isinstance(el, int):
                    msg = f"The element {i} in `size` is not an int"
                    raise ValueError(msg)
        else:
            if not isinstance(size, int):
                raise ValueError("`size` must be an int or a list of ints")
        if not isinstance(fill, str):
            raise ValueError("`fill` must be a string")
        if not isinstance(fill_type, (str, list)):
            raise ValueError("`fill_type` must be a string or a list")

        err_msg = (f"`fill_type` must be a list "
                    "of floats or ints that sum up to 1. "
                    "Given `fill_type`={fill_type}")
        if not isinstance(fill_type, list) or len(fill_type) != 2:
            raise ValueError(err_msg)
        if not isinstance(fill_type[0], (int, float)) or not isinstance(fill_type[1], (int, float)):
            raise ValueError(err_msg)

        self.key_label = key_label
        self.key_label_class = key_label_class
        self.holes = holes
        self.size = size
        self.fill = fill
        self.fill_type = fill_type

        self.transform = CoLeTra(prob=1.0, mix_ratio=fill_type, ws=size[0])


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # size: CHW(d). At this point, C=1 because it's not onehot encoded
        labelmap = d[self.key_label] == self.key_label_class
        num_holes = getNumberOfHoles(self.holes)
        size = getSize(size=self.size, dims=len(d[self.key_label].shape[1:]))

        # Center coordinates where the boxes will be located
        coors = getRandomCoordinates(labelmap[0], num_holes, size)
        # Slices (with start:end) of those boxes
        slices = coor2slices(coors, size)


        for key in self.key_iterator(d):

            d[key] = self.transform(
                    convert_to_tensor(d[key], track_meta=get_track_meta()), # Image
                    d['inpainted'], slices)

        return d

# 假设你已从 transforms 文件 import 了：
# getNumberOfHoles, getSize, getRandomCoordinates, coor2slices, CoLeTra

def apply_colettra_on_batch(x, y, inpainted, holes=30, size=(15,15), fill_type=(-2,-2), prob=1.0, cls=1, device=None):
    """
    x: [B,C,H,W] float
    y: [B,1,H,W] float/bool (0/1)
    inpainted: [B,C,H,W] float
    return: x_prime [B,C,H,W]
    """
    B = x.shape[0]
    out = x.clone()
    colettra = CoLeTra(prob=prob, mix_ratio=list[Any](fill_type), ws=size[0])

    for b in range(B):
        # labelmap: [1,H,W] bool
        labelmap = (y[b:b+1] == cls)
        num_holes = getNumberOfHoles(holes)
        ss = getSize(list(size), dims=2)  # 2D

        coors = getRandomCoordinates(labelmap[0, 0], num_holes, ss)
        slices = coor2slices(coors, ss)   # slice on [C,H,W]

        out[b] = colettra(out[b], inpainted[b], slices)

    return out

def main():
    dataset = VESSEL_inpainted_dataset(root="/home/liumingsi/VPTTA/data", img_list=["CHASEDB1/image/Image_01L.png", "CHASEDB1/image/Image_01L.png"], label_list=["CHASEDB1/mask1/Image_01L_1stHO.png", "CHASEDB1/mask1/Image_01L_1stHO.png"])
    target_test_loader = DataLoader(dataset=dataset,
                                             batch_size=2,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                            #  collate_fn=collate_fn_wo_transform,
                                             num_workers=0)
    for data in target_test_loader:
        x, y, inpainted = data
        print(x.shape, y.shape, inpainted.shape)
        x = x.to(device="cuda")
        y = y.to(device="cuda")
        inpainted = inpainted.to(device="cuda")
        output = apply_colettra_on_batch(x, y, inpainted, holes=30, size=(15,15), fill_type=(-2,-2), prob=1.0, cls=1, device=x.device)
    
    print(output.shape)



if __name__ == "__main__":
    main()