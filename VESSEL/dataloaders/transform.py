import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, ContrastAugmentationTransform, FancyColorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform



def get_train_transform(patch_size=(512, 512)):
    """
    只做随机水平/垂直翻转：
    - data: (B, C, H, W)
    - mask: (B, 1, H, W)
    → 空间维 = (2, 3)
    """
    tr_transforms = []
    tr_transforms.append(MirrorTransform(axes=(0, 1)))  # H & W 方向随机翻转
    return Compose(tr_transforms)


# def collate_fn_w_transform(batch):
#     image, label, name = zip(*batch)
#     image = np.stack(image, 0)
#     label = np.stack(label, 0)
#     name = np.stack(name, 0)
#     data_dict = {'data': image, 'mask': label, 'name': name}
#     tr_transforms = get_train_transform()
#     data_dict = tr_transforms(**data_dict)
#     data_dict['mask'] = to_one_hot_list(data_dict['mask'])
#     return data_dict

def collate_fn_w_transform(batch):
    """
    batch: list of (img_npy, mask, img_file)
      - img_npy: (3, H, W), float32
      - mask:    (1, H, W), uint8 0/1
    """
    image, label, name = zip(*batch)

    # 堆成 batch
    image = np.stack(image, 0)   # (B, 3, H, W)
    label = np.stack(label, 0)   # (B, 1, H, W)
    name = np.stack(name, 0)

    data_dict = {'data': image, 'mask': label, 'name': name}

    # 只做 H/V flip
    tr_transforms = get_train_transform()
    data_dict = tr_transforms(**data_dict)

    # 转 float32，方便后续 torch.from_numpy
    data_dict['data'] = data_dict['data'].astype(np.float32)
    data_dict['mask'] = data_dict['mask'].astype(np.float32)  # 仍然是 0/1

    return data_dict


def collate_fn_wo_transform(batch):
    image, label, name = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    data_dict = {'data': image, 'mask': label, 'name': name}
    data_dict['mask'] = to_one_hot_list(data_dict['mask'])
    return data_dict


def to_one_hot_list(mask_list):
    list = []
    for i in range(mask_list.shape[0]):
        mask = to_one_hot(mask_list[i].squeeze(0))
        list.append(mask)
    return np.stack(list, 0)


def to_one_hot(pre_mask, classes=1):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [1]
    return mask.transpose(2, 0, 1)

