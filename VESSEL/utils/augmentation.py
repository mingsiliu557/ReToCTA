import torch
import torch.nn as nn
from .style_aug import get_strong_style_transform, get_weak_style_transform
# from .slaug import LocationScaleAugmentation
import torch.nn.functional as F
import random

class Rotate_and_Flip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, factor):
        # Flip
        if factor == 0:
            return x.flip(-1)  # horizontal
        elif factor == 1:
            return x.flip(-2)  # vertical
        # Rotate
        elif factor == 2:
            return x.flip(-1).transpose(-2, -1)  # 90
        elif factor == 3:
            return x.flip(-1).flip(-2)           # 180
        elif factor == 4:
            return x.transpose(-2, -1).flip(-1)  # 270

    def inverse(self, pred, factor):
        # Flip
        if factor == 0:
            return pred.flip(-1)
        elif factor == 1:
            return pred.flip(-2)
        # Rotate
        elif factor == 2:
            return pred.transpose(-2, -1).flip(-1)
        elif factor == 3:
            return pred.flip(-1).flip(-2)
        elif factor == 4:
            return pred.flip(-1).transpose(-2, -1)


# def augmentation_strong_style(batch):
#     style_trans = get_strong_style_transform()
#     Bezier_curve = LocationScaleAugmentation(vrange=(0., 1.), background_threshold=0.01)

#     if isinstance(batch, (tuple, list)):
#         image, mask = batch
#     elif isinstance(batch, dict):
#         image = batch.get('data', batch.get('image'))
#         mask = batch.get('seg', batch.get('mask'))
#     else:
#         raise TypeError(f"Unsupported batch type {type(batch)} for augmentation_strong_style")

#     if image is None:
#         raise ValueError("Image tensor is required for augmentation_strong_style")

#     mask = mask if mask is not None else torch.zeros_like(image[:, :1])

#     image_np = image.detach().cpu().numpy()
#     mask_np = mask.detach().cpu().numpy()

#     data_dict = {'data': image_np, 'seg': mask_np}
#     data_aug = style_trans(**data_dict)
#     img = data_aug['data']

#     # img = Bezier_curve.Global_Location_cale_Augmentation(img)
#     return img

class ScaleFlipAug(nn.Module):
    def __init__(self, scales=(0.5, 1.0, 1.25, 1.5)):
        super().__init__()
        self.scales = scales

    def _resize_and_center(self, x, scale):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        new_h = int(h * scale)
        new_w = int(w * scale)

        x_scaled = F.interpolate(
            x, size=(new_h, new_w),
            mode="bilinear", align_corners=False
        )

        if scale < 1.0:
            pad_h = h - new_h
            pad_w = w - new_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x_out = F.pad(
                x_scaled,
                (pad_left, pad_right, pad_top, pad_bottom)
            )
        else:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            x_out = x_scaled[:, :, start_h:start_h + h, start_w:start_w + w]

        return x_out

    def forward(self, x, idx):
        """
        x: [B, C, H, W]
        idx: 0~len(scales)-1
        返回:
          x_aug: 增强后的图像
          info:  (scale, hflip, vflip)
        """
        scale = self.scales[idx]
        x_aug = self._resize_and_center(x, scale)

        hflip = random.random() < 0.5
        vflip = random.random() < 0.5

        if hflip:
            x_aug = x_aug.flip(-1)
        if vflip:
            x_aug = x_aug.flip(-2)

        info = (scale, hflip, vflip)
        return x_aug, info

    def inverse(self, pred, info):
        """
        只还原 flip，使其回到原图坐标系。
        pred: [B, C, H, W]
        info: (scale, hflip, vflip)
        """
        _, hflip, vflip = info

        if vflip:
            pred = pred.flip(-2)
        if hflip:
            pred = pred.flip(-1)

        return pred

    def apply_on_target(self, y, info):
        """
        用于把 pseudo-label 做同样的几何变换：
        y    : [B, C, H, W] 或 [B, H, W]
        info : (scale, hflip, vflip)
        """
        scale, hflip, vflip = info

        need_unsqueeze = (y.dim() == 3)  # [B, H, W] -> [B, 1, H, W]
        if need_unsqueeze:
            y = y.unsqueeze(1)

        y = self._resize_and_center(y, scale)

        if hflip:
            y = y.flip(-1)
        if vflip:
            y = y.flip(-2)

        if need_unsqueeze:
            y = y.squeeze(1)

        return y
