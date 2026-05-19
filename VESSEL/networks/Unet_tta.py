import torch
import torch.nn as nn
from networks.adapter import Cotta_Adapter
# from networks.adapter import BufferLayer
from utils.convert import *

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self, enable_adapter: list = [True, True, True, True], convert=True, newBN=AdaBN, warm_n=5):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        """ Adapter """
        self.adapter1 = Cotta_Adapter(num_experts=4, dim=64, k=4, drop="our")
        self.adapter2 = Cotta_Adapter(num_experts=4, dim=128, k=4, drop="our")
        self.adapter3 = Cotta_Adapter(num_experts=4, dim=256, k=4, drop="our")
        self.adapter4 = Cotta_Adapter(num_experts=4, dim=512, k=4, drop="our")
        # self.adapter1 = BufferLayer(channels=64)
        # self.adapter2 = BufferLayer(channels=128)
        # self.adapter3 = BufferLayer(channels=256)
        # self.adapter4 = BufferLayer(channels=512)

        """ Adapter Gate """
        self.enable_adapter = enable_adapter

        """ Convert BN layer """
        self.newBN = newBN
        if convert:
            # Convert encoder BN layers (e1, e2, e3, e4, b)
            self = convert_encoder_to_target(self, newBN, start=0, end=5, verbose=False, input_size=512, warm_n=warm_n)
            # Convert decoder BN layers (d1, d2, d3, d4, outputs)
            self.d1, self.d2, self.d3, self.d4, self.outputs = convert_decoder_to_target(
                [self.d1, self.d2, self.d3, self.d4, self.outputs], newBN, start=0, end=5, verbose=False, warm_n=warm_n)


    def change_BN_status(self, new_sample=True):
        for nm, m in self.named_modules():
            if isinstance(m, self.newBN):
                m.new_sample = new_sample

    def reset_sample_num(self):
        for nm, m in self.named_modules():
            if isinstance(m, self.newBN):
                m.new_sample = 0


    def forward(self, inputs):
        # inputs: (N, 3, H, W)

        # ----- Encoder -----
        s1, p1 = self.e1(inputs)
        # s1: (N, 64,  H,    W) - skip connection，保持不变
        # p1: (N, 64,  H/2,  W/2) - 主特征，应用adapter
        
        # Apply adapter to p1: convert (b, c, h, w) -> (b, n, d) -> adapter -> (b, n, d) -> (b, c, h, w)
        if self.enable_adapter[0]:
            b, c, h, w = p1.shape
            p1_flat = p1.flatten(2).transpose(1, 2)  # (N, 64, H/2, W/2) -> (N, H/2*W/2, 64)
            p1_adapter = self.adapter1(p1_flat, (h, w))  # (N, H/2*W/2, 64) -> (N, H/2*W/2, 64)
            p1_adapter = p1_adapter.transpose(1, 2).view(b, c, h, w)  # (N, H/2*W/2, 64) -> (N, 64, H/2, W/2)
            p1 = p1 + p1_adapter  # 残差连接：原始p1 + adapter输出
            # p1 = self.adapter1(p1)

        s2, p2 = self.e2(p1)
        # s2: (N, 128, H/2,  W/2)
        # p2: (N, 128, H/4,  W/4)

        if self.enable_adapter[1]:
            b, c, h, w = p2.shape
            p2_flat = p2.flatten(2).transpose(1, 2)  # (N, 128, H/4, W/4) -> (N, H/4*W/4, 128)
            p2_adapter = self.adapter2(p2_flat, (h, w))  # (N, H/4*W/4, 128) -> (N, H/4*W/4, 128)
            p2_adapter = p2_adapter.transpose(1, 2).view(b, c, h, w)  # (N, H/4*W/4, 128) -> (N, 128, H/4, W/4)
            p2 = p2 + p2_adapter  # 残差连接：原始p2 + adapter输出
            # p2 = self.adapter2(p2)

        s3, p3 = self.e3(p2)
        # s3: (N, 256, H/4,  W/4)
        # p3: (N, 256, H/8,  W/8)

        if self.enable_adapter[2]:
            b, c, h, w = p3.shape
            p3_flat = p3.flatten(2).transpose(1, 2)  # (N, 256, H/8, W/8) -> (N, H/8*W/8, 256)
            p3_adapter = self.adapter3(p3_flat, (h, w))  # (N, H/8*W/8, 256) -> (N, H/8*W/8, 256)
            p3_adapter = p3_adapter.transpose(1, 2).view(b, c, h, w)  # (N, H/8*W/8, 256) -> (N, 256, H/8, W/8)
            p3 = p3 + p3_adapter  # 残差连接：原始p3 + adapter输出
            # p3 = self.adapter3(p3)

        s4, p4 = self.e4(p3)
        # s4: (N, 512, H/8,  W/8)
        # p4: (N, 512, H/16, W/16)

        if self.enable_adapter[3]:
            b, c, h, w = p4.shape
            p4_flat = p4.flatten(2).transpose(1, 2)  # (N, 512, H/16, W/16) -> (N, H/16*W/16, 512)
            p4_adapter = self.adapter4(p4_flat, (h, w))  # (N, H/16*W/16, 512) -> (N, H/16*W/16, 512)
            p4_adapter = p4_adapter.transpose(1, 2).view(b, c, h, w)  # (N, H/16*W/16, 512) -> (N, 512, H/16, W/16)
            p4 = p4 + p4_adapter  # 残差连接：原始p4 + adapter输出
            # p4 = self.adapter4(p4)

        # ----- Bottleneck -----
        b = self.b(p4)
        # b:  (N, 1024, H/16, W/16)

        # ----- Decoder -----
        d1 = self.d1(b, s4)
        # up(b): (N, 512,  H/8,  W/8)
        # cat with s4 -> (N, 1024, H/8,  W/8)
        # d1: (N, 512,  H/8,  W/8)

        d2 = self.d2(d1, s3)
        # up(d1): (N, 256,  H/4,  W/4)
        # cat with s3 -> (N, 512,  H/4,  W/4)
        # d2: (N, 256,  H/4,  W/4)

        d3 = self.d3(d2, s2)
        # up(d2): (N, 128,  H/2,  W/2)
        # cat with s2 -> (N, 256,  H/2,  W/2)
        # d3: (N, 128,  H/2,  W/2)

        d4 = self.d4(d3, s1)
        # up(d3): (N, 64,   H,    W)
        # cat with s1 -> (N, 128,  H,    W)
        # d4: (N, 64,   H,    W)

        outputs = self.outputs(d4)
        # outputs: (N, 1, H, W)

        return outputs, [s1,s4]


if __name__ == "__main__":
    x = torch.randn((2, 3, 384, 384))
    f = build_unet()
    y = f(x)
    print(y.shape)