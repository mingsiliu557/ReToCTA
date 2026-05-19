import torch.nn as nn
import numpy as np


class AdaBN(nn.BatchNorm2d):
    def __init__(self, in_ch, warm_n=5):
        super(AdaBN, self).__init__(in_ch)
        self.warm_n = warm_n
        self.sample_num = 0
        self.new_sample = False

    def get_mu_var(self, x):
        if self.new_sample:
            self.sample_num += 1
        C = x.shape[1]

        cur_mu = x.mean((0, 2, 3), keepdims=True).detach()
        cur_var = x.var((0, 2, 3), keepdims=True).detach()

        src_mu = self.running_mean.view(1, C, 1, 1)
        src_var = self.running_var.view(1, C, 1, 1)

        moment = 1 / ((np.sqrt(self.sample_num) / self.warm_n) + 1)

        new_mu = moment * cur_mu + (1 - moment) * src_mu
        new_var = moment * cur_var + (1 - moment) * src_var
        return new_mu, new_var

    def forward(self, x):
        N, C, H, W = x.shape

        new_mu, new_var = self.get_mu_var(x)

        cur_mu = x.mean((2, 3), keepdims=True)
        cur_std = x.std((2, 3), keepdims=True)
        self.bn_loss = (
                (new_mu - cur_mu).abs().mean() + (new_var.sqrt() - cur_std).abs().mean()
        )

        # Normalization with new statistics
        new_sig = (new_var + self.eps).sqrt()
        new_x = ((x - new_mu) / new_sig) * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return new_x


def convert_resnet_encoder_to_target(net, norm, start=0, end=5, verbose=True, bottleneck=False, input_size=512, warm_n=5):
    def convert_norm(old_norm, new_norm, num_features, idx, fea_size):
        norm_layer = new_norm(num_features, warm_n).to(net.conv1.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = [0, net.layer1, net.layer2, net.layer3, net.layer4]

    idx = 0
    for i, layer in enumerate(layers):
        if not (start <= i < end):
            continue
        if i == 0:
            net.bn1 = convert_norm(net.bn1, norm, net.bn1.num_features, idx, fea_size=input_size // 2)
            idx += 1
        else:
            down_sample = 2 ** (1 + i)

            for j, block in enumerate(layer):
                block.bn1 = convert_norm(block.bn1, norm, block.bn1.num_features, idx, fea_size=input_size // down_sample)
                idx += 1
                block.bn2 = convert_norm(block.bn2, norm, block.bn2.num_features, idx, fea_size=input_size // down_sample)
                idx += 1
                if bottleneck:
                    block.bn3 = convert_norm(block.bn3, norm, block.bn3.num_features, idx, fea_size=input_size // down_sample)
                    idx += 1
                if block.downsample is not None:
                    block.downsample[1] = convert_norm(block.downsample[1], norm, block.downsample[1].num_features, idx, fea_size=input_size // down_sample)
                    idx += 1
    return net


def convert_encoder_to_target(net, norm, start=0, end=5, verbose=True, input_size=512, warm_n=5):
    """
    Convert BN layers in U-Net encoder to target norm (e.g., AdaBN)
    
    Args:
        net: build_unet instance with e1, e2, e3, e4 (encoder blocks) and b (bottleneck)
        norm: target normalization class (e.g., AdaBN)
        start: start index (0=e1, 1=e2, 2=e3, 3=e4, 4=b)
        end: end index (exclusive)
        verbose: whether to print conversion info
        input_size: input image size (H or W)
        warm_n: warm_n parameter for AdaBN
    """
    def convert_norm(old_norm, new_norm, num_features, idx, fea_size):
        norm_layer = new_norm(num_features, warm_n).to(old_norm.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    # Encoder blocks: e1, e2, e3, e4, and bottleneck b
    encoder_layers = [
        ('e1', net.e1, 1),   # e1: H×W resolution
        ('e2', net.e2, 2),   # e2: H/2×W/2 resolution
        ('e3', net.e3, 4),   # e3: H/4×W/4 resolution
        ('e4', net.e4, 8),   # e4: H/8×W/8 resolution
        ('b', net.b, 16),    # b: H/16×W/16 resolution
    ]

    idx = 0
    for i, (name, layer, down_sample) in enumerate(encoder_layers):
        if not (start <= i < end):
            continue
        
        fea_size = input_size // down_sample
        
        if i == 4:  # Bottleneck: conv_block with bn1 and bn2
            # Convert bn1
            layer.bn1 = convert_norm(layer.bn1, norm, layer.bn1.num_features, idx, fea_size)
            idx += 1
            # Convert bn2
            layer.bn2 = convert_norm(layer.bn2, norm, layer.bn2.num_features, idx, fea_size)
            idx += 1
        else:  # Encoder blocks: encoder_block.conv (conv_block) with bn1 and bn2
            # Convert bn1
            layer.conv.bn1 = convert_norm(layer.conv.bn1, norm, layer.conv.bn1.num_features, idx, fea_size)
            idx += 1
            # Convert bn2
            layer.conv.bn2 = convert_norm(layer.conv.bn2, norm, layer.conv.bn2.num_features, idx, fea_size)
            idx += 1
    
    return net


def convert_decoder_to_target(net, norm, start=0, end=5, verbose=True, input_size=512, warm_n=5):
    """
    Convert BN layers in U-Net decoder to target norm (e.g., AdaBN)
    
    Args:
        net: list of decoder blocks [d1, d2, d3, d4, outputs]
            - d1, d2, d3, d4 are decoder_block (with conv_block containing bn1 and bn2)
            - outputs is nn.Conv2d (no BN layer, skip if i == 4)
        norm: target normalization class (e.g., AdaBN)
        start: start index (0=d1, 1=d2, 2=d3, 3=d4, 4=outputs)
        end: end index (exclusive)
        verbose: whether to print conversion info
        input_size: input image size (H or W)
        warm_n: warm_n parameter for AdaBN
    """
    def convert_norm(old_norm, new_norm, num_features, idx, fea_size):
        norm_layer = new_norm(num_features, warm_n).to(old_norm.weight.device)
        if hasattr(norm_layer, 'load_old_dict'):
            info = 'Converted to : {}'.format(norm)
            norm_layer.load_old_dict(old_norm)
        elif hasattr(norm_layer, 'load_state_dict'):
            state_dict = old_norm.state_dict()
            info = norm_layer.load_state_dict(state_dict, strict=False)
        else:
            info = 'No load_old_dict() found!!!'
        if verbose:
            print(info)
        return norm_layer

    layers = [net[0], net[1], net[2], net[3], net[4]]

    idx = 0
    for i, layer in enumerate(layers):
        if not (start <= i < end):
            continue
        if i == 4:
            # outputs is nn.Conv2d, no BN layer to convert, skip
            continue
        else:
            # decoder_block: convert bn1 and bn2 in layer.conv (conv_block)
            down_sample = 2 ** (4 - i)
            fea_size = input_size // down_sample
            # Convert bn1
            layer.conv.bn1 = convert_norm(layer.conv.bn1, norm, layer.conv.bn1.num_features, idx, fea_size)
            idx += 1
            # Convert bn2
            layer.conv.bn2 = convert_norm(layer.conv.bn2, norm, layer.conv.bn2.num_features, idx, fea_size)
            idx += 1
    return net

