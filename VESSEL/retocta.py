import os
import torch
import numpy as np
import argparse, sys, datetime
from copy import deepcopy  
from config import Logger
from utils.metrics import calculate_metrics
from utils.model_complexity import print_model_spatiotemporal_complexity
from networks.Unet_tta import build_unet
# from networks.Unet import build_unet
from torch.utils.data import DataLoader
from dataloaders.VESSEL_dataloader import VESSEL_dataset, VESSEL_inpainted_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from utils.augmentation import ScaleFlipAug
from copy import deepcopy
import torch.nn as nn
import random
from utils.topoloss import getTopoLoss, calculate_topo_loss
from utils.spatio_topoloss import PDMatchingLoss
from networks.adapter import Cotta_Adapter
import ast
from utils.convert import AdaBN
from utils.disconnect import apply_colettra_on_batch
import cv2
from PIL import Image
# def configure_model(model: nn.Module, bn_only: bool = True, adapter_only: bool = False):

#     model.train()
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             if bn_only:
#                 m.requires_grad_(True)
#             else:
#                 m.requires_grad_(False)
#             # 使用当前batch统计，禁用累计统计
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#         elif isinstance(m, Cotta_Adapter) & adapter_only:
#             m.requires_grad_(adapter_only)
#         else:
#             m.requires_grad_(not bn_only)
#     return model

class SimpleOpt:
    def __init__(self, precal_PD=False):
        self.precal_PD = precal_PD

def configure_model(model: nn.Module, bn_only: bool = False, adapter_only: bool = True):
    # 1. 先冻结所有参数
    model.requires_grad_(False)
    
    # 2. 强制进入 train 模式，确保 BN 统计量或 Adapter 的 Dropout/Norm 生效
    model.train()

    for m in model.modules():
        # 处理 BN 层
        if isinstance(m, nn.BatchNorm2d) & bn_only:
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.requires_grad_(True)
        
        elif isinstance(m, Cotta_Adapter):
            if adapter_only:
                m.requires_grad_(True)
                
    return model

def collect_params(model: nn.Module, bn_only: bool = True, adapter_only: bool = False):

    params = []
    if bn_only:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for name, p in m.named_parameters(recurse=False):
                    if name in ('weight', 'bias') and p.requires_grad:
                        params.append(p)
    elif adapter_only:
        for m in model.modules():
            if isinstance(m, Cotta_Adapter):
                for name, p in m.named_parameters(recurse=False):
                    if p.requires_grad:
                        params.append(p)
    else:
        for p in model.parameters():
            if p.requires_grad:
                params.append(p)
                
    return params

@torch.no_grad()
def stochastic_restore(model: nn.Module, anchor_state: dict, rst: float):

    if rst <= 0:
        return
    for mod_name, m in model.named_modules():
        for pname, p in m.named_parameters(recurse=False):
            if not p.requires_grad or pname not in ('weight', 'bias'):
                continue
            mask = (torch.rand_like(p) < rst).to(p.dtype)
            key = f"{mod_name}.{pname}"
            if key in anchor_state:
                anchor = anchor_state[key].to(p.device, dtype=p.dtype)
                p.data = anchor * mask + p.data * (1.0 - mask)

torch.set_num_threads(1)


   



@torch.no_grad()
def ema_update(ema_model, student_model, alpha: float = 0.999):
    for ep, p in zip(ema_model.parameters(), student_model.parameters()):
        ep.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)

def set_seed(seed: int = 42):

    # os.environ["PYTHONHASHSEED"] = str(seed)
    # if deterministic:
    #     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    base = seed
    random.seed(base)
    np.random.seed(base)
    torch.manual_seed(base)
    torch.cuda.manual_seed(base)
    torch.cuda.manual_seed_all(base)

class RetoCTA:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, 'my_method')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)
        
        # 设置预测结果保存目录
        self.predict_dir = os.path.join(log_root, f'predict_{time_now}')
        os.makedirs(self.predict_dir, exist_ok=True)

        # Data Loading
        target_test_csv = []
        for target in config.Target_Dataset:
            target_test_csv.append(target + '.csv')
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)
        target_test_dataset = VESSEL_inpainted_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                            config.image_size, augmentation=False)
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                            #  collate_fn=collate_fn_wo_transform,
                                             num_workers=config.num_workers)
        self.image_size = config.image_size

        # Model
        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)

        # GPU
        self.device = config.device
        self.iters = config.iters

        # TTA
        self.bn_only = False
        self.adapter_only = True
        self.enable_adapter = [True, True, True, True]
        # self.connect_loss = Connect_Loss().to(self.device)
        # pd_opt = SimpleOpt(precal_PD=False)
        # self.pd_loss = PDMatchingLoss(pd_opt, p=2).to(self.device)

        # ===== 新增：EMA 衰减系数（可加到 argparse；见文末） =====
        self.ema_alpha = getattr(config, "ema_alpha", 0.999)
        self.rst_ratio = float(getattr(config, "rst_ratio", 0.01))

        self.warm_n = 5
        self.save = True

        # Initialize the pre-trained model and optimizer
        self.build_model()

        if getattr(config, "print_model_complexity", False):
            print_model_spatiotemporal_complexity(
                self.model,
                self.device,
                self.image_size,
                in_channels=self.in_ch,
            )

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        print('***' * 20)

    def build_model(self):
        self.model = build_unet(self.enable_adapter, convert=False).to(self.device)
        # self.model = build_unet().to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'best_Unet.pth'),
                                map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)

        self.model = configure_model(self.model, bn_only=self.bn_only, adapter_only=self.adapter_only)
        if self.bn_only | self.adapter_only:
            trainable = collect_params(self.model, bn_only=self.bn_only, adapter_only=self.adapter_only)
            # print(f"trainable: {trainable}")
        else:
            self.model.train()
            self.model.requires_grad_(True)
        
        trainable = [p for p in self.model.parameters() if p.requires_grad]
            # print(f"trainable: {trainable}")
        # print(f"trainable: {trainable}")
        if self.optim == 'SGD':
            self.base_optimizer = torch.optim.SGD(
                trainable, lr=self.lr, momentum=self.momentum,
                nesterov=True, weight_decay=self.weight_decay
            )
        else:
            self.base_optimizer = torch.optim.Adam(
                trainable, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
            )

        self.ema_model = deepcopy(self.model).to(self.device).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.anchor_state = deepcopy(self.model.state_dict())

    def cal_consis_loss(self, data, criterion=torch.nn.BCEWithLogitsLoss(), topo_criterion=calculate_topo_loss, lambda_topo = 0.002):
        self.model.train()
        # self.model.change_BN_status(new_sample=True)
        x_batch, y_batch, inpainted = data[0], data[1], data[2]
        x = x_batch.to(self.device)
        y = y_batch.to(self.device)
        inpainted = inpainted.to(self.device)
        y_pesudo = torch.sigmoid(self.ema_model(x)).detach()  # [B, C, H, W]
        x_disconnected = apply_colettra_on_batch(
            x, y_pesudo, inpainted,
            holes=50,
            size=(15,15),
            fill_type=(-2,-2),   
            prob=1.0,
            cls=1,
            device=self.device
        )

 
 
        weak_transform = ScaleFlipAug(scales=(0.5, 1.0, 1.25, 1.5))

        # ================= Teacher: Strong Augmentation =================
        predictions = []
        with torch.no_grad():
            for i in range(4):  # 对应 4 个 scale
                x_weak, info = weak_transform(x, i)          # 增强后图像
                pred_weak, _ = self.ema_model(x_weak)        # [B, C, H, W] logits, features

                # 只 inverse flip，让预测回到原图坐标
                pred_inv = weak_transform.inverse(pred_weak, info)
                predictions.append(pred_inv)

        teacher_logits = torch.stack(predictions, dim=0).mean(0)
        teacher_prob = torch.stack(predictions, dim=0).sigmoid().mean(0)  # pseudo_soft
        teacher_logits = teacher_logits.to(self.device)
        teacher_prob = teacher_prob.to(self.device)

        # ================= Student: Weak Augmentation =================
        idx = random.randint(0, 3)
        x_student, info_student = weak_transform(x_disconnected, idx)  # x' = A(x)

        student_logits, _ = self.model(x_student)              # [B, C, H, W] logits
        student_prob = torch.sigmoid(student_logits)        # [B,1,H,W]
        # 把 pseudo-label 也做同样增强：P_teacher' = A(P_teacher)
        teacher_logits = weak_transform.apply_on_target(teacher_logits, info_student)  # [B, C, H, W], prob
        teacher_prob = weak_transform.apply_on_target(teacher_prob, info_student)  # [B, C, H, W], prob
        # ================= 一致性损失 =================
        self.base_optimizer.zero_grad()
        consis_loss = criterion(student_prob, teacher_prob)


        topo_loss = topo_criterion(student_prob, teacher_prob)
        total_loss =  consis_loss + lambda_topo * topo_loss
        # total_loss = total_loss + consis_loss
        total_loss.backward()
        self.base_optimizer.step() 

        ema_update(self.ema_model, self.model, alpha=self.ema_alpha)
        if self.bn_only:
            stochastic_restore(self.model, anchor_state=self.anchor_state, rst=self.rst_ratio)
        # self.model.change_BN_status(new_sample=False)

        return total_loss.item()

    def run(self):
        metric_dict = ['Dice', 'clDice', 'Beta Error']
        metrics_test = [[] for _ in metric_dict]

        for batch, data in enumerate(self.target_test_loader):
            # ====== 一步一致性自训练（Teacher–Student）======
            # self.model.train()
            for i in range(self.iters):
                consis_loss = self.cal_consis_loss(data)
                # print(f"Consis Loss: {consis_loss}, Topo Loss: {topo_loss}")

            # ====== 评估（student 在干净图上）======
            x, y = data[0], data[1]
            # 从数据加载器提取文件名：data[2] 是图像文件路径列表（如果有）
            if len(data) > 2:
                img_paths = data[3] if isinstance(data[3], (list, tuple)) else [data[3]]
                name_list = [os.path.basename(str(path)) for path in img_paths]
            else:
                # 如果没有文件名，使用批次编号
                name_list = [f"batch_{batch}_img_{i}.png" for i in range(x.shape[0])]

            x = x.to(self.device)
            y = y.to(self.device)

            self.model.eval()
            with torch.no_grad():
                pred_logit, _ = self.model(x)

            seg_output = torch.sigmoid(pred_logit)  # 你的评估逻辑不改
            
            
            metrics = calculate_metrics(
                seg_output.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                include_topology=True
            )
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {metric_dict[i]: test_metrics_y[i] for i in range(len(test_metrics_y))}
        print("Test Metrics: ", print_test_metric_mean)
        for metric_name in metric_dict:
            print(f'Mean {metric_name}:', print_test_metric_mean[metric_name])


if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='RIM_ONE_r3',
                        help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
    parser.add_argument('--Target_Dataset', type=str)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=384)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=1)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--iters', type=int, default=4)


    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs')
    parser.add_argument('--model_root', type=str, default='./models')
    parser.add_argument('--dataset_root', type=str, default='/home/liumingsi/VPTTA/data')

    # Cuda
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--ema_alpha', type=float, default=0.999)
    parser.add_argument(
        '--print_model_complexity',
        action='store_true',
        help='After build, print FLOPs (G), trainable params (M), s/image, peak GPU memory (MB)',
    )

    config = parser.parse_args()

    # config.Target_Dataset = ['CHASEDB1', 'DRIVE', 'STARE']
    config.Target_Dataset = ast.literal_eval(config.Target_Dataset)
    config.Target_Dataset.remove(config.Source_Dataset)

    RetoCTA = RetoCTA(config)
    RetoCTA.run()
