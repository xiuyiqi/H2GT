import bisect
import math
from typing import List
import torch

train_set = 0.7
percentage = 1
batch_size = 64
train_length = int(1009*train_set*percentage)
steps_per_epoch = (train_length - 1) // batch_size + 1 # 每个训练epoch的步数
total_steps = steps_per_epoch * 50 # 总训练步数
base_lr = 1.0e-4 # 基础学习率
target_lr = 1.0e-5 # 目标学习率
lr_mode = "cos" # 调度模式有效
warm_up_from = 0.0
warm_up_steps = 0
milestones = [0.3, 0.6, 0.9]
decay_factor = 0.1
poly_power = 2.0
lr_decay = 0.98

# 学习率调度
class LRScheduler:
    def __init__(self):
        self.lr = 1.0e-4 # 初始学习率

    def update(self, optimizer: torch.optim.Optimizer, step: int): # 根据当前步数来更新学习率
        assert 0 <= step <= total_steps # warm-up阶段 线性增加
        if step < warm_up_steps:
            current_ratio = step / warm_up_steps
            self.lr = warm_up_from + (base_lr - warm_up_from) * current_ratio
        else:
            current_ratio = (step - warm_up_steps) / \
                (total_steps - warm_up_steps)
            if lr_mode == 'step': # 每当步数到达一个指定的milestone时，学习率按decay_factor进行衰减
                count = bisect.bisect_left(milestones, current_ratio)
                self.lr = base_lr * pow(decay_factor, count)
            elif lr_mode == 'poly': # 多项式衰减，随着训练进度逐渐减小学习率
                poly = pow(1 - current_ratio, poly_power)
                self.lr = target_lr + (base_lr - target_lr) * poly
            elif lr_mode == 'cos': # 采用余弦退火方式衰减学习率
                cosine = math.cos(math.pi * current_ratio)
                self.lr = target_lr + (base_lr - target_lr) * (1 + cosine) / 2
            elif lr_mode == 'linear': # 线性衰减，学习率从base_lr线性下降到target_lr
                self.lr = target_lr + \
                    (base_lr - target_lr) * (1 - current_ratio)
            elif lr_mode == 'decay': # 基于epoch数进行衰减
                epoch = step // steps_per_epoch
                self.lr = base_lr * lr_decay ** epoch
        # 更新优化器使用的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr


def lr_scheduler_factory() -> List[LRScheduler]:
    return [LRScheduler()]
