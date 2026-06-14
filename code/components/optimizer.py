import logging
from collections import defaultdict
from typing import List
import torch

# 获取不使用权重衰减的参数
def get_param_group_no_wd(model: torch.nn.Module, match_rule: str = None, except_rule: str = None):
    param_group_no_wd = [] # 存储不应用权重衰减的参数
    names_no_wd = [] # 存储不应用权重衰减的参数名称
    param_group_normal = [] # 存储应用权重衰减的参数

    type2num = defaultdict(lambda: 0) # 统计不同类型的参数的数量
    for name, m in model.named_modules(): # 遍历模块
        if match_rule is not None and match_rule not in name:
            continue
        if except_rule is not None and except_rule in name:
            continue
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
        elif isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
        elif isinstance(m, torch.nn.BatchNorm2d) \
                or isinstance(m, torch.nn.BatchNorm1d):
            if m.weight is not None:
                param_group_no_wd.append(m.weight)
                names_no_wd.append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight'] += 1
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1

    for name, p in model.named_parameters(): # 遍历参数
        if match_rule is not None and match_rule not in name:
            continue
        if except_rule is not None and except_rule in name:
            continue
        if name not in names_no_wd:
            param_group_normal.append(p)

    params_length = len(param_group_normal) + len(param_group_no_wd)
    logging.info(f'Parameters [no weight decay] length [{params_length}]')
    return [{'params': param_group_normal}, {'params': param_group_no_wd, 'weight_decay': 0.0}], type2num

name = 'Adam'
lr = 1.0e-4
weight_decay = 1.0e-4
no_weight_decay = False
match_rule = None
except_rule = None
momentum = 0.9
nesterov = False

# 优化器工厂
def optimizer_factory(model: torch.nn.Module) -> torch.optim.Optimizer:
    parameters = {
        'lr': lr,
        'weight_decay': weight_decay
    } # 初始化优化器参数
    # 获取参与权重衰减的参数
    if no_weight_decay:
        params, _ = get_param_group_no_wd(model,
                                          match_rule=match_rule,
                                          except_rule=except_rule)
    else:
        params = list(model.parameters())
        logging.info(f'Parameters [normal] length [{len(params)}]')

    parameters['params'] = params

    optimizer_type = name
    if optimizer_type == 'SGD':
        parameters['momentum'] = momentum
        parameters['nesterov'] = nesterov
    return getattr(torch.optim, optimizer_type)(**parameters) # 使用torch.optim动态获取并创建相应的优化器


def optimizers_factory(model: torch.nn.Module) -> List[torch.optim.Optimizer]:
    if model is None:
        return None
    return [optimizer_factory(model=model)]
