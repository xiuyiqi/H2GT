from .abide import load_abide_data
from .dataloader import init_dataloader, init_stratified_dataloader
from typing import List
import torch.utils as utils

stratified = True

# 加载数据 主函数调用
def dataset_factory() -> List[utils.data.DataLoader]: # 函数的返回值是一个Dataloader对象的列表


    datasets = load_abide_data() # 得到加载好的数据集

    # 选择分层采样的数据加载器或是普通的数据加载器
    dataloaders = init_stratified_dataloader(*datasets) \
        if stratified \
        else init_dataloader(*datasets)

    return dataloaders
