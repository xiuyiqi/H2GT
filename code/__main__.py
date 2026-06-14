from dataset import dataset_factory
from model import model_factory
from components import lr_scheduler_factory, optimizers_factory, logger_factory
from training import training_factory
import numpy as np
import random
import torch

batch_size = 64
nhead = 8
pooling = [False,True]
num_MHSA =1
repeat_time = 1

# 创建训练组件，执行训练过程，得到评估指标
def model_training():

    dataloaders = dataset_factory() # 数据加载器
    logger = logger_factory() # 日志记录器
    model = model_factory() # 模型
    optimizers = optimizers_factory(model=model) # 优化器
    lr_schedulers = lr_scheduler_factory() # 学习率调度器
    training = training_factory(model, optimizers, lr_schedulers, dataloaders, logger) # 整合训练组件
    t_acc,t_auc,t_sen,t_spec = training.train() # 执行训练过程，得到评估指标
    return t_acc,t_auc,t_sen,t_spec

# 通过多次实验评估模型的性能
def main():
    count = 0 # 实验计数器
    acc_list = [] # 准确率
    auc_list = [] # AUC
    sen_list = [] # 灵敏度
    spec_list = [] # 特异性
    seeds = list(range(repeat_time)) # 种子列表
    for it in range(len(seeds)):
        SEED = seeds[it]
        random.seed(
            SEED
        )  # set the random seed so that the random permutations can be reproduced again 设置随机种子，确保可复现性
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        t_acc,t_auc,t_sen,t_spec = model_training() # 完成训练测试过程
        acc_list.append(t_acc)
        auc_list.append(t_auc)
        sen_list.append(t_sen)
        spec_list.append(t_spec)
        count = count + 1
    print("test acc mean:{}  std: {}".format(np.mean(acc_list),np.std(acc_list))) # 得到指标的平均值和标准差
    print("test auc mean:{}  std: {}".format(np.mean(auc_list),np.std(auc_list)))
    print("test sensitivity mean:{}  std: {}".format(np.mean(sen_list),np.std(sen_list)))
    print("test specficity mean:{}  std: {}".format(np.mean(spec_list),np.std(spec_list)))

if __name__ == '__main__':
    main()
