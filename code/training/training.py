from utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from typing import List
import torch.utils.data as utils
from components.lr_scheduler import LRScheduler
import logging
from datetime import datetime
import pickle

training_epochs = 50
train_set = 0.7
percentage = 1
batch_size = 64
train_length = int(1009*train_set*percentage)
steps_per_epoch = (train_length - 1) // batch_size + 1 # 每个训练epoch的步数
total_steps = steps_per_epoch * 50 # 总训练步数
repeat_time = 1
save_learnable_graph =  False
save_attn_weights = True
save_test_attn_weights = False

class Train:

    def __init__(self,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = training_epochs
        self.total_steps = total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")
        self.save_path = Path("path to result") / unique_id
        self.save_learnable_graph = save_learnable_graph
        self.save_attn_weights = save_attn_weights
        self.save_test_attn_weights = save_test_attn_weights

        self.init_meters()
        self.mseLoss = torch.nn.MSELoss()
        self.l1Loss = torch.nn.L1Loss()

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()
        # 遍历训练数据 - 现在返回18个元素
        for batch in self.train_dataloader:
            # 解包数据加载器返回的18个元素
            features = batch[:8]  # 8个特征矩阵
            adjacencies = batch[8:16]  # 8个邻接矩阵
            module_adjacency = batch[16]  # 模块邻接矩阵
            labels = batch[17]  # 标签
            
            self.current_step += 1
            # 更新学习率
            lr_scheduler.update(optimizer=optimizer, step=self.current_step)
            
            # 将数据移到GPU
            features = [f.cuda() for f in features]
            adjacencies = [a.cuda() for a in adjacencies]
            module_adjacency = module_adjacency.cuda()
            labels = labels.cuda().float()
            
            # 启用数据增强 - 需要重新设计以适应新的数据结构
            # 暂时跳过数据增强
            # if self.config.preprocess.continus:
            #     # 需要重新设计数据增强逻辑
            #     pass
            
            # 模型预测 - 传入所有输入
            predict = self.model(*features, *adjacencies, module_adjacency)
            
            # 交叉熵损失
            loss = self.loss_fn(predict, labels)
            
            # 更新训练损失
            self.train_loss.update_with_weight(loss.item(), labels.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            top1 = accuracy(predict, labels[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, labels.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels_list = []
        result_list = []

        self.model.eval()

        for batch in dataloader:
            # 解包数据
            features = batch[:8]  # 8个特征矩阵
            adjacencies = batch[8:16]  # 8个邻接矩阵
            module_adjacency = batch[16]  # 模块邻接矩阵
            labels = batch[17]  # 标签
            
            # 将数据移到GPU
            features = [f.cuda() for f in features]
            adjacencies = [a.cuda() for a in adjacencies]
            module_adjacency = module_adjacency.cuda()
            labels = labels.cuda().float()
            
            # 模型预测
            output = self.model(*features, *adjacencies, module_adjacency)
            
            # 计算损失
            loss = self.loss_fn(output, labels)
            loss_meter.update_with_weight(loss.item(), labels.shape[0])
            
            # 计算准确率
            top1 = accuracy(output, labels[:, 1])[0]
            acc_meter.update_with_weight(top1, labels.shape[0])
            
            # 收集结果
            result_list += F.softmax(output, dim=1)[:, 1].tolist()
            labels_list += labels[:, 1].tolist()

        # 计算AUC和其他指标
        auc = roc_auc_score(labels_list, result_list)
        result_arr = np.array(result_list)
        labels_arr = np.array(labels_list)
        result_arr[result_arr > 0.5] = 1
        result_arr[result_arr <= 0.5] = 0
        
        metric = precision_recall_fscore_support(labels_arr, result_arr, average='micro')
        report = classification_report(labels_arr, result_arr, output_dict=True, zero_division=0)
        
        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        
        return [auc] + list(metric) + recall

    def save_attention_weights(self):
        global_attn_weights = []
        local_attn_weights = []
        assign_matrices = []
        labels_list = []
        
        self.model.eval()
        
        # 处理训练集和验证集
        for dataloader in [self.train_dataloader, self.val_dataloader]:
            for batch in dataloader:
                # 解包数据
                features = batch[:8]
                adjacencies = batch[8:16]
                module_adjacency = batch[16]
                labels = batch[17]
                
                # 移到GPU
                features = [f.cuda() for f in features]
                adjacencies = [a.cuda() for a in adjacencies]
                module_adjacency = module_adjacency.cuda()
                
                # 模型预测
                predict = self.model(*features, *adjacencies, module_adjacency)
                
                # 获取全局注意力权重
                global_attn = self.model.get_attention_weights()
                global_attn_np = [attn.detach().cpu().numpy() for attn in global_attn]
                global_attn_weights.append(global_attn_np)
                
                # 获取局部注意力权重
                local_attn = self.model.get_local_attention_weights()
                local_attn_np = []
                for mod_attns in local_attn:
                    mod_np = [attn.detach().cpu().numpy() for attn in mod_attns]
                    local_attn_np.append(mod_np)
                local_attn_weights.append(local_attn_np)
                
                # 获取分配矩阵
                assign_mat = self.model.get_assign_mat()
                if assign_mat is not None:
                    assign_np = assign_mat.detach().cpu().numpy()
                    assign_matrices.append(assign_np)
                
                # 保存标签
                labels_np = labels.detach().cpu().numpy()
                labels_list.append(labels_np)
        
        # 处理测试集（如果需要）
        if self.save_test_attn_weights:
            global_attn_weights_test = []
            local_attn_weights_test = []
            assign_matrices_test = []
            labels_test = []
            
            for batch in self.test_dataloader:
                # 解包数据
                features = batch[:8]
                adjacencies = batch[8:16]
                module_adjacency = batch[16]
                labels = batch[17]
                
                # 移到GPU
                features = [f.cuda() for f in features]
                adjacencies = [a.cuda() for a in adjacencies]
                module_adjacency = module_adjacency.cuda()
                
                # 模型预测
                predict = self.model(*features, *adjacencies, module_adjacency)
                
                # 获取全局注意力权重
                global_attn = self.model.get_attention_weights()
                global_attn_np = [attn.detach().cpu().numpy() for attn in global_attn]
                global_attn_weights_test.append(global_attn_np)
                
                # 获取局部注意力权重
                local_attn = self.model.get_local_attention_weights()
                local_attn_np = []
                for mod_attns in local_attn:
                    mod_np = [attn.detach().cpu().numpy() for attn in mod_attns]
                    local_attn_np.append(mod_np)
                local_attn_weights_test.append(local_attn_np)
                
                # 获取分配矩阵
                assign_mat = self.model.get_assign_mat()
                if assign_mat is not None:
                    assign_np = assign_mat.detach().cpu().numpy()
                    assign_matrices_test.append(assign_np)
                
                # 保存标签
                labels_np = labels.detach().cpu().numpy()
                labels_test.append(labels_np)
        
        # 保存所有数据
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"global_attn_weights.npy", global_attn_weights, allow_pickle=True)
        # np.save(self.save_path/"local_attn_weights.npy", local_attn_weights, allow_pickle=True)
        with open(self.save_path / "local_attn_weights.pkl", "wb") as f:
            pickle.dump(local_attn_weights, f)
        np.save(self.save_path/"assign_matrices.npy", assign_matrices, allow_pickle=True)
        np.save(self.save_path/"labels.npy", labels_list, allow_pickle=True)
        
        if self.save_test_attn_weights:
            np.save(self.save_path/"global_attn_weights_test.npy", global_attn_weights_test, allow_pickle=True)
            # np.save(self.save_path/"local_attn_weights_test.npy", local_attn_weights_test, allow_pickle=True)
            with open(self.save_path / "local_attn_weights_test.pkl", "wb") as f:
                pickle.dump(local_attn_weights_test, f)
            np.save(self.save_path/"assign_matrices_test.npy", assign_matrices_test, allow_pickle=True)
            np.save(self.save_path/"labels_test.npy", labels_test, allow_pickle=True)

    def generate_save_learnable_matrix(self):
        learable_matrixs = []
        labels_list = []
        
        for batch in self.test_dataloader:
            # 解包数据
            features = batch[:8]
            adjacencies = batch[8:16]
            module_adjacency = batch[16]
            labels = batch[17]
            
            # 移到GPU
            features = [f.cuda() for f in features]
            adjacencies = [a.cuda() for a in adjacencies]
            module_adjacency = module_adjacency.cuda()
            
            # 模型预测
            predict = self.model(*features, *adjacencies, module_adjacency)
            
            # 获取可学习矩阵（这里需要根据模型实现确定）
            # 假设模型有返回可学习矩阵的方法
            if hasattr(self.model, 'get_learnable_matrix'):
                learable_matrix = self.model.get_learnable_matrix()
                learable_matrixs.append(learable_matrix.cpu().detach().numpy())
                labels_list += labels.tolist()
        
        # 保存结果
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", 
                {'matrix': np.vstack(learable_matrixs), "label": np.array(labels_list)}, 
                allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy", results, allow_pickle=True)
        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        self.current_step = 0
        best_val_AUC = 0
        best_test_acc = 0
        best_test_AUC = 0
        best_test_sen = 0
        best_test_spec = 0
        
        for epoch in range(self.epochs):
            self.reset_meters()
            
            # 训练一个epoch
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            
            # 验证集评估
            val_result = self.test_per_epoch(self.val_dataloader, self.val_loss, self.val_accuracy)
            
            # 测试集评估
            test_result = self.test_per_epoch(self.test_dataloader, self.test_loss, self.test_accuracy)
            
            # 日志记录
            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Test AUC:{test_result[0]:.4f}',
                f'Val Accuracy:{self.val_accuracy.avg: .3f}',
                f'Val Loss{self.val_loss.avg: .3f}',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.5f}'
            ]))
            
            # 更新最佳结果
            if val_result[0] > best_val_AUC:
                best_val_AUC = val_result[0]
                best_test_acc = self.test_accuracy.avg
                best_test_AUC = test_result[0]
                best_test_sen = test_result[-1]
                best_test_spec = test_result[-2]
            
            # 保存训练过程
            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                "Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
                "Val Accuracy": self.val_accuracy.avg,
            })
        
        # 保存注意力权重和可学习矩阵
        if self.save_attn_weights:
            self.save_attention_weights()
        
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        
        # 保存训练结果和模型
        self.save_result(training_process)
        
        return [best_test_acc, best_test_AUC, best_test_sen, best_test_spec]