import torch
import torch.nn as nn
from .dec import DEC
from .component import InterpretableGraphTransformerEncoder
from .base import BaseModel
from typing import Optional
from torch import Tensor

class TransPoolingEncoder(nn.Module):
    """
    Graph Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(
        self,
        input_feature_size,  # 节点特征维度
        edge_feature_size,   # 边特征维度
        input_node_num,      # 输入节点数
        hidden_size,         # 隐藏层大小
        output_node_num,     # 输出节点数（池化后）
        pooling=True,        # 是否使用池化
        orthogonal=True,     # 是否正交化
        freeze_center=False, # 是否冻结聚类中心
        project_assignment=True, # 是否投影分配矩阵
        nHead=4,             # 注意力头数
        local_transformer=False # 是否为局部transformer
    ):
        super().__init__()
        
        # 使用可解释的图Transformer编码器
        self.transformer = InterpretableGraphTransformerEncoder(
            d_model=input_feature_size,
            nhead=nHead,
            edge_dim=edge_feature_size,
            dim_feedforward=hidden_size,
            dropout=0.1,
            gated_residual=True,
            with_feedforwards=False,
            rel_pos_emb=False,
            local_transformer=local_transformer
        )
        
        # 标记是否为局部transformer
        self.local_transformer = local_transformer
        
        # 局部transformer不定义池化层，全局transformer定义池化层
        if local_transformer:
            self.pooling = False
        else:
            self.pooling = pooling
            
        # 池化机制
        if self.pooling:
            encoder_hidden_size = 32
            # 全连接的编码器
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden_size), # (input_feature_size * 200, 32)
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
            )
            # 聚类模块
            self.dec = DEC(
                cluster_number=output_node_num,
                hidden_dimension=input_feature_size, # 200
                encoder=self.encoder,
                orthogonal=orthogonal,
                freeze_center=freeze_center,
                project_assignment=project_assignment
            )
        
    # 返回是否启用池化
    def is_pooling_enabled(self):
        return self.pooling

    def forward(
        self, 
        x: torch.Tensor, 
        edges: torch.Tensor,  # 添加边特征输入
        module_idx: Optional[int] = None
    ):
        """
        前向传播
        
        参数:
            x: 节点特征 [batch_size, node_num, feature_size]
            edges: 边特征 [batch_size, node_num, node_num, edge_dim]
        
        返回:
            node_output: 节点输出特征
            assignment: 分配矩阵（如果使用池化）
            graph_rep: 图级别表示
        """
        
        # 应用图Transformer
        x = self.transformer(nodes=x, edges=edges, module_idx = module_idx)  # 不再需要mask
        
        # 池化处理
        assignment = None
        if not self.pooling: # 局部
            # 局部transformer：使用三种池化拼接
            max_pool = torch.max(x, dim=1, keepdim=True)[0]   # [batch_size, 1, feature_size]
            avg_pool = torch.mean(x, dim=1, keepdim=True)     # [batch_size, 1, feature_size]
            sum_pool = torch.sum(x, dim=1, keepdim=True)       # [batch_size, 1, feature_size]
            # 拼接三种池化结果
            x = torch.cat([max_pool, avg_pool, sum_pool], dim=-1)  # [batch_size, 1, feature_size*3]

        else: # 全局
            # 全局transformer：使用DEC池化
            x, assignment = self.dec(x)
        
        return x, assignment
    
    def get_attention_weights(self) -> Optional[Tensor]:
        """获取注意力权重"""
        return self.transformer.get_attention_weights()
    
    def clear_attention_weights(self):
        """清空存储的注意力权重"""
        self.transformer.clear_attention_weights()
    
node_sz = 200
edge_sz = 1
node_sz_per_module = [41, 29, 21, 19, 20, 7, 21, 42]
orthogonal=True,     # 是否正交化
freeze_center=False, # 是否冻结聚类中心
project_assignment=True, # 是否投影分配矩阵
nhead = 8
num_MHSA = 1
sizes = [360, 8]
pooling = [False,True]

class H2GT(BaseModel):

    def __init__(self):
        super().__init__()

        self.attention_list = nn.ModuleList()  # 注意力模块列表初始化
        forward_dim = node_sz  # 节点特征维度，输入特征维度

        self.num_modules = 8  # 有8个模块

        # 局部transformer用于处理每个模块
        self.local_transformers = nn.ModuleList([
            TransPoolingEncoder(
                input_feature_size=forward_dim,
                edge_feature_size=edge_sz,  # 添加边特征维度
                input_node_num=node_sz_per_module[i],  # 每个模块的节点数
                hidden_size=1024,
                output_node_num=1, 
                pooling=True,  # 启用池化
                orthogonal=orthogonal,
                freeze_center=freeze_center,
                project_assignment=project_assignment,
                nHead=nhead,
                local_transformer=True  # 标记为局部transformer
            ) for i in range(self.num_modules)
        ])
        
        # 全局transformer配置
        self.num_MHSA = num_MHSA  # 获取多头自注意力层的数量
        sizes = [360, 8]  # 获取网络中各层的尺寸
        sizes[0] = self.num_modules  # 设置第一个尺寸为模块数量
        in_sizes = [self.num_modules] + sizes[:-1]  # 构建每一层输入的大小列表
        do_pooling = pooling  # 获取池化设置
        self.do_pooling = do_pooling 
        
        # 全局transformer初始化
        if num_MHSA == 1:
            self.attention_list.append(
                TransPoolingEncoder(
                    # input_feature_size=self.local_transformers[0].pooled_feature_size,  # 使用局部transformer的输出特征维度
                    input_feature_size = forward_dim * 3,
                    edge_feature_size=edge_sz,  # 边特征维度
                    input_node_num=in_sizes[1],
                    hidden_size=1024,
                    output_node_num=sizes[1],
                    # output_node_num=1, # 2-改进全局池化
                    pooling=do_pooling[1],
                    orthogonal=orthogonal,
                    freeze_center=freeze_center,
                    project_assignment=project_assignment,
                    nHead=nhead,
                    local_transformer=False
                )
            )
        else:
            for index, size in enumerate(sizes):
                self.attention_list.append(
                    TransPoolingEncoder(
                        input_feature_size=forward_dim*3,
                        edge_feature_size=edge_sz,
                        input_node_num=in_sizes[index],
                        hidden_size=1024,
                        output_node_num=size,
                        pooling=do_pooling[index],
                        orthogonal=orthogonal,
                        freeze_center=freeze_center,
                        project_assignment=project_assignment,
                        nHead=nhead,
                        local_transformer=False
                    )
                )
        
        # 维度缩减层
        # self.dim_reduction = nn.Sequential(
        #     # nn.Linear(self.local_transformers[0].pooled_feature_size, 8),  # 使用局部transformer的输出特征维度
        #     nn.Linear(forward_dim * 3, 8),
        #     nn.LeakyReLU()
        # )
        # 1-优化降维思路
        self.dim_reduction = nn.Sequential(
        nn.Linear(600, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 8)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
        # 4-建议
        # self.fc = nn.Sequential(
        #         nn.Linear(64, 128),
        #         nn.LeakyReLU(),
        #         nn.Linear(128, 64),
        #         nn.LeakyReLU(),
        #         nn.Linear(64, 32),
        #         nn.LeakyReLU(),
        #         nn.Linear(32, 2)
        #     )
                
        # 赋值矩阵
        self.assignMat = None
        
        # MLP用于处理模块表示
        self.mlp = nn.Sequential(
            # nn.Linear(8 * self.local_transformers[0].pooled_feature_size, 512),
            nn.Linear(8 * forward_dim * 3, 512),
            nn.LeakyReLU(),
            # nn.Linear(512, self.local_transformers[0].pooled_feature_size),
            nn.Linear(512, forward_dim * 3),
            nn.LeakyReLU()
        )
        # 3-建议：跨模块注意力
        # self.cross_attention = nn.MultiheadAttention(600, 8)

    def forward(self, *inputs):
        """
        前向传播
        输入格式:
            feature_matrix_0, feature_matrix_1, ..., feature_matrix_7 (8个)
            adjacency_matrix_0, adjacency_matrix_1, ..., adjacency_matrix_7 (8个)
            module_adjacency (模块级别的邻接矩阵)
            labels (标签)
        """
        # 解包输入
        features = inputs[0:8]  # 前8个是特征矩阵
        adjacencies = inputs[8:16]  # 接下来8个是邻接矩阵
        module_adjacency = inputs[16]  # 模块级别的邻接矩阵
        
        batch_size = features[0].shape[0] # 64
        
        # 处理每个模块的局部transformer
        module_representations = []
        for i in range(self.num_modules):
            # 处理单个模块
            mod_rep, _ = self.local_transformers[i](
                x=features[i], 
                edges=adjacencies[i].unsqueeze(-1),
                module_idx = i
            )
            module_representations.append(mod_rep)
        
        # 将所有模块表示拼接成模块级别的图
        # 每个模块表示形状: [batch_size, 1, feature_size*3]
        # 拼接后形状: [batch_size, 8, feature_size*3]
        module_graph = torch.cat(module_representations, dim=1)

        
        # 使用MLP增强模块表示--可省略
        # module_graph_flat = module_graph.reshape(batch_size, -1) # （64， 4800）
        # module_graph_enhanced = self.mlp(module_graph_flat)
        # module_graph_enhanced = module_graph_enhanced.reshape(
        #     batch_size, self.num_modules, -1
        # )#(64, 8, 75)
        
        # 全局transformer处理模块级别的图
        # 使用模块级别的邻接矩阵作为边特征
        if self.num_MHSA == 1:
            global_rep, assign_mat = self.attention_list[0](
                x=module_graph, 
                edges=module_adjacency.unsqueeze(-1)  # 添加边特征维度
            )
            self.assignMat = assign_mat
        else:
            global_rep = module_graph
            for atten in self.attention_list:
                global_rep, _ = atten(
                    x=global_rep, 
                    edges=module_adjacency.unsqueeze(-1))  # 使用相同的模块邻接矩阵
        
        # 维度缩减
        reduced_rep = self.dim_reduction(global_rep)
        
        # 展平
        flat_rep = reduced_rep.reshape(batch_size, -1)
        
        # 分类
        return self.fc(flat_rep)

    # 返回分配矩阵
    def get_assign_mat(self):
        return self.assignMat
    
    # 返回全局transformer注意力权重
    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]
    
    # 返回局部transformer注意力权重
    def get_local_attention_weights(self):
        return [trans.get_attention_weights() for trans in self.local_transformers]
