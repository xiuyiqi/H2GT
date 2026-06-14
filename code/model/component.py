# from torch.nn import TransformerEncoderLayer
# from torch import Tensor
# from typing import Optional
# import torch.nn.functional as F
# import torch


# # 构建具有自注意力机制的 Transformer 编码器
# class InterpretableTransformerEncoder(TransformerEncoderLayer):
#     # 初始化Transformer编码器层的所有参数，定义了一个可选的属性来保存自注意力层计算出的注意力权重
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.3, activation=F.relu,
#                  layer_norm_eps=1e-5, batch_first=False, norm_first=False,
#                  device=None, dtype=None) -> None:
#         super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
#                          layer_norm_eps, batch_first, norm_first, device, dtype)
#         self.attention_weights: Optional[Tensor] = None
#         print(f"nhead = {nhead}")
#     # 自注意力块
#     def _sa_block(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x, weights = self.self_attn(x, x, x,
#                                     attn_mask=attn_mask,
#                                     key_padding_mask=key_padding_mask,
#                                     need_weights=True,
#                                     average_attn_weights=False)
#         self.attention_weights = weights
#         return self.dropout1(x)
#     # 获取注意力权重
#     def get_attention_weights(self) -> Optional[Tensor]:
#         return self.attention_weights



# from graph_transformer_pytorch import GraphTransformerWithWeights
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from typing import Optional

# class InterpretableGraphTransformerEncoder(nn.Module):
#     """
#     精简版可解释图 Transformer 编码器
#     核心功能：处理图结构数据并返回注意力权重
#     """
#     def __init__(
#         self,
#         d_model: int,                  # 节点特征维度
#         nhead: int,                    # 注意力头数
#         edge_dim: int,                 # 边特征维度
#         dim_feedforward: int = 2048,   # 前馈网络维度
#         dropout: float = 0.3,          # Dropout 概率
#         gated_residual: bool = True,   # 使用门控残差连接
#         with_feedforwards: bool = True, # 包含前馈网络
#         rel_pos_emb: bool = False,       # 使用相对位置编码
#         local_transformer: bool = True # 是否为局部Transformer
#     ) -> None:
#         super().__init__()
        
#         # 创建单层图 Transformer
#         self.transformer = GraphTransformerWithWeights(
#             dim=d_model,
#             depth=1,  # 单层
#             dim_head=dim_feedforward // nhead,
#             edge_dim=edge_dim,
#             heads=nhead,
#             gated_residual=gated_residual,
#             with_feedforwards=with_feedforwards,
#             rel_pos_emb=rel_pos_emb,
#             return_attn=True
#         )
    
#     def forward(
#         self, 
#         nodes: Tensor,
#         edges: Tensor,
#         mask: Optional[Tensor] = None
#     ) -> Tensor:
#         """
#         前向传播
        
#         参数:
#             nodes: 节点特征 [batch_size, seq_len, d_model]
#             edges: 边特征 [batch_size, seq_len, seq_len, edge_dim]
#             mask: 节点掩码 [batch_size, seq_len] (True 表示有效位置)
        
#         返回:
#             输出节点特征 [batch_size, seq_len, d_model]
#         """
        
#         # 应用图 Transformer
#         output, _ = self.transformer(
#             nodes=nodes,
#             edges=edges,
#             mask=mask
#         )
        
#         return output
    
#     def get_attention_weights(self) -> Optional[Tensor]:
#         """
#         获取注意力权重
        
#         返回:
#             注意力权重张量 [batch_size * heads, seq_len, seq_len]
#             如果没有权重则返回 None
#         """
#         weights = self.transformer.get_attention_weights()
#         return weights[0] if weights else None
    
#     def clear_attention_weights(self):
#         """清空存储的注意力权重"""
#         self.transformer.clear_attention_weights()


# 带有枢纽节点增强注意力
from graph_transformer_pytorch import GraphTransformerWithWeights
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict
import pickle

# hub_info: Dict[str, List[List[int]]]  # 枢纽节点信息
hub_boost_factor: float = 1.5  # 枢纽节点增强因子

with open('path to hub_detection_results.pickle', 'rb') as f:
    data = pickle.load(f)

hub_info = data.get('hub_info', None)

# if hub_info:
#     for key, value in hub_info.items():
#         print(f"节点 {key}: {value}")

class InterpretableGraphTransformerEncoder(nn.Module):
    """
    增强版可解释图 Transformer 编码器
    支持两种模式：
    - 局部Transformer：加强特定模块的省级枢纽节点
    - 全局Transformer：基于各模块连接枢纽数量增强模块表示
    """
    def __init__(
        self,
        d_model: int,                  # 节点特征维度
        nhead: int,                    # 注意力头数
        edge_dim: int,                 # 边特征维度
        dim_feedforward: int = 2048,   # 前馈网络维度
        dropout: float = 0.3,          # Dropout 概率
        gated_residual: bool = True,   # 使用门控残差连接
        with_feedforwards: bool = True, # 包含前馈网络
        rel_pos_emb: bool = False,     # 使用相对位置编码
        local_transformer: bool = True, # 是否为局部Transformer
    ) -> None:
        super().__init__()
        
        # 保存枢纽节点信息和配置
        self.provincial_hubs = hub_info["provincial"]  # 省级枢纽 [模块][节点]
        self.connector_hubs = hub_info["connector"]    # 连接枢纽 [模块][节点]
        self.local_transformer = local_transformer
        self.hub_boost_factor = hub_boost_factor
        self.num_modules = len(self.connector_hubs)
        
        # 预计算连接枢纽数量（全局模式使用）
        self.connector_counts = torch.tensor(
            [len(hubs) for hubs in self.connector_hubs], 
            dtype=torch.float32
        )
        
        # 创建图 Transformer
        self.transformer = GraphTransformerWithWeights(
            dim=d_model,
            depth=1,  # 单层
            dim_head=dim_feedforward // nhead,
            edge_dim=edge_dim,
            heads=nhead,
            gated_residual=gated_residual,
            with_feedforwards=with_feedforwards,
            rel_pos_emb=rel_pos_emb,
            return_attn=True
        )
        
        # 全局Transformer：连接枢纽注意力增强网络
        if not local_transformer:
            self.connector_attention_boost = nn.Sequential(
                nn.Linear(self.num_modules, self.num_modules),
                nn.ReLU(),
                nn.Linear(self.num_modules, self.num_modules)
            )
    
    def forward(
        self, 
        nodes: Tensor,
        edges: Tensor,
        mask: Optional[Tensor] = None,
        module_idx: Optional[int] = None  # 当前处理的模块索引（仅局部模式需要）
    ) -> Tensor:
        """
        前向传播
        
        参数:
            nodes: 节点特征 [batch_size, seq_len, d_model]
            edges: 边特征 [batch_size, seq_len, seq_len, edge_dim]
            mask: 节点掩码 [batch_size, seq_len] (True 表示有效位置)
            module_idx: 当前处理的模块索引（仅局部模式需要）
        
        返回:
            输出节点特征 [batch_size, seq_len, d_model]
        """
        # 增强枢纽节点影响
        nodes = self.enhance_hub_nodes(nodes, module_idx)
        
        # 应用图 Transformer
        output, _ = self.transformer(
            nodes=nodes,
            edges=edges,
            mask=mask
        )
        
        return output
    
    def enhance_hub_nodes(self, nodes: Tensor, module_idx: int) -> Tensor:
        """
        增强枢纽节点的影响
        
        参数:
            nodes: 原始节点特征 [batch_size, seq_len, d_model]
            module_idx: 当前处理的模块索引（仅局部模式需要）
        
        返回:
            增强后的节点特征 [batch_size, seq_len, d_model]
        """
        if self.local_transformer:
            # 局部Transformer模式：加强省级枢纽节点
            if module_idx is None:
                raise ValueError("模块索引必须提供用于局部Transformer模式")
            
            # 获取当前模块的省级枢纽节点索引
            hub_indices = self.provincial_hubs[module_idx]
            if not hub_indices:
                return nodes
            
            # 创建增强因子张量
            boost_factor = torch.ones_like(nodes)
            hub_tensor = torch.tensor(hub_indices, device=nodes.device)
            
            # 过滤无效索引
            valid_mask = hub_tensor < nodes.size(1)
            hub_tensor = hub_tensor[valid_mask]
            
            if hub_tensor.numel() > 0:
                # 向量化增强枢纽节点
                boost_factor.index_fill_(
                    dim=1, 
                    index=hub_tensor, 
                    value=1 + self.hub_boost_factor
                )
                return nodes * boost_factor
            
            return nodes
        else:
            # 全局Transformer模式：基于连接枢纽数量增强模块表示
            # 确保输入节点数匹配模块数
            if nodes.size(1) != self.num_modules:
                raise ValueError(
                    f"全局Transformer需要{self.num_modules}个模块节点，"
                    f"实际收到{nodes.size(1)}个节点"
                )
            
            # 计算连接枢纽增强因子
            counts = self.connector_counts.to(nodes.device)
            normalized_counts = F.softmax(counts, dim=0)
            attention_boost = self.connector_attention_boost(normalized_counts)
            attention_boost = F.softmax(attention_boost, dim=0)
            
            # 向量化增强模块表示
            boost_factor = (1 + attention_boost).view(1, -1, 1)
            return nodes * boost_factor
    
    def get_attention_weights(self) -> Optional[Tensor]:
        """
        获取注意力权重
        
        返回:
            注意力权重张量 [batch_size * heads, seq_len, seq_len]
            如果没有权重则返回 None
        """
        weights = self.transformer.get_attention_weights()
        return weights[0] if weights else None
    
    def clear_attention_weights(self):
        """清空存储的注意力权重"""
        self.transformer.clear_attention_weights()