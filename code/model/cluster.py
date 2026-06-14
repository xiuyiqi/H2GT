"""
From https://github.com/vlukiyanov/pt-dec
"""

import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional
from torch.nn.functional import softmax

# 计算特征向量到每个聚类中心的“软”分配 聚类分配
class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
        orthogonal=True,
        freeze_center=True,
        project_assignment=True
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.project_assignment = project_assignment
        if cluster_centers is None: # 如果没有提供cluster_centers,则用全零矩阵初始化聚类中心
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers) # 并使用Xavier均匀分布初始化

        else:
            initial_cluster_centers = cluster_centers

        if orthogonal: # 对聚类中心进行正交化处理，确保聚类中心之间的正交性
            orthogonal_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            orthogonal_cluster_centers[0] = initial_cluster_centers[0]
            for i in range(1, cluster_number):
                project = 0
                for j in range(i):
                    project += self.project(initial_cluster_centers[j], initial_cluster_centers[i]) # 累积所有之前按聚类中心对当前聚类中心的投影
                initial_cluster_centers[i] -= project # 减去投影部分
                orthogonal_cluster_centers[i] = initial_cluster_centers[i] / torch.norm(initial_cluster_centers[i], p=2) # 变为单位向量

            initial_cluster_centers = orthogonal_cluster_centers

        self.cluster_centers = Parameter(
            initial_cluster_centers, requires_grad=(not freeze_center)) # 将聚类中心作为可学习的参数，并决定是否冻结其梯度
    # 静态方法 计算向量v在向量u上的投影
    @staticmethod
    def project(u, v):
        return (torch.dot(u, v)/torch.dot(u, u))*u

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """

        if self.project_assignment:

            assignment = batch@self.cluster_centers.T # 计算每个特征向量对每个聚类中心的内积
            # prove
            assignment = torch.pow(assignment, 2) # 平方

            norm = torch.norm(self.cluster_centers, p=2, dim=-1)
            soft_assign = assignment/norm # 归一化
            return softmax(soft_assign, dim=-1) # 经过sofetmax处理

        else:
            norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2) # 计算每个样本和聚类中心的距离，欧氏距离的平方
            numerator = 1.0 / (1.0 + (norm_squared / self.alpha)) 
            power = float(self.alpha + 1) / 2
            numerator = numerator ** power # 根据距离和alpha参数，计算软分配
            return numerator / torch.sum(numerator, dim=1, keepdim=True) # 归一化
    # 获取当前的聚类中心
    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers.

        :return: FloatTensor [number of clusters, embedding dimension]
        """
        return self.cluster_centers
