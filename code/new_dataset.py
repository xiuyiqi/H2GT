import numpy as np
import pickle
from tqdm import tqdm
# 基于原数据字典生成新的数据字典
def reorder_adjacency_matrix(adj_matrix, node_clus_map):
    """重排邻接矩阵使其按模块顺序排列"""
    # 获取按模块排序的索引 (先按模块号排序，同模块内按原始索引排序)
    # sorted_indices = sorted(node_clus_map.keys(), 
    #                         key=lambda k: (node_clus_map[k], k))
    # 按模块排序
    sorted_indices = list(node_clus_map.keys())
    # 重排邻接矩阵
    adj_reordered = adj_matrix[sorted_indices, :]
    adj_reordered = adj_reordered[:, sorted_indices]
    return adj_reordered, sorted_indices

def create_binary_adjacency(matrix, percentile=30):
    """创建二值邻接矩阵 (保留前percentile%的强连接)"""
    # 复制矩阵并移除对角线
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    abs_mat = np.abs(mat)
    
    # 获取非零元素
    non_zero_values = abs_mat[abs_mat > 0]
    
    # 如果没有非零元素，返回全零矩阵
    if non_zero_values.size == 0:
        print("警告：矩阵中没有非零值。")
        return np.zeros_like(abs_mat)
    
    # 计算阈值 (保留前percentile%的值)
    threshold = np.percentile(non_zero_values, 100 - percentile)
    
    # 创建二值矩阵
    binary_adj = np.zeros_like(abs_mat)
    binary_adj[abs_mat >= threshold] = 1
    return binary_adj

def calculate_module_adjacency(adj_reordered, module_ranges):
    """计算模块级别的邻接矩阵"""
    n_modules = len(module_ranges)
    module_adj = np.zeros((n_modules, n_modules))
    
    for i, (start_i, end_i) in enumerate(module_ranges):
        for j, (start_j, end_j) in enumerate(module_ranges):
            # 提取两个模块间的连接子矩阵
            inter_matrix = adj_reordered[start_i:end_i, start_j:end_j]
            # 计算绝对值的平均值作为连接强度
            module_adj[i, j] = np.mean(np.abs(inter_matrix))
    return module_adj

# 加载脑区-模块映射
with open('node_clus_map.pickle', 'rb') as f:
    node_clus_map = pickle.load(f)

# 计算每个模块的索引范围
module_indices = {}
for node, module_id in node_clus_map.items():
    module_indices.setdefault(module_id, []).append(node)

# 确定模块ID顺序 (0-7)
module_ids = sorted(module_indices.keys())
module_ranges = []
current_start = 0
for module_id in module_ids:
    n_nodes = len(module_indices[module_id])
    module_ranges.append((current_start, current_start + n_nodes))
    current_start += n_nodes

# 加载原始数据
original_data = np.load('path to abide.npy', allow_pickle=True).item()
n_subjects = len(original_data['label'])

# 初始化新数据结构
new_data = {
    **{f'feature_matrix_{i}': [] for i in range(8)},  # 0-7 模块ID
    **{f'adjacency_matrix_{i}': [] for i in range(8)},  # 0-7 模块ID
    'module_adjacency': [],
    'label': [],
    'site':[]
}

# 处理每个被试
for i in tqdm(range(n_subjects)):
    # 获取当前被试数据
    corr_matrix = original_data['corr'][i]
    label = original_data['label'][i]
    site = original_data['site'][i]
    
    # 拓扑重排
    adj_reordered, _ = reorder_adjacency_matrix(corr_matrix, node_clus_map)
    
    # 生成每个模块的数据
    feature_matrices = []
    adjacency_matrices = []
    
    for mod_idx, (start, end) in enumerate(module_ranges):
        # 提取模块特征矩阵
        # mod_feature = adj_reordered[start:end, start:end]
        mod_feature = adj_reordered[start:end, :]
        feature_matrices.append(mod_feature)

        temp = adj_reordered[start:end, start:end]
        # 创建二值邻接矩阵
        mod_adj = create_binary_adjacency(temp)
        adjacency_matrices.append(mod_adj)
    
    # 计算模块级别邻接矩阵
    mod_level_adj = calculate_module_adjacency(adj_reordered, module_ranges)
    
    # 存储结果
    for mod_idx in range(8):  # 0-7 模块ID
        new_data[f'feature_matrix_{mod_idx}'].append(feature_matrices[mod_idx])
        new_data[f'adjacency_matrix_{mod_idx}'].append(adjacency_matrices[mod_idx])
    
    new_data['module_adjacency'].append(mod_level_adj)
    new_data['label'].append(label)
    new_data['site'].append(site)

# 保存新数据
np.save('new_dataset.npy', new_data)
print("新数据已保存为 new_dataset.npy")
