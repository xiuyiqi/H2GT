import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def detect_hub_nodes(fc_matrix, node_clus_map_path, thrWD=1.0, thrP=0.3, plot_results=True):
    """
    从群体级功能连接矩阵中识别省级枢纽和连接枢纽节点
    
    参数:
    fc_matrix (np.ndarray): 群体级功能连接矩阵 (200x200)
    node_clus_map_path (str): 脑区和功能模块对应文件的路径
    thrWD (float): 模块内度阈值 (默认1.0)
    thrP (float): 参与系数阈值 (默认0.3)
    plot_results (bool): 是否绘制结果可视化图表 (默认True)
    
    返回:
    dict: 包含枢纽节点信息的字典
    """
    # 1. 加载脑区和功能模块映射
    with open(node_clus_map_path, 'rb') as file:
        node_clus_map = pickle.load(file)
    
    # 确保映射长度匹配
    if len(node_clus_map) != fc_matrix.shape[0]:
        raise ValueError(f"节点映射长度({len(node_clus_map)})与FC矩阵大小({fc_matrix.shape[0]})不匹配")
    
    # 2. 创建加权无向图
    G = nx.Graph()
    n_nodes = fc_matrix.shape[0]
    G.add_nodes_from(range(n_nodes))
    
    # 添加边（忽略自连接）
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if fc_matrix[i, j] != 0:  # 忽略零权重
                # 使用绝对值作为连接强度
                G.add_edge(i, j, weight=abs(fc_matrix[i, j]))
    
    # 3. 提取模块信息
    modules = {}
    for node, module_id in node_clus_map.items():
        if module_id not in modules:
            modules[module_id] = []  # 如果模块ID还没有对应的节点列表，则初始化一个
        modules[module_id].append(node)  # 将节点加入对应模块的列表
    
    n_modules = len(modules)
    print(f"检测到 {n_modules} 个功能模块:")
    for module_id, nodes in modules.items():
        print(f"  模块 {module_id}: {len(nodes)} 个节点")
    
    # 4. 计算每个节点的总连接强度（加权度）
    total_strength = np.zeros(n_nodes)
    for i in range(n_nodes):
        # 忽略自连接（对角线）
        total_strength[i] = np.sum(np.abs(fc_matrix[i, :])) - np.abs(fc_matrix[i, i])
    
    # 5. 计算每个节点的模块内连接强度
    within_strength = np.zeros(n_nodes)
    for node in range(n_nodes):
        module_id = node_clus_map[node]
        module_nodes = modules[module_id]
        # 计算与同模块节点的连接强度（忽略自连接）
        within_strength[node] = np.sum([
            abs(fc_matrix[node, n]) for n in module_nodes if n != node
        ])
    
    # 6. 计算模块内度（WD） - 模块内强度的Z分数
    WD = np.zeros(n_nodes)
    module_stats = {}
    
    for module_id, nodes in modules.items():
        module_nodes = np.array(nodes)
        module_within = within_strength[module_nodes]
        
        if len(module_nodes) > 1:
            mean_wd = np.mean(module_within)
            std_wd = np.std(module_within)
        else:
            mean_wd = module_within[0] if len(module_nodes) == 1 else 0
            std_wd = 1.0  # 避免除以零
        
        module_stats[module_id] = (mean_wd, std_wd)
        
        # 计算模块内节点的WD
        for node in module_nodes:
            if std_wd > 0:
                WD[node] = (within_strength[node] - mean_wd) / std_wd
            else:
                WD[node] = 0
    
    # 7. 计算参与系数（P）
    P = np.zeros(n_nodes)
    
    for node in range(n_nodes):
        p_term = 0.0
        # 计算节点到每个模块的连接强度
        for module_id, module_nodes in modules.items():
            # 计算节点到当前模块的总连接强度
            module_strength = np.sum([
                abs(fc_matrix[node, n]) for n in module_nodes if n != node
            ])
            
            # 计算比例并累加
            if total_strength[node] > 0:
                ratio = module_strength / total_strength[node]
                p_term += ratio ** 2
        
        # 参与系数公式
        P[node] = 1.0 - p_term
    
    # 8. 识别枢纽节点（全局索引）
    provincial_hubs_global = []
    connector_hubs_global = []
    
    for node in range(n_nodes):
        if WD[node] > thrWD:
            if P[node] < thrP:
                provincial_hubs_global.append(node)
            elif P[node] > thrP:
                connector_hubs_global.append(node)
    
    # 9. 创建模块内局部索引的枢纽信息
    hub_info: Dict[str, List[List[int]]] = {
        "provincial": [[] for _ in range(n_modules)],
        "connector": [[] for _ in range(n_modules)]
    }
    
    # 为每个模块创建全局节点到局部索引的映射
    module_local_mapping = {}
    for module_id, module_nodes in modules.items():
        # 创建映射：全局节点索引 -> 模块内局部索引
        local_mapping = {global_idx: local_idx for local_idx, global_idx in enumerate(module_nodes)}
        module_local_mapping[module_id] = local_mapping
    
    # 填充省级枢纽信息（按模块）
    for global_idx in provincial_hubs_global:
        module_id = node_clus_map[global_idx]
        local_idx = module_local_mapping[module_id].get(global_idx)
        if local_idx is not None:
            hub_info["provincial"][module_id].append(local_idx)
    
    # 填充连接枢纽信息（按模块）
    for global_idx in connector_hubs_global:
        module_id = node_clus_map[global_idx]
        local_idx = module_local_mapping[module_id].get(global_idx)
        if local_idx is not None:
            hub_info["connector"][module_id].append(local_idx)
    
    # 10. 结果可视化
    if plot_results:
        plot_hub_detection_results(WD, P, provincial_hubs_global, connector_hubs_global, thrWD, thrP, n_modules)
    
    # 11. 返回结果
    return {
        'hub_info': hub_info,  # 枢纽节点信息（模块内局部索引）
        'provincial_hubs': provincial_hubs_global,  # 省级枢纽（全局索引）
        'connector_hubs': connector_hubs_global,    # 连接枢纽（全局索引）
        'WD': WD,
        'P': P,
        'modules': modules,
        'module_stats': module_stats,
        'total_strength': total_strength,
        'within_strength': within_strength
    }

def plot_hub_detection_results(WD, P, provincial_hubs, connector_hubs, thrWD, thrP, n_modules):
    """Plot the hub detection results visualization"""
    plt.figure(figsize=(15, 10))
    
    # 1. Scatter plot of WD and P
    plt.subplot(221)
    plt.scatter(WD, P, alpha=0.6, s=30, label='Normal Nodes')  # 普通节点
    plt.scatter(WD[provincial_hubs], P[provincial_hubs], color='red', s=50, 
                label=f'Provincial Hubs ({len(provincial_hubs)} nodes)')  # 省级枢纽
    plt.scatter(WD[connector_hubs], P[connector_hubs], color='blue', s=50, 
                label=f'Connector Hubs ({len(connector_hubs)} nodes)')  # 连接枢纽
    
    plt.axvline(x=thrWD, color='gray', linestyle='--', alpha=0.7)  # 设置WD阈值的竖线
    plt.axhline(y=thrP, color='gray', linestyle='--', alpha=0.7)  # 设置P阈值的横线
    
    plt.xlabel('Module Degree (WD)')  # 模块内度 (WD)
    plt.ylabel('Participation Coefficient (P)')  # 参与系数 (P)
    plt.title(f'Hub Node Detection (Thresholds: WD>{thrWD}, P<{thrP} or P>{thrP})')  # 枢纽节点识别
    plt.legend()  # 图例
    plt.grid(True, alpha=0.3)  # 网格
    
    # 2. WD distribution plot
    plt.subplot(222)
    plt.hist(WD, bins=30, color='skyblue', edgecolor='black', alpha=0.7)  # WD分布图
    plt.axvline(x=thrWD, color='red', linestyle='--', label=f'WD Threshold ({thrWD})')  # WD阈值
    plt.xlabel('Module Degree (WD)')  # 模块内度 (WD)
    plt.ylabel('Node Count')  # 节点数量
    plt.title('Module Degree Distribution')  # 模块内度分布
    plt.legend()  # 图例
    plt.grid(True, alpha=0.3)  # 网格
    
    # 3. P distribution plot
    plt.subplot(223)
    plt.hist(P, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)  # P分布图
    plt.axvline(x=thrP, color='red', linestyle='--', label=f'P Threshold ({thrP})')  # P阈值
    plt.xlabel('Participation Coefficient (P)')  # 参与系数 (P)
    plt.ylabel('Node Count')  # 节点数量
    plt.title('Participation Coefficient Distribution')  # 参与系数分布
    plt.legend()  # 图例
    plt.grid(True, alpha=0.3)  # 网格
    
    # 4. Distribution of hub nodes across modules
    plt.subplot(224)
    hub_types = ['Provincial Hubs', 'Connector Hubs']  # 枢纽类型
    hub_counts = [len(provincial_hubs), len(connector_hubs)]  # 各类型枢纽节点数量
    plt.bar(hub_types, hub_counts, color=['red', 'blue'])  # 柱状图

    plt.xlabel('Hub Type')  # 枢纽类型
    plt.ylabel('Node Count')  # 节点数量
    plt.title(f'Hub Node Distribution (Total: {sum(hub_counts)})')  # 枢纽节点分布
    
    for i, count in enumerate(hub_counts):
        plt.text(i, count + 0.5, str(count), ha='center')  # 显示每个柱子的值
    
    plt.tight_layout()  # 紧凑布局
    plt.savefig('hub_detection_results.png', dpi=300)  # 保存图像
    plt.show()  # 显示图像
    
    # 5. Additional visualization: Module connection strength
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[WD, P], inner="quartile", palette="Set2")  # 小提琴图
    plt.xticks([0, 1], ['Module Degree (WD)', 'Participation Coefficient (P)'])  # x轴标签
    plt.title('Node Metric Distribution')  # 节点指标分布
    plt.grid(True, alpha=0.3)  # 网格
    plt.tight_layout()  # 紧凑布局
    plt.savefig('hub_metrics_distribution.png', dpi=300)  # 保存图像

# 示例使用
if __name__ == "__main__":
    # 1. 创建示例FC矩阵 (200x200)
    # np.random.seed(42)
    # fc_matrix = np.random.rand(200, 200)
    # fc_matrix = (fc_matrix + fc_matrix.T) / 2  # 使矩阵对称
    # np.fill_diagonal(fc_matrix, 0)  # 对角线置零
    fc_matrix = np.load('path to cbt.npy')
    
    
    # 3. 检测枢纽节点
    results = detect_hub_nodes(
        fc_matrix=fc_matrix,
        node_clus_map_path='path to node_clus_map.pickle',
        thrWD=1.0,  # 模块内度阈值
        thrP=0.85    # 参与系数阈值
    )
    
    # 4. 打印结果
    print("\n检测结果摘要:")
    print(f"省级枢纽节点数量: {len(results['provincial_hubs'])}")
    print(f"连接枢纽节点数量: {len(results['connector_hubs'])}")
    
    # 打印枢纽信息（模块内局部索引）
    hub_info = results['hub_info']
    print("\n按模块组织的枢纽信息:")
    for module_id in range(len(hub_info['provincial'])):
        print(f"模块 {module_id}:")
        print(f"  省级枢纽节点: {hub_info['provincial'][module_id]}")
        print(f"  连接枢纽节点: {hub_info['connector'][module_id]}")
    
    # 打印前10个省级枢纽（全局索引）
    print("\n省级枢纽节点 (前10个，全局索引):")
    for i, node in enumerate(results['provincial_hubs'][:10]):
        print(f"  节点 {node}: WD={results['WD'][node]:.2f}, P={results['P'][node]:.2f}")
    
    # 打印前10个连接枢纽（全局索引）
    print("\n连接枢纽节点 (前10个，全局索引):")
    for i, node in enumerate(results['connector_hubs'][:10]):
        print(f"  节点 {node}: WD={results['WD'][node]:.2f}, P={results['P'][node]:.2f}")
    
    # 保存结果
    with open('hub_detection_results.pickle', 'wb') as f:
        pickle.dump(results, f)
    print("\n结果已保存到 'hub_detection_results.pickle'")