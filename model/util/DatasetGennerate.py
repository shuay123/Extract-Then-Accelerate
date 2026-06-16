import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
import argparse
from sklearn.model_selection import train_test_split

class ClusteringDataset(Dataset):
    """集合划分数据集"""
    
    def __init__(self, n_samples, m, n, n_nodes=7, n_clusters=3):
        self.n_samples = n_samples
        self.m = m
        self.n = n
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        
        self.data = self.generate_data()
    
    def generate_data(self):
        """生成模拟数据"""
        data = []
        
        for _ in range(self.n_samples):
            # 生成节点特征（加工能力）
            node_features = torch.rand(self.n_nodes, self.m * self.n) * 0.8 + 0.1
            
            # 生成集合划分
            cluster_assignment = torch.randint(0, self.n_clusters, (self.n_nodes,))
            
            # 生成对称的边标签矩阵
            labels = torch.zeros(self.n_nodes, self.n_nodes)
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if cluster_assignment[i] == cluster_assignment[j]:
                        labels[i, j] = 1.0
            
            data.append((node_features, labels))
        
        return data
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

class MeaningfulClusteringDataset(Dataset):
    """有意义的集合划分数据集 - 特征与标签相关"""
    
    def __init__(self, n_samples, m, n, n_nodes=7, n_clusters=2, 
                 cluster_separation=2.0):
        self.n_samples = n_samples
        self.m = m
        self.n = n
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        self.feature_dim = m * n
        self.cluster_separation = cluster_separation  # 簇间距离
        
        self.data = self.generate_data()
    
    def generate_cluster_centers(self):
        """
        生成有区分度的聚类中心
        
        关键：让不同cluster的中心在特征空间中远离
        """
        centers = []
        
        for c in range(self.n_clusters):
            # 方法1：在特征空间的不同区域
            # 将特征空间分成n_clusters个区域
            base_value = c / self.n_clusters  # 0, 0.33, 0.67
            
            # 每个cluster的中心在不同的特征区域
            center = torch.ones(self.feature_dim) * (0.3 + base_value * 0.4)
            # cluster 0: ~0.3
            # cluster 1: ~0.43
            # cluster 2: ~0.57
            
            # 添加一些随机扰动
            center += torch.randn(self.feature_dim) * 0.05
            center = torch.clamp(center, 0.1, 0.9)
            
            centers.append(center)
        
        return centers
    
    def generate_data(self):
        """生成有意义的训练数据"""
        data = []
        
        for _ in range(self.n_samples):
            # Step 1: 生成聚类中心
            cluster_centers = self.generate_cluster_centers()
            
            # Step 2: 随机分配节点到各个集合
            cluster_assignment = torch.randint(0, self.n_clusters, (self.n_nodes,))
            
            # Step 3: 根据cluster生成节点特征（关键！）
            node_features = torch.zeros(self.n_nodes, self.feature_dim)
            
            for i in range(self.n_nodes):
                cluster_id = cluster_assignment[i].item()
                cluster_center = cluster_centers[cluster_id]
                
                # 🔥 关键：从cluster中心采样，添加高斯噪声
                # 同一cluster的节点特征相似
                noise = torch.randn(self.feature_dim) * 0.1  # 标准差0.1
                node_features[i] = cluster_center + noise
                node_features[i] = torch.clamp(node_features[i], 0.0, 1.0)
            
            # Step 4: 生成边标签（基于cluster）
            labels = torch.zeros(self.n_nodes, self.n_nodes)
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if cluster_assignment[i] == cluster_assignment[j]:
                        labels[i, j] = 1.0
            
            data.append((node_features, labels))
        
        return data
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

from sklearn.datasets import make_blobs

class RealisticClusteringDataset(Dataset):
    """使用sklearn生成真实的聚类数据"""
    
    def __init__(self, n_samples, m, n, n_nodes=7, n_clusters=3):
        self.n_samples = n_samples
        self.m = m
        self.n = n
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        self.feature_dim = m * n
        
        self.data = self.generate_data()
    
    def generate_data(self):
        data = []
        
        for _ in range(self.n_samples):
            # 使用make_blobs生成有真实聚类结构的数据
            X, y = make_blobs(
                n_samples=self.n_nodes,
                n_features=self.feature_dim,
                centers=self.n_clusters,
                cluster_std=0.5,  # 簇内标准差
                center_box=(0.0, 1.0),  # 中心在[0,1]范围内
                random_state=None
            )
            
            # 转换为torch tensor
            node_features = torch.FloatTensor(X)
            node_features = torch.clamp(node_features, 0.0, 1.0)
            
            cluster_assignment = torch.LongTensor(y)
            
            # 生成边标签
            labels = torch.zeros(self.n_nodes, self.n_nodes)
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if cluster_assignment[i] == cluster_assignment[j]:
                        labels[i, j] = 1.0
            
            data.append((node_features, labels))
        
        return data
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

class SemanticClusteringDataset(Dataset):
    """
    基于加工能力语义的数据生成
    
    设计思路：
    - 不同集合擅长不同的产品/工序组合
    - 同一集合内的机器有相似的能力分布
    """
    
    def __init__(self, n_samples, m, n, n_nodes=7, n_clusters=3):
        self.n_samples = n_samples
        self.m = m  # 产品数
        self.n = n  # 工序数
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        
        self.data = self.generate_data()
    
    def generate_data(self):
        data = []
        
        for _ in range(self.n_samples):
            # Step 1: 为每个cluster定义"擅长的能力模式"
            cluster_patterns = []
            
            for c in range(self.n_clusters):
                # 每个cluster随机选择擅长的产品和工序
                strong_products = torch.randperm(self.m)[:self.m // 2]  # 擅长一半产品
                strong_processes = torch.randperm(self.n)[:self.n // 2]  # 擅长一半工序
                
                cluster_patterns.append({
                    'strong_products': strong_products,
                    'strong_processes': strong_processes
                })
            
            # Step 2: 分配节点到cluster
            cluster_assignment = self.balanced_assignment(self.n_nodes, self.n_clusters)
            
            # Step 3: 根据cluster模式生成节点特征
            node_features = torch.zeros(self.n_nodes, self.m * self.n)
            
            for i in range(self.n_nodes):
                cluster_id = cluster_assignment[i].item()
                pattern = cluster_patterns[cluster_id]
                
                # 生成m×n的能力矩阵
                capability_matrix = torch.rand(self.m, self.n) * 0.3 + 0.1  # 基础能力
                
                # 对擅长的产品和工序，提升能力
                for p in pattern['strong_products']:
                    for proc in pattern['strong_processes']:
                        capability_matrix[p, proc] += torch.rand(1).item() * 0.5 + 0.3
                
                capability_matrix = torch.clamp(capability_matrix, 0.0, 1.0)
                node_features[i] = capability_matrix.reshape(-1)
            
            # Step 4: 生成边标签
            labels = torch.zeros(self.n_nodes, self.n_nodes)
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if cluster_assignment[i] == cluster_assignment[j]:
                        labels[i, j] = 1.0
            
            data.append((node_features, labels))
        
        return data
    
    def balanced_assignment(self, n_nodes, n_clusters):
        """尽量平衡地分配节点到各个cluster"""
        assignment = []
        for i in range(n_nodes):
            assignment.append(i % n_clusters)
        
        # 随机打乱
        indices = torch.randperm(n_nodes)
        assignment = torch.tensor(assignment)[indices]
        
        return assignment
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]



# ----------------------------
# 数据加载与处理
# ----------------------------
def load_dataset(pt_path: str):
    """加载数据集 - 完整实现"""
    try:
        obj = torch.load(pt_path, weights_only=False)
    except TypeError:
        obj = torch.load(pt_path)
    
    # 处理不同的存储格式
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple) and len(obj) == 2:
        # 处理 PyG 的 (data, slices) 格式
        data_proto, slices = obj
        return separate_data_from_slices(data_proto, slices)
    elif isinstance(obj, Data):
        return [obj]
    else:
        raise ValueError(f"不支持的数据格式: {type(obj)}")


def separate_data_from_slices(data_proto: Data, slices: dict) -> list:
    """从 (data, slices) 格式还原数据列表"""
    from torch_geometric.data import Data
    import torch
    
    # 获取数据集大小
    num_graphs = len(slices[list(slices.keys())[0]]) - 1
    
    dataset = []
    for idx in range(num_graphs):
        data = Data()
        
        # 获取所有的键
        if hasattr(data_proto, 'keys'):
            if callable(data_proto.keys):
                keys = list(data_proto.keys())  # 如果是方法，调用它
            else:
                keys = list(data_proto.keys)  # 如果是属性，直接使用
        else:
            # 从 slices 中获取键
            keys = list(slices.keys())
        
        for key in keys:
            if not hasattr(data_proto, key):
                continue
                
            if key in slices:
                # 获取切片索引
                s = slices[key]
                start = s[idx].item() if torch.is_tensor(s[idx]) else s[idx]
                end = s[idx + 1].item() if torch.is_tensor(s[idx + 1]) else s[idx + 1]
                
                # 获取对应的数据切片
                val = getattr(data_proto, key)
                if torch.is_tensor(val):
                    if key == 'edge_index':
                        # edge_index 特殊处理（按列切片）
                        setattr(data, key, val[:, start:end])
                    else:
                        # 其他张量按行切片
                        setattr(data, key, val[start:end])
                else:
                    setattr(data, key, val)
            else:
                # 没有切片信息的属性直接复制
                val = getattr(data_proto, key, None)
                if val is not None:
                    setattr(data, key, val)
        
        dataset.append(data)
    
    print(f"从 (data, slices) 格式加载了 {len(dataset)} 个样本")
    return dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\code\datasets\Seru_datasets\processed\randomT_W2_Exemples\processed\data_randomT_W2_Exemples.pt", help="数据文件路径")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    
    return parser.parse_args()

def get_dataset(args, SizeofDataset):
    dataset0 = load_dataset(args.get('data_dir', r"C:\code\Exemples.pt"))
    dataset = []
    for data in dataset0:
        dataset.append([data.x, data.y_label])
    print(f"数据集大小: {len(dataset)}")

    print("\n" + "="*50)
    print("2. 分割数据集...")
    print("="*50)
    indices = list(range(min(SizeofDataset, len(dataset))))
    # indices = list(range(2000))
    # train_idx, test_idx = train_test_split(
    #     indices, test_size=args.test_size, random_state=0
    # )
    train_idx, test_idx = train_test_split(
        indices, test_size=args.get('test_size', 0.2), shuffle=False
    )
    # train_idx = range(420)

    # test_idx = range(0,420,42)
    train_ds = [dataset[i] for i in train_idx]
    test_ds = [dataset[i] for i in test_idx]

    train_loader = GeoDataLoader(train_ds, batch_size=args.get('batch_size', 32), shuffle=False)
    val_loader = GeoDataLoader(test_ds, batch_size=args.get('batch_size', 32), shuffle=False)
    test_loader = GeoDataLoader(test_ds, batch_size=args.get('batch_size', 32), shuffle=False)

    return train_loader, val_loader, test_loader

def get_dataset_big_config(args, SizeofDataset):
    dataset0 = load_dataset(args.get('data_dir', r"C:\code\Exemples.pt"))
    dataset = []
    for data in dataset0:
        dataset.append([data.x_workers, data.y_worker_label])
    print(f"数据集大小: {len(dataset)}")

    print("\n" + "="*50)
    print("2. 分割数据集...")
    print("="*50)
    indices = list(range(min(SizeofDataset, len(dataset))))
    # indices = list(range(2000))
    # train_idx, test_idx = train_test_split(
    #     indices, test_size=args.test_size, random_state=0
    # )
    train_idx, test_idx = train_test_split(
        indices, test_size=args.get('test_size', 0.2), shuffle=False
    )
    # train_idx = range(420)

    # test_idx = range(0,420,42)
    train_ds = [dataset[i] for i in train_idx]
    test_ds = [dataset[i] for i in test_idx]

    train_loader = GeoDataLoader(train_ds, batch_size=args.get('batch_size', 32), shuffle=False)
    val_loader = GeoDataLoader(test_ds, batch_size=args.get('batch_size', 32), shuffle=False)
    test_loader = GeoDataLoader(test_ds, batch_size=args.get('batch_size', 32), shuffle=False)

    return train_loader, val_loader, test_loader

def get_dataset_big_shedule(args, SizeofDataset):
    dataset0 = load_dataset(args.get('data_dir', r"C:\code\Exemples.pt"))
    dataset = []
    for data in dataset0:
        dataset.append([data.x_batches, data.y_batch_label])
    print(f"数据集大小: {len(dataset)}")

    print("\n" + "="*50)
    print("2. 分割数据集...")
    print("="*50)
    indices = list(range(min(SizeofDataset, len(dataset))))
    # indices = list(range(2000))
    # train_idx, test_idx = train_test_split(
    #     indices, test_size=args.test_size, random_state=0
    # )
    train_idx, test_idx = train_test_split(
        indices, test_size=args.get('test_size', 0.2), shuffle=False
    )
    # train_idx = range(420)

    # test_idx = range(0,420,42)
    train_ds = [dataset[i] for i in train_idx]
    test_ds = [dataset[i] for i in test_idx]

    train_loader = GeoDataLoader(train_ds, batch_size=args.get('batch_size', 32), shuffle=False)
    val_loader = GeoDataLoader(test_ds, batch_size=args.get('batch_size', 32), shuffle=False)
    test_loader = GeoDataLoader(test_ds, batch_size=args.get('batch_size', 32), shuffle=False)

    return train_loader, val_loader, test_loader
def batch_upper_triangular_flatten(batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    将 batch 的 n×n 矩阵转化为上三角非对角线元素的展平向量
    
    参数:
        batch_tensor: 输入 Tensor，形状为 [batchsize, n, n]
    
    返回:
        输出 Tensor，形状为 [batchsize, n*(n-1)//2]
    """
    batch_size, n, _ = batch_tensor.shape  # 确认输入是 [B, n, n]
    
    # 生成上三角非对角线的索引（offset=1 排除主对角线）
    rows, cols = torch.triu_indices(
        row=n,      # 矩阵行数
        col=n,      # 矩阵列数
        offset=1    # 主对角线偏移量：offset=1 表示从主对角线以上开始
    )
    
    # 批量提取元素：每个样本取 (rows[i], cols[i]) 位置的元素
    # 结果形状：[batch_size, len(rows)] = [B, n*(n-1)//2]
    output = batch_tensor[:, rows, cols]
    
    return output
    
if __name__ == '__main__':
    # 测试数据集生成
    args = parse_args()
    train_loader, test_loader = get_dataset(args)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    