"""数据处理模块"""
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


# ==================== 数据生成函数 ====================
def create_linear_data(weight=0.7, bias=0.3, num_samples=1000):
    """
    创建线性回归数据
    
    生成符合 y = weight * x + bias + noise 的数据
    用于测试和演示线性回归模型
    
    参数:
        weight (float): 线性关系的斜率，默认 0.7
        bias (float): 线性关系的截距，默认 0.3
        num_samples (int): 生成的样本数量，默认 1000
    
    返回:
        X (Tensor): 输入特征，形状 (num_samples, 1)
        y (Tensor): 目标值，形状 (num_samples, 1)
    """
    # torch.linspace(start, end, steps): 生成等间距的数值
    # 在 [0, 1] 区间生成 num_samples 个均匀分布的点
    # .unsqueeze(1): 增加一个维度，从 (1000,) 变为 (1000, 1)
    # 这是因为 PyTorch 模型期望输入是 2D 张量 (batch_size, features)
    X = torch.linspace(0, 1, num_samples).unsqueeze(1)
    
    # 根据线性公式计算目标值: y = wx + b
    y = weight * X + bias
    
    # 添加高斯噪声，模拟真实世界数据的随机性
    # torch.randn_like(y): 生成与 y 形状相同的标准正态分布随机数
    # * 0.02: 控制噪声强度（标准差为 0.02）
    y = y + torch.randn_like(y) * 0.02
    
    return X, y


# ==================== 数据加载器准备函数 ====================
def prepare_dataloaders(X, y, train_ratio=0.7, val_ratio=0.15,
                       batch_size=32, num_workers=2):
    """
    准备训练、验证、测试数据加载器
    
    将原始数据按比例分割，并创建 PyTorch DataLoader
    用于批量训练和评估模型
    
    参数:
        X (Tensor): 输入特征
        y (Tensor): 目标值
        train_ratio (float): 训练集比例，默认 0.7 (70%)
        val_ratio (float): 验证集比例，默认 0.15 (15%)
                          测试集比例自动计算为 1 - train_ratio - val_ratio
        batch_size (int): 每个批次的样本数，默认 32
        num_workers (int): 数据加载的并行进程数，默认 2
                          设为 0 表示在主进程中加载（调试时有用）
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # ========== 步骤 1: 创建 Dataset ==========
    # TensorDataset: 将多个张量打包成数据集
    # 每个样本是 (X[i], y[i]) 的元组
    # 支持索引访问: dataset[0] 返回第一个样本
    dataset = TensorDataset(X, y)

    # ========== 步骤 2: 计算分割大小 ==========
    n = len(dataset)  # 总样本数
    train_size = int(n * train_ratio)  # 训练集大小: 1000 * 0.7 = 700
    val_size = int(n * val_ratio)      # 验证集大小: 1000 * 0.15 = 150
    test_size = n - train_size - val_size  # 测试集大小: 1000 - 700 - 150 = 150
    
    # 注意: 使用减法计算 test_size 确保总数正确
    # 避免因 int() 截断导致的样本丢失

    # ========== 步骤 3: 随机分割数据集 ==========
    # random_split: 随机将数据集分割成多个子集
    # 参数: (dataset, [size1, size2, size3])
    # 返回: 三个 Subset 对象，保持对原数据集的引用
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    # ========== 步骤 4: 创建 DataLoader ==========
    # DataLoader: 提供批量数据迭代、打乱、多进程加载等功能
    
    # 训练集 DataLoader
    # shuffle=True: 每个 epoch 开始时打乱数据顺序
    #              这有助于模型泛化，避免学习到数据顺序的模式
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size,
        shuffle=True,           # 训练时打乱数据
        num_workers=num_workers # 多进程加载，加速数据准备
    )
    
    # 验证集 DataLoader
    # shuffle=False: 验证时不需要打乱，保证每次评估结果一致
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size,
        shuffle=False,          # 验证时不打乱
        num_workers=num_workers
    )
    
    # 测试集 DataLoader
    # 与验证集相同，不打乱数据
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size,
        shuffle=False,          # 测试时不打乱
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
