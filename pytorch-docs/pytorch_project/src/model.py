"""
模型定义模块

职责：
- 定义神经网络架构
- 提供模型创建工厂函数
"""
import torch
from torch import nn


# ==================== 模型类定义 ====================
class LinearRegressionModel(nn.Module):
    """
    线性回归模型
    
    实现简单的线性变换: y = Wx + b
    
    参数:
        input_features (int): 输入特征数量，默认 1
        output_features (int): 输出特征数量，默认 1
    
    示例:
        model = LinearRegressionModel(input_features=2, output_features=1)
        output = model(torch.randn(32, 2))  # 输出形状: (32, 1)
    """
    def __init__(self, input_features=1, output_features=1):
        # 必须调用父类的 __init__
        # 这会初始化 nn.Module 的内部状态（参数注册、钩子等）
        super().__init__()
        
        # nn.Linear: 全连接层（线性层）
        # 内部包含:
        #   - weight: 形状 (output_features, input_features) 的权重矩阵
        #   - bias: 形状 (output_features,) 的偏置向量
        # 计算: output = input @ weight.T + bias
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量，形状 (batch_size, input_features)
        
        返回:
            Tensor: 输出张量，形状 (batch_size, output_features)
        """
        return self.linear(x)


# ==================== 模型工厂函数 ====================
def create_model(config, device="cpu"):
    """
    根据配置创建模型
    
    工厂模式：将模型创建逻辑封装，便于统一管理
    
    参数:
        config (dict): 配置字典，需包含 config['model']['input_features'] 等
        device (str): 目标设备，"cpu" 或 "cuda"
    
    返回:
        nn.Module: 已移动到指定设备的模型实例
    """
    model = LinearRegressionModel(
        input_features=config['model']['input_features'],
        output_features=config['model']['output_features']
    )
    # .to(device): 将模型的所有参数移动到指定设备
    # 必须在训练前完成，确保模型和数据在同一设备上
    return model.to(device)
