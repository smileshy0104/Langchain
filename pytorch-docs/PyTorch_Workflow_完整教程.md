# PyTorch 机器学习工作流程 - 完整教程

> **本文档整合了原始教程、最佳实践和2024年最新技术**

---

## 📚 目录

- [第一部分：核心工作流程](#第一部分核心工作流程)
  - [1. 数据准备与加载](#1-数据准备与加载)
  - [2. 构建模型](#2-构建模型)
  - [3. 训练模型](#3-训练模型)
  - [4. 模型评估与预测](#4-模型评估与预测)
  - [5. 保存与加载模型](#5-保存与加载模型)
  - [6. 完整流程整合](#6-完整流程整合)

- [第二部分：高级技术与最佳实践](#第二部分高级技术与最佳实践)
  - [7. 训练循环优化技术](#7-训练循环优化技术)
  - [8. 模型评估指标](#8-模型评估指标)
  - [9. 调试与监控](#9-调试与监控)
  - [10. 生产部署最佳实践](#10-生产部署最佳实践)

- [第三部分：实战项目](#第三部分实战项目)
  - [11. 完整项目示例](#11-完整项目示例)
  - [12. 常见问题与解决方案](#12-常见问题与解决方案)

---

## 文档信息

**来源:** [Learn PyTorch for Deep Learning - Chapter 01](https://www.learnpytorch.io/01_pytorch_workflow/)
**作者:** Daniel Bourke (Zero to Mastery)
**GitHub:** [pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)
**文档版本:** v2.0 (增强版)
**更新日期:** 2025-11-16
**适用 PyTorch 版本:** 1.12+

---

## 🎯 学习目标

完成本教程后,你将能够:

✅ 理解完整的 PyTorch 机器学习工作流程
✅ 构建、训练和评估 PyTorch 模型
✅ 实现高效的训练循环
✅ 应用最佳实践优化模型性能
✅ 保存和部署训练好的模型
✅ 调试和监控训练过程
✅ 处理实际项目中的常见问题

---

# 第一部分：核心工作流程

## PyTorch 工作流程概览

机器学习的本质:**从过去的数据中学习模式,用这些模式预测未来**

![PyTorch Workflow](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/01_a_pytorch_workflow.png)

### 完整工作流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    PyTorch 机器学习工作流程                        │
└─────────────────────────────────────────────────────────────────┘

1. 数据准备 (Data Preparation)
   ├── 收集和加载数据
   ├── 数据清洗和转换
   ├── 划分训练/验证/测试集
   └── 数据可视化
          ↓
2. 构建模型 (Build Model)
   ├── 定义模型架构
   ├── 初始化参数
   └── 设置前向传播
          ↓
3. 训练模型 (Train Model)
   ├── 选择损失函数
   ├── 选择优化器
   ├── 实现训练循环
   └── 实现验证循环
          ↓
4. 评估与预测 (Evaluate & Predict)
   ├── 在测试集上评估
   ├── 计算评估指标
   ├── 可视化结果
   └── 进行推理预测
          ↓
5. 保存与部署 (Save & Deploy)
   ├── 保存模型参数
   ├── 加载模型
   └── 部署到生产环境
```

---

## 1. 数据准备与加载

### 1.1 机器学习的两个核心任务

**任务一:** 将数据转换为数字表示 (数值化)
**任务二:** 构建或选择模型来学习这些数字表示

### 1.2 环境准备

```python
# 导入必要的库
import torch
from torch import nn  # nn = neural networks
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 检查 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")

# 设置随机种子以确保可重复性
torch.manual_seed(42)
```

### 1.3 创建数据集

#### 示例: 线性回归数据

```python
# 定义真实的参数 (我们的目标是让模型学习到这些值)
weight = 0.7
bias = 0.3

# 创建数据: y = wx + b (线性关系)
start = 0
end = 1
step = 0.02

# 创建特征 X
X = torch.arange(start, end, step).unsqueeze(dim=1)  # shape: [50, 1]

# 创建标签 y
y = weight * X + bias  # shape: [50, 1]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Number of samples: {len(X)}")
print(f"\nFirst 5 X values:\n{X[:5]}")
print(f"\nFirst 5 y values:\n{y[:5]}")
```

**输出示例:**
```bash
X shape: torch.Size([50, 1])
y shape: torch.Size([50, 1])
Number of samples: 50

First 5 X values:
tensor([[0.0000],
        [0.0200],
        [0.0400],
        [0.0600],
        [0.0800]])

First 5 y values:
tensor([[0.3000],
        [0.3140],
        [0.3280],
        [0.3420],
        [0.3560]])
```

### 1.4 数据集划分

#### 训练集/验证集/测试集的作用

| 数据集 | 用途 | 建议比例 | 使用频率 |
|--------|------|---------|---------|
| **训练集 (Training)** | 模型从中学习模式 | 60-80% | 必须 |
| **验证集 (Validation)** | 调整超参数,选择最佳模型 | 10-20% | 推荐 |
| **测试集 (Testing)** | 最终评估模型性能 | 10-20% | 必须 |

#### 实现数据划分

```python
# 80% 训练, 20% 测试
train_split = int(0.8 * len(X))

# 划分数据
X_train = X[:train_split]  # 前 80%
y_train = y[:train_split]

X_test = X[train_split:]   # 后 20%
y_test = y[train_split:]

print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
```

#### 更完整的数据划分<——>三分法 (推荐用于大型项目)

```python
def split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    将数据集划分为训练集、验证集和测试集

    参数:
        X: 特征数据
        y: 标签数据
        train_ratio: 训练集比例 (默认 0.7, 即 70%)
        val_ratio: 验证集比例 (默认 0.15, 即 15%)
        test_ratio: 测试集比例 (默认 0.15, 即 15%)

    返回:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # 验证三个比例之和是否等于 1.0
    # 使用 1e-6 作为容差值来处理浮点数精度问题
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须等于 1"

    # 获取数据集的总样本数
    n = len(X)
    
    # 计算训练集的结束索引 (例如: 1000 * 0.7 = 700)
    train_end = int(n * train_ratio)
    
    # 计算验证集的结束索引 (例如: 1000 * (0.7 + 0.15) = 850)
    val_end = int(n * (train_ratio + val_ratio))

    # 划分训练集: 从开始到 train_end
    X_train, y_train = X[:train_end], y[:train_end]
    
    # 划分验证集: 从 train_end 到 val_end
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    
    # 划分测试集: 从 val_end 到结束
    X_test, y_test = X[val_end:], y[val_end:]

    # 返回划分后的六个数据集
    return X_train, y_train, X_val, y_val, X_test, y_test


# 使用示例: 调用函数进行数据划分
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

# 打印各数据集的样本数量,验证划分结果
print(f"训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
```

### 1.5 数据可视化（非常重要）

> **数据探索者的座右铭:** "可视化,可视化,可视化!"

```python
# 导入 matplotlib 并设置中文字体 解决中文乱码问题
import matplotlib.pyplot as plt

# 设置默认字体为黑体,用于正确显示中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体

# 解决坐标轴负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    绘制训练数据、测试数据和预测结果

    参数:
        train_data: 训练特征
        train_labels: 训练标签
        test_data: 测试特征
        test_labels: 测试标签
        predictions: 模型预测 (可选,默认为 None)
    """
    # 创建图形,设置画布大小为 10x7 英寸
    plt.figure(figsize=(10, 7))

    # 绘制训练数据散点图
    # c="b": 蓝色, s=4: 点的大小, label: 图例标签
    plt.scatter(train_data, train_labels, c="b", s=4, label="训练数据")

    # 绘制测试数据散点图 (绿色)
    plt.scatter(test_data, test_labels, c="g", s=4, label="测试数据")

    # 如果提供了预测结果,则绘制预测散点图
    if predictions is not None:
        # 绘制预测结果 (红色),用于对比真实测试数据
        plt.scatter(test_data, predictions, c="r", s=4, label="预测")

    # 添加图例,设置字体大小为 14
    plt.legend(prop={"size": 14})
    
    # 设置 X 轴标签
    plt.xlabel("X")
    
    # 设置 Y 轴标签
    plt.ylabel("y")
    
    # 设置图表标题
    plt.title("数据和预测可视化")
    
    # 添加网格线,alpha=0.3 设置透明度使网格不会过于突出
    plt.grid(True, alpha=0.3)


# 调用函数可视化原始数据 (不包含预测结果)
plot_predictions()

# 显示图形
plt.show()
```

### 1.6 数据加载最佳实践

#### 使用 DataLoader 处理大型数据集

```python
# 导入 PyTorch 数据加载工具
from torch.utils.data import TensorDataset, DataLoader

# 创建 Dataset 对象,将特征和标签封装在一起
# TensorDataset 会自动将 X 和 y 配对,方便批量加载
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 设置批次大小 (每次训练使用的样本数量)
BATCH_SIZE = 8

# 创建训练数据加载器
train_loader = DataLoader(
    train_dataset,              # 要加载的数据集
    batch_size=BATCH_SIZE,      # 每个批次的样本数量
    shuffle=True,               # 每个 epoch 开始时打乱数据,避免模型记住数据顺序
    num_workers=2,              # 使用 2 个子进程并行加载数据,加快速度
    pin_memory=True             # 将数据固定在内存中,加快 CPU 到 GPU 的传输速度
)

# 创建测试数据加载器
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False               # 测试时不打乱数据,保持评估的一致性
)

# 查看一个 batch 的数据形状
# 使用 break 只查看第一个批次
for batch_X, batch_y in train_loader:
    print(f"Batch X shape: {batch_X.shape}")  # 输出: torch.Size([8, 1])
    print(f"Batch y shape: {batch_y.shape}")  # 输出: torch.Size([8, 1])
    break  # 只查看第一个批次后退出循环
```

---

## 2. 构建模型

### 2.1 PyTorch 模型构建核心组件

| PyTorch 模块 | 功能说明 |
|-------------|---------|
| `torch.nn` | 包含构建神经网络的所有模块 |
| `torch.nn.Parameter` | 可训练的参数,会自动计算梯度 |
| `torch.nn.Module` | 所有神经网络的基类，神经网络的所有构建块都是子类。 |
| `torch.optim` | 包含各种优化算法 （这些算法告诉存储在如何最好地改变以改善梯度下降，进而减少损失）中存储的模型参数。|
| `def forward()` | 定义前向传播的计算过程 |

### 2.2 线性回归模型 (手动参数版本)

```python
import torch
from torch import nn

class LinearRegressionModel(nn.Module):
    """
    简单的线性回归模型: y = wx + b
    
    这是手动定义参数的版本，用于理解 PyTorch 模型的基本构建方式
    继承自 nn.Module，这是所有 PyTorch 神经网络的基类
    """
    def __init__(self):
        # 调用父类的初始化方法，这是必须的
        # 它会设置模型的基础功能（如参数追踪、设备管理等）
        super().__init__()

        # 初始化权重参数 (斜率 w)
        # nn.Parameter: 将张量注册为模型的可学习参数
        # torch.randn(1): 从标准正态分布中随机初始化一个值
        # requires_grad=True: 告诉 PyTorch 在反向传播时计算此参数的梯度
        self.weight = nn.Parameter(
            torch.randn(1, dtype=torch.float),  # 形状为 [1]，单个浮点数
            requires_grad=True  # 启用梯度计算，使其可以通过优化器更新
        )

        # 初始化偏置参数 (截距 b)
        # 同样使用 nn.Parameter 包装，使其成为可学习参数
        self.bias = nn.Parameter(
            torch.randn(1, dtype=torch.float),  # 形状为 [1]，单个浮点数
            requires_grad=True  # 启用梯度计算
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 定义模型如何从输入计算输出
        
        这个方法在调用 model(x) 时会被自动执行
        实现线性方程: y = weight * x + bias

        参数:
            x: 输入特征张量，形状可以是 [batch_size, 1] 或 [n_samples]

        返回:
            预测值张量，与输入形状相同
        """
        # 执行线性变换: y = wx + b
        # PyTorch 会自动进行广播 (broadcasting)
        return self.weight * x + self.bias

# ==================== 使用模型 ====================

# 设置随机种子，确保每次运行结果一致（可重复性）
torch.manual_seed(42)

# 创建模型实例
model_0 = LinearRegressionModel()

# 打印模型结构，显示模型的层次和参数
print(f"模型结构:\n{model_0}")

# 打印初始参数值（随机初始化的）
print(f"\n初始参数:")
print(f"  权重 (weight): {model_0.weight}")
print(f"  偏置 (bias): {model_0.bias}")

# 获取所有可学习参数的列表
# 返回一个迭代器，包含所有 requires_grad=True 的参数
list(model_0.parameters())

# 获取模型的状态字典
# 返回一个字典，键是参数名，值是参数张量
# 常用于保存和加载模型
model_0.state_dict()
```

### 2.3 线性回归模型 (使用 nn.Linear)

```python
class LinearRegressionModelV2(nn.Module):
    """
    使用 nn.Linear 的线性回归模型 (推荐方式)
    
    nn.Linear 是 PyTorch 内置的线性层，会自动处理权重和偏置的初始化
    这是实际开发中的标准做法，比手动定义参数更简洁且更不容易出错
    """
    def __init__(self):
        # 调用父类初始化
        super().__init__()

        # 创建线性层
        # nn.Linear 会自动创建并初始化 weight 和 bias 参数
        # 内部实现: y = x @ weight.T + bias
        self.linear_layer = nn.Linear(
            in_features=1,   # 输入特征的维度（每个样本有 1 个特征）
            out_features=1   # 输出特征的维度（预测 1 个值）
        )
        # 注意: nn.Linear 的 weight 形状是 [out_features, in_features]
        # 这里是 [1, 1]，bias 形状是 [out_features]，这里是 [1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 将输入传递给线性层
        
        参数:
            x: 输入特征张量
            
        返回:
            线性层的输出（预测值）
        """
        # 直接调用线性层进行计算
        # 等价于: weight * x + bias
        return self.linear_layer(x)

# ==================== 使用模型 ====================

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 创建模型实例
model_1 = LinearRegressionModelV2()

# 打印模型结构
# 会显示 LinearRegressionModelV2 包含一个 Linear 层
print(f"模型结构:\n{model_1}")

# 打印线性层的参数信息
print(f"\nLinear 层参数:")
# named_parameters() 返回参数名称和参数张量的迭代器
for name, param in model_1.named_parameters():
    print(f"  {name}: {param.shape}")
    # 输出示例:
    # linear_layer.weight: torch.Size([1, 1])
    # linear_layer.bias: torch.Size([1])
```

### 2.4 查看模型信息

```python
def model_info(model):
    """
    打印模型的详细信息
    
    这个工具函数用于检查模型的结构、参数和状态
    在调试和理解模型时非常有用
    
    参数:
        model: PyTorch 模型实例 (nn.Module)
    """
    print("=" * 70)
    print("模型信息")
    print("=" * 70)

    # ==================== 1. 模型结构 ====================
    # 打印模型的层次结构和组成
    # 会显示模型类名和所有子模块
    print(f"\n模型结构:\n{model}\n")

    # ==================== 2. 参数详情 ====================
    print("参数详情:")
    total_params = 0  # 用于累计总参数数量
    
    # named_parameters() 返回 (参数名, 参数张量) 的迭代器
    # 只包含 requires_grad=True 的参数
    for name, param in model.named_parameters():
        print(f"  {name}:")  # 参数名称，如 'linear_layer.weight'
        
        # param.shape: 参数张量的形状，如 torch.Size([1, 1])
        print(f"    形状: {param.shape}")
        
        # param.numel(): 参数中元素的总数 (number of elements)
        # 例如 [3, 4] 形状的张量有 12 个元素
        print(f"    数量: {param.numel()}")
        
        # param.requires_grad: 是否需要计算梯度
        # True 表示这个参数会在训练中被更新
        print(f"    需要梯度: {param.requires_grad}")
        
        # param.data: 参数的实际数值（不带梯度信息）
        # 直接访问张量的值，不会构建计算图
        print(f"    当前值: {param.data}\n")
        
        # 累加参数数量
        total_params += param.numel()

    # 打印总参数量
    # 对于复杂模型，这个数字可能达到数百万甚至数十亿
    print(f"总参数量: {total_params}")

    # ==================== 3. 状态字典 ====================
    # state_dict() 返回模型所有参数的字典
    # 键是参数名（字符串），值是参数张量
    # 这是保存和加载模型的标准方式
    print(f"\n状态字典:\n{model.state_dict()}")
    # 输出示例: OrderedDict([('linear_layer.weight', tensor([[...]]), 
    #                         ('linear_layer.bias', tensor([...]))])

    print("=" * 70)

# ==================== 使用示例 ====================
# 调用函数查看 model_1 的详细信息
# 这会显示 LinearRegressionModelV2 的所有参数和结构
model_info(model_1)
```

### 2.5 使用未训练模型进行预测

```python
# ==================== 使用未训练模型进行预测 ====================
# 目的: 查看随机初始化的模型预测效果（作为基线对比）

# 将模型设置为评估模式
# eval() 会关闭某些训练时才需要的功能（如 Dropout、BatchNorm）
# 虽然这个简单模型没有这些层，但养成好习惯很重要
model_1.eval()

# 使用推理模式进行预测
# torch.inference_mode() 是推荐的推理上下文管理器
# 作用:
#   1. 禁用梯度计算，节省内存和计算资源
#   2. 比 torch.no_grad() 更快，因为它完全禁用了自动求导引擎
#   3. 适用于不需要反向传播的场景（如预测、验证）
#   4. 以加快前向传播 （数据通过 forward() 方法）的速度
with torch.inference_mode():
    # 将测试数据传入模型进行预测
    # model_1(X_test) 会自动调用 forward() 方法
    y_preds = model_1(X_test)

# 打印前 5 个预测值
# 由于模型未训练，参数是随机的，预测结果会很差
print(f"预测值 (前5个):\n{y_preds[:5]}")

# 打印前 5 个真实值，用于对比
print(f"\n真实值 (前5个):\n{y_test[:5]}")

# ==================== 可视化预测结果 ====================
# 调用之前定义的绘图函数，可视化预测效果
# 未训练的模型预测应该是一条随机的直线，与真实数据相差很远
plot_predictions(predictions=y_preds)
plt.title("未训练模型的预测 (应该很差)")
plt.show()

# 注意: 这个可视化展示了训练的必要性
# 通过对比训练前后的预测，可以直观看到模型学习的效果
```

### 2.6 模型设备管理

```python
# ==================== 设备检测 ====================
# 检查是否有 NVIDIA GPU 可用
# torch.cuda.is_available() 返回 True 表示系统有可用的 CUDA GPU
# GPU 可以大幅加速深度学习训练（通常快 10-100 倍）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 其他设备选项:
# - "mps": Apple Silicon (M1/M2) 的 Metal Performance Shaders
# - "cpu": CPU（所有系统都支持，但速度较慢）

# ==================== 将模型移动到设备 ====================
# .to(device) 方法将模型的所有参数和缓冲区移动到指定设备
# 这是一个就地操作（in-place），但通常习惯重新赋值
# 注意: 模型和数据必须在同一设备上才能进行计算
model_1 = model_1.to(device)

# ==================== 将数据移动到设备 ====================
# 同样需要将训练和测试数据移动到相同设备
# 如果模型在 GPU 上，数据也必须在 GPU 上
# .to() 会创建数据的副本并移动到目标设备

# 移动训练数据
X_train = X_train.to(device)
y_train = y_train.to(device)

# 移动测试数据
X_test = X_test.to(device)
y_test = y_test.to(device)

# ==================== 验证设备位置 ====================
# 检查模型参数所在的设备
# next(model_1.parameters()) 获取模型的第一个参数
# .device 属性显示张量所在的设备
print(f"模型设备: {next(model_1.parameters()).device}")

# 检查数据所在的设备
# 确保模型和数据在同一设备上，否则会报错
print(f"数据设备: {X_train.device}")

# 常见错误: RuntimeError: Expected all tensors to be on the same device
# 解决方法: 确保模型和所有输入数据都在同一设备上
```

---

## 3. 训练模型

### 3.1 损失函数

**损失函数 (Loss Function):** 衡量模型预测与真实值之间的差距

#### 常用损失函数

| 损失函数 | PyTorch 实现 | 适用场景 |
|---------|-------------|---------|
| **均方误差 (MSE)** | `nn.MSELoss()` | 回归问题 (对异常值敏感) |
| **平均绝对误差 (MAE)** | `nn.L1Loss()` | 回归问题 (对异常值鲁棒) |
| **交叉熵损失** | `nn.CrossEntropyLoss()` | 多分类问题 |
| **二元交叉熵** | `nn.BCELoss()` | 二分类问题 |
| **二元交叉熵 (带 logits)** | `nn.BCEWithLogitsLoss()` | 二分类 (更稳定) |

```python
# 创建损失函数
loss_fn = nn.L1Loss()  # MAE for regression

# 手动计算损失示例
with torch.inference_mode():
    y_pred = model_1(X_train)
    loss = loss_fn(y_pred, y_train)
    print(f"初始损失: {loss}")
```

### 3.2 优化器

**优化器 (Optimizer):** 使用梯度来更新模型参数

#### 常用优化器

| 优化器 | PyTorch 实现 | 特点 | 适用场景 |
|-------|-------------|-----|---------|
| **SGD** | `torch.optim.SGD()` | 最基础,稳定 | 简单任务 |
| **Adam** | `torch.optim.Adam()` | 自适应学习率,最流行 | 大多数任务 (2024推荐) |
| **AdamW** | `torch.optim.AdamW()` | Adam + 权重衰减 | NLP, Transformers (2024推荐) |
| **RMSprop** | `torch.optim.RMSprop()` | 适应学习率 | RNN |
| **Adagrad** | `torch.optim.Adagrad()` | 稀疏梯度 | 稀疏数据 |

```python
# 创建优化器
optimizer = torch.optim.SGD(
    params=model_1.parameters(),  # 要优化的参数
    lr=0.01  # 学习率 (learning rate) - 重要的超参数
)

# 查看优化器信息
print(f"优化器: {optimizer}")
print(f"学习率: {optimizer.param_groups[0]['lr']}")
```

### 3.3 训练循环 (核心)

#### 训练循环的标准步骤

```
训练阶段:
1. 前向传播 (Forward pass) - 计算预测
2. 计算损失 (Calculate loss) - 衡量预测质量
3. 清零梯度 (Zero gradients) - 清除上一步的梯度
4. 反向传播 (Backpropagation) - 计算梯度
5. 更新参数 (Optimizer step) - 使用梯度更新参数

评估阶段:
1. 前向传播 (不计算梯度)
2. 计算损失
```

#### 基础训练循环实现

```python
# 训练设置
epochs = 100

# 用于记录
epoch_count = []
train_loss_values = []
test_loss_values = []

print("开始训练...")
print(f"{'Epoch':<6} {'训练损失':<12} {'测试损失':<12}")
print("-" * 35)

for epoch in range(epochs):
    ### 训练模式 ###
    model_1.train()  # 设置为训练模式

    # 1. 前向传播
    y_pred = model_1(X_train)

    # 2. 计算损失
    loss = loss_fn(y_pred, y_train)

    # 3. 清零梯度
    optimizer.zero_grad()

    # 4. 反向传播 (计算梯度)
    loss.backward()

    # 5. 更新参数
    optimizer.step()

    ### 评估模式 ###
    model_1.eval()  # 设置为评估模式

    with torch.inference_mode():
        # 1. 前向传播
        test_pred = model_1(X_test)

        # 2. 计算测试损失
        test_loss = loss_fn(test_pred, y_test)

    # 记录损失
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.item())
        test_loss_values.append(test_loss.item())

        print(f"{epoch:<6} {loss.item():<12.4f} {test_loss.item():<12.4f}")

print("\n训练完成!")
```

### 3.4 可视化训练过程

```python
def plot_loss_curves(epoch_count, train_loss, test_loss):
    """绘制训练和测试损失曲线"""
    plt.figure(figsize=(10, 7))

    plt.plot(epoch_count, train_loss, label="训练损失", color="blue")
    plt.plot(epoch_count, test_loss, label="测试损失", color="orange")

    plt.title("训练和测试损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 绘制损失曲线
plot_loss_curves(epoch_count, train_loss_values, test_loss_values)
```

### 3.5 完整的训练函数 (生产级)

```python
def train_model(model,
                train_loader,
                test_loader,
                loss_fn,
                optimizer,
                epochs,
                device="cpu",
                print_every=10):
    """
    训练 PyTorch 模型的完整函数

    参数:
        model: PyTorch 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        loss_fn: 损失函数
        optimizer: 优化器
        epochs: 训练轮数
        device: 训练设备 (cpu/cuda)
        print_every: 打印频率

    返回:
        results: 包含损失历史的字典
    """
    # 移动模型到设备
    model = model.to(device)

    # 记录历史
    results = {
        "train_loss": [],
        "test_loss": [],
        "epoch": []
    }

    for epoch in range(epochs):
        ### 训练阶段 ###
        model.train()
        train_loss = 0

        for batch, (X, y) in enumerate(train_loader):
            # 移动数据到设备
            X, y = X.to(device), y.to(device)

            # 1. 前向传播
            y_pred = model(X)

            # 2. 计算损失
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. 清零梯度
            optimizer.zero_grad()

            # 4. 反向传播
            loss.backward()

            # 5. 更新参数
            optimizer.step()

        # 计算平均训练损失
        train_loss /= len(train_loader)

        ### 测试阶段 ###
        model.eval()
        test_loss = 0

        with torch.inference_mode():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)

                # 前向传播
                test_pred = model(X)

                # 计算损失
                loss = loss_fn(test_pred, y)
                test_loss += loss.item()

        # 计算平均测试损失
        test_loss /= len(test_loader)

        # 记录结果
        if epoch % print_every == 0:
            results["train_loss"].append(train_loss)
            results["test_loss"].append(test_loss)
            results["epoch"].append(epoch)

            print(f"Epoch: {epoch} | "
                  f"Train loss: {train_loss:.5f} | "
                  f"Test loss: {test_loss:.5f}")

    return results
```

---

## 4. 模型评估与预测

### 4.1 推理模式 (Inference Mode)

```python
# 方法 1: torch.inference_mode() (推荐,更快)
with torch.inference_mode():
    y_preds = model_1(X_test)

# 方法 2: torch.no_grad() (旧版,仍然支持)
with torch.no_grad():
    y_preds = model_1(X_test)
```

**为什么使用推理模式?**
- ✅ 关闭梯度追踪 (节省内存)
- ✅ 加快前向传播速度
- ✅ `torch.inference_mode()` 比 `torch.no_grad()` 更快

### 4.2 评估模型性能

```python
# 设置为评估模式
model_1.eval()

with torch.inference_mode():
    # 进行预测
    y_preds = model_1(X_test)

    # 计算最终测试损失
    final_loss = loss_fn(y_preds, y_test)
    print(f"最终测试损失: {final_loss:.4f}")

# 查看学到的参数
print(f"\n学到的参数:")
print(f"  权重: {model_1.state_dict()['linear_layer.weight'].item():.4f}")
print(f"  偏置: {model_1.state_dict()['linear_layer.bias'].item():.4f}")

print(f"\n真实参数:")
print(f"  权重: {weight}")
print(f"  偏置: {bias}")

# 可视化预测结果
plot_predictions(predictions=y_preds)
plt.title("训练后的预测结果")
plt.show()
```

---

## 5. 保存与加载模型

### 5.1 PyTorch 模型保存方法

PyTorch 提供三种主要方法:

1. **`torch.save()`** - 保存对象到磁盘
2. **`torch.load()`** - 从磁盘加载对象
3. **`model.load_state_dict()`** - 加载模型参数字典

### 5.2 保存和加载 state_dict (推荐)

```python
from pathlib import Path

# 1. 创建模型目录
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. 定义保存路径
MODEL_NAME = "01_pytorch_workflow_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. 保存模型 state_dict
print(f"保存模型到: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),  # 只保存参数
           f=MODEL_SAVE_PATH)

# 4. 加载模型
# 首先创建模型实例
loaded_model = LinearRegressionModelV2()

# 加载保存的参数
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# 5. 设置为评估模式
loaded_model.eval()

# 6. 验证加载的模型
with torch.inference_mode():
    loaded_preds = loaded_model(X_test)

# 检查预测是否一致
print(f"原始模型预测: {y_preds[:3]}")
print(f"加载模型预测: {loaded_preds[:3]}")
print(f"预测是否相同: {torch.allclose(y_preds, loaded_preds)}")
```

### 5.3 保存完整模型 (不推荐但有时有用)

```python
# 保存完整模型
FULL_MODEL_PATH = MODEL_PATH / "full_model.pth"
torch.save(model_1, FULL_MODEL_PATH)

# 加载完整模型
loaded_full_model = torch.load(FULL_MODEL_PATH)
loaded_full_model.eval()
```

### 5.4 保存和加载检查点 (Checkpoint)

```python
def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"检查点已保存到 {filepath}")

def load_checkpoint(filepath, model, optimizer):
    """加载训练检查点"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss

# 使用示例
# 保存
save_checkpoint(model_1, optimizer, epoch=100, loss=final_loss,
                filepath=MODEL_PATH / "checkpoint.pth")

# 加载
model_1, optimizer, start_epoch, loss = load_checkpoint(
    MODEL_PATH / "checkpoint.pth", model_1, optimizer
)
```

### 5.5 最佳实践总结

| 场景 | 推荐方法 | 原因 |
|-----|---------|-----|
| **生产部署** | `state_dict()` | 更灵活,可移植性好 |
| **中断训练** | Checkpoint | 可恢复训练状态 |
| **快速原型** | 完整模型 | 简单快速 |
| **跨版本** | `state_dict()` | 避免 PyTorch 版本问题 |

---

## 6. 完整流程整合

### 6.1 端到端示例代码

```python
"""
完整的 PyTorch 线性回归工作流程
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# ============ 1. 准备数据 ============
print("1. 准备数据...")
weight = 0.7
bias = 0.3

X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(f"   训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

# ============ 2. 构建模型 ============
print("\n2. 构建模型...")

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear_layer(x)

torch.manual_seed(42)
model = LinearRegressionModelV2()
print(f"   模型: {model}")

# ============ 3. 设置损失和优化器 ============
print("\n3. 设置损失函数和优化器...")
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
print(f"   损失函数: {loss_fn}")
print(f"   优化器: {optimizer.__class__.__name__}")

# ============ 4. 训练模型 ============
print("\n4. 开始训练...")
epochs = 200

for epoch in range(epochs):
    # 训练模式
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 评估模式
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 50 == 0:
        print(f"   Epoch {epoch}: Train Loss = {loss:.4f}, Test Loss = {test_loss:.4f}")

# ============ 5. 评估模型 ============
print("\n5. 评估模型...")
model.eval()
with torch.inference_mode():
    y_preds = model(X_test)
    final_loss = loss_fn(y_preds, y_test)

print(f"   最终测试损失: {final_loss:.4f}")
print(f"   学到的权重: {model.state_dict()['linear_layer.weight'].item():.4f}")
print(f"   学到的偏置: {model.state_dict()['linear_layer.bias'].item():.4f}")

# ============ 6. 保存模型 ============
print("\n6. 保存模型...")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)
SAVE_PATH = MODEL_PATH / "linear_model.pth"
torch.save(model.state_dict(), SAVE_PATH)
print(f"   模型已保存到: {SAVE_PATH}")

print("\n✅ 完整流程执行完毕!")
```

---

# 第二部分：高级技术与最佳实践

## 7. 训练循环优化技术

### 7.1 自动混合精度 (AMP)

**优势:**
- ✅ 训练速度提升 1.5-3倍
- ✅ GPU 内存减半
- ✅ 保持相同的准确率

```python
from torch.cuda.amp import autocast, GradScaler

# 创建梯度缩放器
scaler = GradScaler()

for epoch in range(epochs):
    model.train()

    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # 使用混合精度
        with autocast():
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        optimizer.zero_grad()

        # 缩放损失并反向传播
        scaler.scale(loss).backward()

        # 更新参数
        scaler.step(optimizer)
        scaler.update()
```

### 7.2 梯度累积 (Gradient Accumulation)

**用途:** 模拟更大的 batch size (当 GPU 内存不足时)

```python
# 模拟 batch size = 32 (实际 batch size = 8)
ACCUMULATION_STEPS = 4

optimizer.zero_grad()

for i, (X, y) in enumerate(train_loader):
    X, y = X.to(device), y.to(device)

    # 前向传播
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    # 归一化损失
    loss = loss / ACCUMULATION_STEPS

    # 反向传播
    loss.backward()

    # 每 ACCUMULATION_STEPS 更新一次参数
    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 7.3 学习率调度器 (Learning Rate Scheduler)

```python
from torch.optim.lr_scheduler import (
    StepLR,
    ReduceLROnPlateau,
    CosineAnnealingLR
)

# 方法 1: StepLR - 每 N 个 epoch 降低学习率
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 方法 2: ReduceLROnPlateau - 当指标停止改善时降低学习率
scheduler = ReduceLROnPlateau(optimizer, mode='min',
                             factor=0.1, patience=10)

# 方法 3: CosineAnnealingLR - 余弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# 在训练循环中使用
for epoch in range(epochs):
    # ... 训练代码 ...

    # StepLR / CosineAnnealingLR
    scheduler.step()

    # ReduceLROnPlateau (需要监控指标)
    # scheduler.step(val_loss)

    # 打印当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}, LR: {current_lr}")
```

### 7.4 梯度裁剪 (Gradient Clipping)

**用途:** 防止梯度爆炸

```python
import torch.nn.utils as nn_utils

MAX_GRAD_NORM = 1.0

for epoch in range(epochs):
    for X, y in train_loader:
        optimizer.zero_grad()

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()

        # 梯度裁剪
        nn_utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()
```

### 7.5 早停 (Early Stopping)

```python
class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0):
        """
        参数:
            patience: 容忍多少个 epoch 没有改善
            min_delta: 最小改善量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 使用示例
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

for epoch in range(epochs):
    # ... 训练和验证 ...

    early_stopping(val_loss)

    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break
```

---

## 8. 模型评估指标

### 8.1 回归问题指标

```python
def regression_metrics(y_true, y_pred):
    """计算回归问题的常用指标"""
    # 转换为 numpy 数组
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))

    # MSE (Mean Squared Error)
    mse = np.mean((y_true - y_pred) ** 2)

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)

    # R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

# 使用示例
model.eval()
with torch.inference_mode():
    y_pred = model(X_test)

metrics = regression_metrics(y_test, y_pred)
print("模型评估指标:")
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
```

### 8.2 使用 torchmetrics 库

```python
# 安装: pip install torchmetrics

from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

# 创建指标
mae_metric = MeanAbsoluteError()
mse_metric = MeanSquaredError()
r2_metric = R2Score()

# 在训练循环中更新
for X, y in test_loader:
    with torch.inference_mode():
        y_pred = model(X)

    # 更新指标
    mae_metric.update(y_pred, y)
    mse_metric.update(y_pred, y)
    r2_metric.update(y_pred, y)

# 计算最终值
print(f"MAE: {mae_metric.compute():.4f}")
print(f"MSE: {mse_metric.compute():.4f}")
print(f"R²: {r2_metric.compute():.4f}")
```

---

## 9. 调试与监控

### 9.1 PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True) as prof:
    for _ in range(10):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()

# 打印统计信息
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

### 9.2 使用 TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

# 创建 writer
writer = SummaryWriter('runs/linear_regression')

for epoch in range(epochs):
    # ... 训练 ...

    # 记录损失
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)

    # 记录学习率
    writer.add_scalar('Learning Rate',
                     optimizer.param_groups[0]['lr'], epoch)

    # 记录参数直方图
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

writer.close()

# 在终端运行: tensorboard --logdir=runs
```

### 9.3 使用 Weights & Biases (推荐)

```python
# 安装: pip install wandb

import wandb

# 初始化
wandb.init(project="pytorch-linear-regression",
          config={
              "learning_rate": 0.01,
              "epochs": 100,
              "batch_size": 8
          })

# 训练循环
for epoch in range(epochs):
    # ... 训练 ...

    # 记录指标
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

# 保存模型到 W&B
wandb.save("model.pth")
```

---

## 10. 生产部署最佳实践

### 10.1 模型导出为 ONNX

```python
# 导出模型为 ONNX 格式 (跨平台推理)
dummy_input = torch.randn(1, 1)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)

print("模型已导出为 ONNX 格式")
```

### 10.2 模型量化 (加速推理)

```python
# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # 要量化的层类型
    dtype=torch.qint8
)

# 测试量化模型
with torch.inference_mode():
    quantized_pred = quantized_model(X_test)

print(f"原始模型大小: {os.path.getsize('model.pth') / 1024:.2f} KB")
torch.save(quantized_model.state_dict(), 'quantized_model.pth')
print(f"量化模型大小: {os.path.getsize('quantized_model.pth') / 1024:.2f} KB")
```

### 10.3 模型服务化 (使用 FastAPI)

```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# 加载模型
model = LinearRegressionModelV2()
model.load_state_dict(torch.load('model.pth'))
model.eval()

class PredictionRequest(BaseModel):
    value: float

class PredictionResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    with torch.inference_mode():
        X = torch.tensor([[request.value]])
        pred = model(X)

    return PredictionResponse(prediction=pred.item())

# 运行: uvicorn api:app --reload
```

---

# 第三部分：实战项目

## 11. 完整项目示例

### 11.1 项目结构

```
pytorch_project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── checkpoints/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py        # 数据处理
│   ├── model.py       # 模型定义
│   ├── train.py       # 训练逻辑
│   ├── evaluate.py    # 评估逻辑
│   └── utils.py       # 工具函数
├── config.yaml        # 配置文件
├── requirements.txt   # 依赖
└── main.py           # 主入口
```

### 11.2 配置文件 (config.yaml)

```yaml
# 数据配置
data:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  batch_size: 32

# 模型配置
model:
  input_features: 1
  output_features: 1

# 训练配置
training:
  epochs: 100
  learning_rate: 0.01
  optimizer: "Adam"
  loss_function: "MSE"

# 设备配置
device: "cuda"  # or "cpu"

# 保存配置
save:
  model_dir: "models"
  checkpoint_dir: "models/checkpoints"
```

### 11.3 完整代码实现

#### src/data.py

```python
"""数据处理模块"""
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def create_linear_data(weight=0.7, bias=0.3, num_samples=1000):
    """创建线性数据"""
    X = torch.linspace(0, 1, num_samples).unsqueeze(1)
    y = weight * X + bias
    # 添加一些噪声
    y = y + torch.randn_like(y) * 0.02
    return X, y

def prepare_dataloaders(X, y, train_ratio=0.7, val_ratio=0.15,
                       batch_size=32, num_workers=2):
    """准备数据加载器"""
    # 创建 dataset
    dataset = TensorDataset(X, y)

    # 计算分割大小
    n = len(dataset)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    test_size = n - train_size - val_size

    # 分割数据集
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    # 创建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
```

#### src/model.py

```python
"""模型定义模块"""
import torch
from torch import nn

class LinearRegressionModel(nn.Module):
    """线性回归模型"""
    def __init__(self, input_features=1, output_features=1):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        return self.linear(x)

def create_model(config, device="cpu"):
    """根据配置创建模型"""
    model = LinearRegressionModel(
        input_features=config['model']['input_features'],
        output_features=config['model']['output_features']
    )
    return model.to(device)
```

#### src/train.py

```python
"""训练模块"""
import torch
from tqdm.auto import tqdm

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # 前向传播
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device):
    """验证函数"""
    model.eval()
    total_loss = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def train(model, train_loader, val_loader, config, device):
    """完整训练流程"""
    # 设置损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # 训练循环
    epochs = config['training']['epochs']
    history = {'train_loss': [], 'val_loss': []}

    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss = train_epoch(model, train_loader, loss_fn,
                                optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}")

    return history
```

#### src/evaluate.py

```python
"""评估模块"""
import torch
import numpy as np

def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y_pred = model(X)

            all_preds.append(y_pred.cpu())
            all_targets.append(y)

    # 合并所有批次
    predictions = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # 计算指标
    mae = torch.mean(torch.abs(predictions - targets))
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)

    return {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'RMSE': rmse.item()
    }
```

#### main.py

```python
"""主程序入口"""
import torch
import yaml
from pathlib import Path

from src.data import create_linear_data, prepare_dataloaders
from src.model import create_model
from src.train import train
from src.evaluate import evaluate_model

def main():
    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建数据
    print("Creating data...")
    X, y = create_linear_data(num_samples=1000)

    # 准备数据加载器
    print("Preparing dataloaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        X, y,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        batch_size=config['data']['batch_size']
    )

    # 创建模型
    print("Creating model...")
    model = create_model(config, device)

    # 训练模型
    print("Training model...")
    history = train(model, train_loader, val_loader, config, device)

    # 评估模型
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    print("Test Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # 保存模型
    model_dir = Path(config['save']['model_dir'])
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / 'final_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
```

---

## 12. 常见问题与解决方案

### 12.1 训练问题

#### Q1: 损失不下降

**可能原因和解决方案:**

```python
# 1. 学习率太小
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 尝试增大

# 2. 学习率太大
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  # 尝试减小

# 3. 检查梯度是否为 None
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"{name} has no gradient!")

# 4. 检查是否忘记调用 optimizer.zero_grad()
optimizer.zero_grad()  # 必须在 loss.backward() 之前

# 5. 使用不同的优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

#### Q2: 模型过拟合

```python
# 1. 添加 Dropout
class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 50)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)

# 2. 使用权重衰减 (L2 正则化)
optimizer = torch.optim.Adam(model.parameters(),
                            lr=0.001,
                            weight_decay=1e-4)

# 3. 早停
# (见前面的早停实现)

# 4. 数据增强 (对于图像/文本)
# 5. 使用更多训练数据
```

#### Q3: 模型欠拟合

```python
# 1. 增加模型复杂度
class BiggerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# 2. 训练更多 epochs
epochs = 500  # 增加

# 3. 降低正则化强度
optimizer = torch.optim.Adam(model.parameters(),
                            lr=0.001,
                            weight_decay=1e-5)  # 减小

# 4. 尝试更复杂的特征工程
```

### 12.2 性能问题

#### Q4: 训练速度慢

```python
# 1. 使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 2. 增加 batch size
train_loader = DataLoader(dataset, batch_size=64)  # 增大

# 3. 使用 DataLoader 的 num_workers
train_loader = DataLoader(dataset, num_workers=4, pin_memory=True)

# 4. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = loss_fn(output, target)

# 5. 使用 torch.compile (PyTorch 2.0+)
model = torch.compile(model)
```

#### Q5: 内存不足

```python
# 1. 减小 batch size
train_loader = DataLoader(dataset, batch_size=16)  # 减小

# 2. 使用梯度累积
# (见前面的梯度累积实现)

# 3. 使用梯度检查点 (Gradient Checkpointing)
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x

# 4. 清理缓存
torch.cuda.empty_cache()

# 5. 使用 FP16
model = model.half()
```

### 12.3 数据问题

#### Q6: 数据不平衡

```python
# 1. 使用加权损失函数
weights = torch.tensor([1.0, 10.0])  # 类别权重
loss_fn = nn.CrossEntropyLoss(weight=weights)

# 2. 过采样少数类
from torch.utils.data import WeightedRandomSampler

# 计算样本权重
samples_weight = [...]  # 根据类别分配权重
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
train_loader = DataLoader(dataset, sampler=sampler)

# 3. 欠采样多数类
```

#### Q7: 数据标准化

```python
# 标准化数据 (均值=0, 标准差=1)
mean = X_train.mean()
std = X_train.std()

X_train_normalized = (X_train - mean) / std
X_test_normalized = (X_test - mean) / std

# 注意: 使用训练集的统计量来标准化测试集
```

---

## 📊 性能对比表

### 优化技术效果对比

| 技术 | 训练速度提升 | 内存节省 | 精度影响 | 实现难度 |
|-----|------------|---------|---------|---------|
| **AMP** | 1.5-3x | 50% | 无 | 低 |
| **梯度累积** | -10% | 75% | 无 | 低 |
| **DataLoader workers** | 1.5-2x | 无 | 无 | 低 |
| **Gradient Checkpointing** | -20% | 80% | 无 | 中 |
| **模型量化** | 2-4x | 75% | 微小 | 中 |
| **ONNX导出** | 1.2-2x | 无 | 无 | 低 |

---

## 🎓 学习检查清单

完成本教程后,你应该能够:

- [ ] 理解 PyTorch 的完整工作流程
- [ ] 创建和划分数据集
- [ ] 构建自定义 PyTorch 模型
- [ ] 实现训练和评估循环
- [ ] 使用各种损失函数和优化器
- [ ] 保存和加载模型
- [ ] 应用混合精度训练
- [ ] 实现学习率调度
- [ ] 使用 TensorBoard/W&B 监控训练
- [ ] 导出模型为 ONNX
- [ ] 处理常见的训练问题
- [ ] 优化训练性能
- [ ] 构建完整的机器学习项目

---

## 📚 推荐资源

### 官方文档
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [PyTorch 教程](https://pytorch.org/tutorials/)
- [PyTorch 示例](https://github.com/pytorch/examples)

### 学习资源
- [PyTorch 论坛](https://discuss.pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

### 书籍推荐
- "Deep Learning with PyTorch" (官方书籍)
- "Programming PyTorch for Deep Learning"

### 在线课程
- Fast.ai - Practical Deep Learning
- Coursera - Deep Learning Specialization
- Zero to Mastery - PyTorch for Deep Learning

---

## 🚀 下一步

完成本教程后,建议学习:

1. **02 - Neural Network Classification** - 神经网络分类
2. **03 - Computer Vision** - 计算机视觉
3. **04 - Custom Datasets** - 自定义数据集
4. **05 - Going Modular** - 模块化开发
5. **06 - Transfer Learning** - 迁移学习
6. **07 - Experiment Tracking** - 实验追踪
7. **08 - Model Deployment** - 模型部署

---

## ✍️ 练习题

### 初级练习

1. **创建自己的数据集**
   - 创建一个二次函数数据集: y = ax² + bx + c
   - 构建模型来学习参数 a, b, c

2. **实验不同的超参数**
   - 尝试学习率: [0.001, 0.01, 0.1]
   - 尝试优化器: SGD, Adam, AdamW
   - 绘制损失曲线并比较

3. **实现验证集**
   - 将数据划分为 训练/验证/测试 (70/15/15)
   - 在验证集上选择最佳模型

### 中级练习

4. **实现学习率调度器**
   - 使用 ReduceLROnPlateau
   - 绘制学习率变化曲线

5. **添加正则化**
   - 实现 L1/L2 正则化
   - 比较有无正则化的效果

6. **实现早停**
   - 监控验证损失
   - 在性能不再提升时停止训练

### 高级练习

7. **完整项目**
   - 构建一个房价预测模型
   - 使用真实数据集 (如 Boston Housing)
   - 实现完整的训练、评估、保存流程
   - 部署为 API 服务

8. **性能优化**
   - 实现混合精度训练
   - 使用梯度累积
   - 导出为 ONNX 并比较性能

9. **实验追踪**
   - 使用 W&B 或 TensorBoard
   - 记录所有超参数和指标
   - 进行超参数搜索

---

## 📝 总结

本教程涵盖了:

✅ **基础工作流程** - 从数据到部署的完整流程
✅ **核心概念** - 模型、损失、优化器、训练循环
✅ **最佳实践** - 数据划分、模型保存、推理模式
✅ **高级技术** - AMP、梯度累积、学习率调度
✅ **性能优化** - 提升训练速度和减少内存使用
✅ **生产部署** - ONNX 导出、量化、API 服务
✅ **实战项目** - 完整的项目结构和代码
✅ **问题解决** - 常见问题的诊断和解决

**记住:** 机器学习是一个迭代的过程。不断实验、可视化、调试,直到获得满意的结果!

---

**文档版本:** v2.0 (完整增强版)
**创建日期:** 2025-11-16
**作者:** 整合自 Learn PyTorch + 2024 最佳实践
**许可:** MIT License

**快速开始下一章:** [PyTorch Neural Network Classification →](https://www.learnpytorch.io/02_pytorch_classification/)
