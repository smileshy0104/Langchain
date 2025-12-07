# PyTorch 神经网络分类完整教程

> 基于 [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/02_pytorch_classification/) 整理
> 本教程涵盖二分类和多分类问题的完整 PyTorch 实现

---

## 目录

1. [什么是分类问题](#1-什么是分类问题)
2. [分类神经网络架构](#2-分类神经网络架构)
3. [准备分类数据](#3-准备分类数据)
4. [构建分类模型](#4-构建分类模型)
5. [设置损失函数和优化器](#5-设置损失函数和优化器)
6. [训练分类模型](#6-训练分类模型)
7. [从 Logits 到预测标签](#7-从-logits-到预测标签)
8. [模型改进与非线性](#8-模型改进与非线性)
9. [多分类问题](#9-多分类问题)
10. [分类评估指标](#10-分类评估指标)

---

## 1. 什么是分类问题

### 1.1 分类问题定义

**分类问题**是预测某个样本属于哪个类别的问题。

| 问题类型 | 输出 | 示例 |
|---------|------|------|
| **回归** | 连续数值 | 预测房价、温度 |
| **二分类** | 两个类别之一 | 是/否、猫/狗 |
| **多分类** | 多个类别之一 | 数字 0-9、动物种类 |

---

## 2. 分类神经网络架构

### 2.1 架构组件对比

| 组件 | 二分类 | 多分类 |
|------|--------|--------|
| **输入层** | `in_features` = 特征数 | `in_features` = 特征数 |
| **隐藏层激活** | ReLU, Tanh 等 | ReLU, Tanh 等 |
| **输出层** | `out_features` = 1 | `out_features` = 类别数 |
| **输出激活** | Sigmoid | Softmax |
| **损失函数** | `BCEWithLogitsLoss` | `CrossEntropyLoss` |
| **优化器** | SGD, Adam | SGD, Adam |

---

## 3. 准备分类数据

### 3.1 创建二分类数据（圆形数据）

```python
# ==================== 导入必要的库 ====================
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

# ==================== 生成圆形分类数据 ====================
# make_circles() 生成两个同心圆的数据点
# - 内圆的点标签为 1（蓝色）
# - 外圆的点标签为 0（红色）
# 
# 这是一个经典的非线性分类问题：
# - 无法用一条直线将两类数据分开
# - 需要非线性决策边界（圆形）
n_samples = 1000

X, y = make_circles(
    n_samples=n_samples,  # 总样本数量
    noise=0.03,           # 噪声程度（0-1），使数据更真实
                          # noise=0: 完美的圆
                          # noise=0.1: 有一些散乱
    random_state=42       # 随机种子，确保结果可重复
)

# ==================== 查看数据结构 ====================
print(f"前 5 个特征 X:\n{X[:5]}")
print(f"\n前 5 个标签 y:\n{y[:5]}")

# 输出示例：
# 前 5 个特征 X:
# [[ 0.75424625  0.23148074]   <- 每个样本有 2 个特征 (x, y 坐标)
#  [-0.75615888  0.15325888]
#  [-0.81539193  0.17328203]
#  [-0.39373073  0.69288277]
#  [ 0.44220765 -0.89672343]]
# 
# 前 5 个标签 y:
# [1 1 1 1 0]                   <- 标签是 0 或 1

# ==================== 数据探索 ====================
# 转换为 DataFrame 便于查看
circles = pd.DataFrame({
    "X1": X[:, 0],    # 第一个特征
    "X2": X[:, 1],    # 第二个特征
    "label": y        # 标签
})
print(circles.head(10))

# 检查类别分布（是否平衡）
print(f"\n类别分布:\n{circles.label.value_counts()}")
# 输出：
# label
# 1    500
# 0    500
# 数据是平衡的，每个类别各 500 个样本

# ==================== 可视化数据 ====================
plt.figure(figsize=(10, 7))
plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.title("圆形分类数据 - 红色(0) vs 蓝色(1)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar(label="类别")
plt.show()

# ==================== 转换为 PyTorch 张量 ====================
# PyTorch 需要张量格式的数据，NumPy 数组需要转换
# 
# torch.from_numpy(): 从 NumPy 数组创建张量
# .type(torch.float): 转换为 float32（PyTorch 默认浮点类型）
# 
# 为什么用 float？
# - 神经网络的计算需要浮点数
# - float32 是精度和速度的平衡
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)  # 二分类标签也用 float

print(f"\n张量类型:")
print(f"X dtype: {X.dtype}")  # torch.float32
print(f"y dtype: {y.dtype}")  # torch.float32

# ==================== 划分训练集和测试集 ====================
# train_test_split() 随机划分数据
# 
# 为什么要划分？
# - 训练集：用于训练模型，学习数据中的模式
# - 测试集：用于评估模型，检验泛化能力
# 
# 常见比例：
# - 80/20（最常用）
# - 70/30
# - 90/10（数据量大时）
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,      # 20% 作为测试集
    random_state=42     # 随机种子，确保划分可重复
)

print(f"\n数据集大小:")
print(f"训练集: {len(X_train)} 样本")  # 800
print(f"测试集: {len(X_test)} 样本")   # 200
```

### 3.2 理解输入输出形状

```python
# ==================== 检查数据形状 ====================
# 这是深度学习中最重要的步骤之一！
# 形状不匹配是最常见的错误来源
# 
# 数据探索者的座右铭："可视化，可视化，可视化！"
# 对于形状："检查，检查，检查！"

print(f"X 形状: {X.shape}")  # torch.Size([1000, 2])
print(f"y 形状: {y.shape}")  # torch.Size([1000])

# 形状解读：
# X: (1000, 2)
#    - 1000: 样本数量
#    - 2: 每个样本的特征数量（X1, X2 坐标）
# 
# y: (1000,)
#    - 1000: 标签数量（与样本数量匹配）
#    - 每个标签是一个标量（0 或 1）

# ==================== 查看单个样本 ====================
X_sample = X[0]
y_sample = y[0]

print(f"\n单个样本:")
print(f"特征值: {X_sample}")           # tensor([0.7542, 0.2315])
print(f"特征形状: {X_sample.shape}")   # torch.Size([2])
print(f"标签值: {y_sample}")           # tensor(1.)
print(f"标签形状: {y_sample.shape}")   # torch.Size([])  <- 标量

# ==================== 关键理解 ====================
# 
# 输入 → 模型 → 输出
# (2,)  → ???  → (1,)
# 
# 模型需要：
# - in_features = 2  （输入特征数）
# - out_features = 1 （输出特征数，二分类只需要 1 个）
# 
# 这就是为什么我们的第一层是：
# nn.Linear(in_features=2, out_features=5)
#           ↑ 匹配输入特征数
```

### 3.3 创建多分类数据（Blob 数据）

```python
from sklearn.datasets import make_blobs

# ==================== 设置超参数 ====================
NUM_CLASSES = 4      # 类别数量（4 个不同的簇）
NUM_FEATURES = 2     # 特征数量（2D 数据，便于可视化）
RANDOM_SEED = 42     # 随机种子

# ==================== 生成多分类数据 ====================
# make_blobs() 生成多个高斯分布的数据簇
# 每个簇代表一个类别
# 
# 与 make_circles 的区别：
# - make_circles: 生成同心圆（非线性边界）
# - make_blobs: 生成分离的簇（可能线性可分）
X_blob, y_blob = make_blobs(
    n_samples=1000,           # 总样本数
    n_features=NUM_FEATURES,  # 特征数量
    centers=NUM_CLASSES,      # 类别/簇的数量
    cluster_std=1.5,          # 簇的标准差（控制分散程度）
                              # 值越大，簇越分散，分类越难
    random_state=RANDOM_SEED
)

# ==================== 转换为张量 ====================
# 注意：多分类标签必须用 LongTensor！
# 
# 为什么？
# - CrossEntropyLoss 要求标签是整数索引
# - LongTensor 是 64 位整数类型
# - 标签值是 [0, 1, 2, 3]，代表 4 个类别
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)  # 重要！

print(f"X_blob 形状: {X_blob.shape}")  # torch.Size([1000, 2])
print(f"y_blob 形状: {y_blob.shape}")  # torch.Size([1000])
print(f"y_blob dtype: {y_blob.dtype}") # torch.int64

print(f"\n前 5 个标签: {y_blob[:5]}")  # tensor([3, 2, 2, 1, 1])
print(f"唯一标签值: {torch.unique(y_blob)}")  # tensor([0, 1, 2, 3])

# ==================== 标签格式对比 ====================
# 
# | 问题类型 | 标签类型 | 标签值示例 | PyTorch 类型 |
# |---------|---------|-----------|-------------|
# | 二分类 | Float | 0.0, 1.0 | torch.float |
# | 多分类 | Long | 0, 1, 2, 3 | torch.LongTensor |
# 
# 这是一个常见的错误来源！

# ==================== 划分数据集 ====================
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
    X_blob,
    y_blob,
    test_size=0.2,
    random_state=RANDOM_SEED
)

# ==================== 可视化多分类数据 ====================
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.title("多分类 Blob 数据 - 4 个类别")
plt.xlabel("X1")
plt.ylabel("X2")
plt.colorbar(label="类别")
plt.show()

# 问题思考：这个数据集需要非线性激活函数吗？
# 答案：不一定！因为这些簇可能用直线就能分开
```

---

## 4. 构建分类模型

### 4.1 设备无关代码

```python
import torch
from torch import nn

# ==================== 设备无关代码 ====================
# 自动选择最佳可用设备
# - 如果有 NVIDIA GPU 且安装了 CUDA，使用 GPU
# - 否则使用 CPU
# 
# 为什么重要？
# - GPU 训练速度比 CPU 快 10-100 倍
# - 但代码需要能在两种设备上运行
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 检查 GPU 信息（如果可用）
if device == "cuda":
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 4.2 二分类模型（无非线性）- 版本 0

```python
# ==================== 简单二分类模型（无激活函数）====================
# 这个模型只有线性层，没有非线性激活函数
# 
# 问题：只能学习线性决策边界（直线）
# 对于圆形数据，这个模型会失败！

class CircleModelV0(nn.Module):
    """
    简单的二分类模型 - 纯线性
    
    结构：
    输入(2) → 隐藏层(5) → 输出(1)
    
    问题：无法处理非线性数据
    """
    def __init__(self):
        super().__init__()
        
        # ==================== 第一层 ====================
        # in_features=2: 输入特征数（X1, X2）
        # out_features=5: 隐藏单元数（超参数，可调整）
        # 
        # 为什么是 5？
        # - 这是一个超参数，没有固定答案
        # - 更多隐藏单元 = 更强的学习能力
        # - 但也可能导致过拟合
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        
        # ==================== 第二层（输出层）====================
        # in_features=5: 必须匹配上一层的 out_features
        # out_features=1: 二分类只需要 1 个输出
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
    
    def forward(self, x):
        """
        前向传播
        
        数据流：x → layer_1 → layer_2 → 输出
        
        注意：没有激活函数！
        这意味着：layer_2(layer_1(x)) = W2(W1*x + b1) + b2
                                      = W2*W1*x + W2*b1 + b2
                                      = W'*x + b'  （仍然是线性的！）
        """
        return self.layer_2(self.layer_1(x))

# 创建模型并移动到设备
model_0 = CircleModelV0().to(device)
print(model_0)

# 输出：
# CircleModelV0(
#   (layer_1): Linear(in_features=2, out_features=5, bias=True)
#   (layer_2): Linear(in_features=5, out_features=1, bias=True)
# )
```

### 4.3 使用 nn.Sequential 构建模型

```python
# ==================== 使用 nn.Sequential ====================
# nn.Sequential 是一种更简洁的方式来定义顺序模型
# 数据按顺序通过每一层

model_0_sequential = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

print(model_0_sequential)
# Sequential(
#   (0): Linear(in_features=2, out_features=5, bias=True)
#   (1): Linear(in_features=5, out_features=1, bias=True)
# )

# ==================== nn.Sequential vs nn.Module 子类 ====================
# 
# nn.Sequential 优点：
# ✅ 代码更简洁
# ✅ 适合简单的顺序模型
# ✅ 不需要定义 forward 方法
# 
# nn.Module 子类优点：
# ✅ 更灵活（可以自定义前向传播逻辑）
# ✅ 可以实现复杂结构（残差连接、多分支、跳跃连接）
# ✅ 更容易调试和理解
# ✅ 可以在 forward 中添加条件逻辑
# 
# 建议：
# - 简单模型 → nn.Sequential
# - 复杂模型 → nn.Module 子类
```

### 4.4 带非线性激活的二分类模型 - 版本 2

```python
# ==================== 带 ReLU 激活函数的模型 ====================
# 关键改进：在隐藏层之间添加 ReLU 激活函数
# 这使模型能够学习非线性决策边界！

class CircleModelV2(nn.Module):
    """
    带非线性激活函数的二分类模型
    
    结构：
    输入(2) → 线性(10) → ReLU → 线性(10) → ReLU → 线性(1) → 输出
    
    ReLU 的作用：
    - 引入非线性
    - 使模型能够学习曲线边界
    - 解决纯线性模型的局限性
    """
    def __init__(self):
        super().__init__()
        
        # 三个线性层，逐渐减少特征数
        self.layer_1 = nn.Linear(in_features=2, out_features=10)   # 2 → 10
        self.layer_2 = nn.Linear(in_features=10, out_features=10)  # 10 → 10
        self.layer_3 = nn.Linear(in_features=10, out_features=1)   # 10 → 1
        
        # ==================== ReLU 激活函数 ====================
        # ReLU(x) = max(0, x)
        # 
        # 特点：
        # - 负数 → 0
        # - 正数 → 保持不变
        # - 计算简单高效
        # - 解决梯度消失问题
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        前向传播（带非线性激活）
        
        数据流：
        x → layer_1 → ReLU → layer_2 → ReLU → layer_3 → 输出
        
        为什么在隐藏层后加 ReLU？
        - 引入非线性变换
        - 使模型能够学习复杂模式
        - 输出层后通常不加激活（由损失函数处理）
        """
        # 方法 1：逐步计算（更清晰）
        # z1 = self.layer_1(x)
        # a1 = self.relu(z1)
        # z2 = self.layer_2(a1)
        # a2 = self.relu(z2)
        # z3 = self.layer_3(a2)
        # return z3
        
        # 方法 2：链式调用（更简洁）
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

# 创建模型
model_3 = CircleModelV2().to(device)
print(model_3)

# ==================== 激活函数放置位置 ====================
# 
# 正确：
# Input → Linear → ReLU → Linear → ReLU → Linear → Output
#                  ↑              ↑
#               激活函数        激活函数
# 
# 注意：
# - 输出层后通常不加激活函数
# - BCEWithLogitsLoss 内置 Sigmoid
# - CrossEntropyLoss 内置 Softmax
```

### 4.5 多分类模型

```python
# ==================== 多分类模型 ====================
# 与二分类的主要区别：
# - 输出层的 out_features = 类别数量（而不是 1）
# - 使用 CrossEntropyLoss（而不是 BCELoss）
# - 使用 Softmax（而不是 Sigmoid）获取概率

class BlobModel(nn.Module):
    """
    多分类模型
    
    参数化设计：可以适应不同的输入/输出大小
    
    参数：
        input_features: 输入特征数量
        output_features: 输出类别数量
        hidden_units: 隐藏层神经元数量
    """
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        
        # ==================== 使用 nn.Sequential 构建层栈 ====================
        self.linear_layer_stack = nn.Sequential(
            # 第一层：输入 → 隐藏
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),  # 激活函数
            
            # 第二层：隐藏 → 隐藏
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),  # 激活函数
            
            # 第三层：隐藏 → 输出
            # out_features = 类别数量！
            nn.Linear(in_features=hidden_units, out_features=output_features)
            # 注意：输出层后没有激活函数
            # CrossEntropyLoss 会自动处理 Softmax
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)

# ==================== 创建多分类模型 ====================
# input_features=2: 2 个输入特征
# output_features=4: 4 个类别
# hidden_units=8: 8 个隐藏神经元
model_4 = BlobModel(
    input_features=NUM_FEATURES,  # 2
    output_features=NUM_CLASSES,  # 4
    hidden_units=8
).to(device)

print(model_4)
# BlobModel(
#   (linear_layer_stack): Sequential(
#     (0): Linear(in_features=2, out_features=8, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=8, out_features=8, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=8, out_features=4, bias=True)  <- 4 个输出！
#   )
# )

# ==================== 测试模型输出形状 ====================
# 在训练前，先检查模型输出是否正确
with torch.inference_mode():
    sample_output = model_4(X_blob_train[:5].to(device))
    print(f"\n输入形状: {X_blob_train[:5].shape}")   # [5, 2]
    print(f"输出形状: {sample_output.shape}")        # [5, 4] - 每个样本 4 个类别的 logits
    print(f"输出示例:\n{sample_output}")
```

---

## 5. 设置损失函数和优化器

### 5.1 分类问题的损失函数

```python
# ==================== 二分类损失函数 ====================

# 方法 1: nn.BCELoss（需要先手动应用 Sigmoid）
# loss_fn = nn.BCELoss()
# 使用时：loss = loss_fn(torch.sigmoid(y_logits), y_train)
# 
# 缺点：
# - 需要手动添加 Sigmoid
# - 数值不稳定（可能出现 log(0)）

# 方法 2: nn.BCEWithLogitsLoss（推荐！内置 Sigmoid）
loss_fn_binary = nn.BCEWithLogitsLoss()
# 使用时：loss = loss_fn(y_logits, y_train)  # 直接使用原始 logits
# 
# 优点：
# ✅ 数值更稳定（内部优化了计算）
# ✅ 更高效（Sigmoid 和 Loss 合并计算）
# ✅ PyTorch 官方推荐

# ==================== 多分类损失函数 ====================
loss_fn_multi = nn.CrossEntropyLoss()
# 
# CrossEntropyLoss 的特点：
# ✅ 内置 Softmax（不需要在模型中添加）
# ✅ 期望输入：原始 logits，形状 (batch_size, num_classes)
# ✅ 期望标签：类别索引，形状 (batch_size,)，类型 LongTensor
# 
# 内部计算：
# 1. 对 logits 应用 log_softmax
# 2. 计算负对数似然损失 (NLLLoss)
# 
# 公式：CE = -Σ y_true * log(softmax(logits))

# ==================== 损失函数数学原理 ====================
# 
# 二分类交叉熵 (BCE):
# L = -[y * log(p) + (1-y) * log(1-p)]
# 其中 p = sigmoid(logits)
# 
# 多分类交叉熵 (CE):
# L = -Σ y_i * log(p_i)
# 其中 p = softmax(logits)
# 
# 直观理解：
# - 损失函数衡量预测概率与真实标签的差距
# - 预测越准确，损失越小
# - 预测越错误，损失越大（惩罚错误预测）
```

### 5.2 损失函数对比表

| 问题类型 | 损失函数 | 输入要求 | 标签要求 | 内置激活 |
|---------|---------|---------|---------|---------|
| 二分类 | `BCELoss` | Sigmoid 后的概率 | Float [0, 1] | ❌ |
| 二分类 | `BCEWithLogitsLoss` | 原始 logits | Float [0, 1] | ✅ Sigmoid |
| 多分类 | `CrossEntropyLoss` | 原始 logits | Long [0, C-1] | ✅ Softmax |
| 多分类 | `NLLLoss` | log_softmax 后 | Long [0, C-1] | ❌ |

### 5.3 优化器设置

```python
# ==================== 创建优化器 ====================
# 优化器负责更新模型参数，使损失最小化

# SGD（随机梯度下降）- 最基础的优化器
optimizer = torch.optim.SGD(
    params=model_0.parameters(),  # 要优化的参数
    lr=0.1                        # 学习率（步长）
)

# ==================== 其他常用优化器 ====================

# Adam - 自适应学习率，通常效果更好
# optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# SGD + Momentum - 加速收敛
# optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01, momentum=0.9)

# AdamW - Adam + 权重衰减（正则化）
# optimizer = torch.optim.AdamW(params=model_0.parameters(), lr=0.001, weight_decay=0.01)

# ==================== 学习率选择建议 ====================
# 
# | 优化器 | 典型学习率 | 特点 |
# |-------|-----------|------|
# | SGD | 0.01 - 0.1 | 简单，可能需要调参 |
# | SGD + Momentum | 0.01 - 0.1 | 更快收敛，减少震荡 |
# | Adam | 0.001 - 0.01 | 自适应，通常效果好 |
# | AdamW | 0.001 - 0.01 | Adam + 权重衰减 |
# 
# 学习率太大：损失震荡，无法收敛
# 学习率太小：收敛太慢，可能陷入局部最优
```

### 5.4 准确率评估函数

```python
# ==================== 准确率函数 ====================
def accuracy_fn(y_true, y_pred):
    """
    计算分类准确率
    
    参数：
        y_true: 真实标签
        y_pred: 预测标签（不是概率！）
    
    返回：
        准确率（百分比）
    
    公式：
        准确率 = 正确预测数 / 总预测数 × 100%
    """
    # torch.eq(): 逐元素比较两个张量是否相等
    # 返回布尔张量，True 表示相等
    # .sum(): 统计 True 的数量（即正确预测的数量）
    # .item(): 将单元素张量转换为 Python 数值
    correct = torch.eq(y_true, y_pred).sum().item()
    
    # 计算准确率百分比
    acc = (correct / len(y_pred)) * 100
    
    return acc

# ==================== 使用示例 ====================
# 假设：
# y_true = tensor([0, 1, 1, 0, 1])
# y_pred = tensor([0, 1, 0, 0, 1])
# 
# torch.eq(y_true, y_pred) = tensor([True, True, False, True, True])
# .sum() = 4
# 准确率 = 4/5 * 100 = 80%
```

---

## 6. 训练分类模型

### 6.1 训练循环的五个步骤

```python
"""
PyTorch 训练循环的标准步骤（必须记住！）

┌─────────────────────────────────────────────────────────────┐
│  1. 前向传播 (Forward Pass)                                  │
│     y_pred = model(X_train)                                 │
│     - 数据通过模型，得到预测值                                │
├─────────────────────────────────────────────────────────────┤
│  2. 计算损失 (Calculate Loss)                                │
│     loss = loss_fn(y_pred, y_train)                         │
│     - 比较预测值和真实值，计算误差                            │
├─────────────────────────────────────────────────────────────┤
│  3. 清零梯度 (Zero Gradients)                                │
│     optimizer.zero_grad()                                   │
│     - PyTorch 默认累积梯度，必须手动清零                      │
│     - 如果不清零，梯度会越来越大！                            │
├─────────────────────────────────────────────────────────────┤
│  4. 反向传播 (Backpropagation)                               │
│     loss.backward()                                         │
│     - 计算损失对每个参数的梯度                                │
│     - 梯度 = 损失变化 / 参数变化                              │
├─────────────────────────────────────────────────────────────┤
│  5. 更新参数 (Optimizer Step)                                │
│     optimizer.step()                                        │
│     - 使用梯度更新参数：param = param - lr * grad            │
│     - 目标：使损失最小化                                      │
└─────────────────────────────────────────────────────────────┘

记忆口诀：前向 → 损失 → 清零 → 反向 → 更新
"""
```

### 6.2 完整的二分类训练循环

```python
# ==================== 设置随机种子 ====================
# 确保结果可重复
torch.manual_seed(42)

# ==================== 设置训练参数 ====================
epochs = 1000  # 训练轮数

# ==================== 移动数据到设备 ====================
# 数据和模型必须在同一设备上！
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# ==================== 训练循环 ====================
for epoch in range(epochs):
    
    # ========== 训练阶段 ==========
    model_3.train()  # 设置为训练模式
                     # 启用 Dropout、BatchNorm 等训练行为
    
    # 1. 前向传播
    y_logits = model_3(X_train).squeeze()  # 获取原始 logits
    # .squeeze(): 去除多余的维度
    # 例如：[800, 1] → [800]
    
    # 将 logits 转换为预测标签（用于计算准确率）
    # logits → sigmoid → 概率 → round → 标签
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    # 2. 计算损失和准确率
    loss = loss_fn(y_logits, y_train)  # BCEWithLogitsLoss 直接使用 logits
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    
    # 3. 清零梯度
    optimizer.zero_grad()
    
    # 4. 反向传播
    loss.backward()
    
    # 5. 更新参数
    optimizer.step()
    
    # ========== 测试/评估阶段 ==========
    model_3.eval()  # 设置为评估模式
                    # 禁用 Dropout、BatchNorm 使用运行统计
    
    with torch.inference_mode():  # 不计算梯度，节省内存和计算
        # 前向传播
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        # 计算测试损失和准确率
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    
    # ========== 打印进度 ==========
    if epoch % 100 == 0:
        print(f"Epoch: {epoch:4d} | "
              f"Train Loss: {loss:.5f}, Train Acc: {acc:.2f}% | "
              f"Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

# ==================== 预期输出（带非线性模型）====================
# Epoch:    0 | Train Loss: 0.69295, Train Acc: 50.00% | Test Loss: 0.69319, Test Acc: 50.00%
# Epoch:  100 | Train Loss: 0.69115, Train Acc: 52.88% | Test Loss: 0.69102, Test Acc: 52.50%
# ...
# Epoch:  900 | Train Loss: 0.62362, Train Acc: 74.00% | Test Loss: 0.62145, Test Acc: 79.00%
# 
# 注意：准确率从 50%（随机猜测）提升到 ~80%！
```

### 6.3 model.train() vs model.eval()

```python
# ==================== 训练模式 vs 评估模式 ====================
# 
# model.train():
# - 启用 Dropout（随机丢弃神经元，防止过拟合）
# - BatchNorm 使用当前 batch 的统计量
# - 用于训练阶段
# 
# model.eval():
# - 禁用 Dropout（使用所有神经元）
# - BatchNorm 使用运行时统计量
# - 用于评估/推理阶段
# 
# 重要：即使模型没有 Dropout/BatchNorm，也建议使用这两个方法
# 这是一个好习惯，确保代码在添加这些层后仍然正确

# ==================== torch.inference_mode() ====================
# 
# 与 torch.no_grad() 类似，但更高效
# - 不计算梯度
# - 不构建计算图
# - 节省内存和计算时间
# 
# 用于：
# - 模型评估
# - 模型推理/预测
# - 任何不需要反向传播的场景
```

---

## 7. 从 Logits 到预测标签

### 7.1 什么是 Logits？

```python
# ==================== 理解 Logits ====================
# 
# Logits（对数几率）是模型最后一层的原始输出
# - 未经过激活函数处理
# - 可以是任意实数（正数、负数、零）
# - 不能直接解释为概率
# 
# 为什么叫 logits？
# - 来自 "log-odds"（对数几率）
# - 在统计学中，logit 函数是 sigmoid 的反函数
# - logit(p) = log(p / (1-p))

# 查看模型的原始输出
y_logits = model_3(X_test.to(device))[:5]
print(f"原始 logits:\n{y_logits}")
# tensor([[0.0555],
#         [0.0169],
#         [0.2254],
#         [0.0071],
#         [0.3345]], device='cuda:0')
# 
# 这些数字是什么意思？
# - 正数：模型倾向于预测类别 1
# - 负数：模型倾向于预测类别 0
# - 绝对值越大：模型越确信
```

### 7.2 二分类：Logits → 概率 → 标签

```python
# ==================== 步骤 1: Logits → 预测概率 ====================
# 使用 Sigmoid 函数将 logits 转换为 [0, 1] 之间的概率
# 
# Sigmoid 公式: σ(x) = 1 / (1 + e^(-x))
# 
# 特点：
# - 输入任意实数，输出 [0, 1]
# - 输入越大，输出越接近 1
# - 输入越小，输出越接近 0
# - 输入为 0 时，输出为 0.5

y_logits = model_3(X_test.to(device))[:5]
y_pred_probs = torch.sigmoid(y_logits)

print(f"Logits: {y_logits.squeeze()}")
print(f"概率:   {y_pred_probs.squeeze()}")
# Logits: tensor([0.0555, 0.0169, 0.2254, 0.0071, 0.3345])
# 概率:   tensor([0.5139, 0.5042, 0.5561, 0.5018, 0.5829])

# ==================== 步骤 2: 预测概率 → 预测标签 ====================
# 使用阈值 0.5 将概率转换为类别标签
# 
# 规则：
# - 概率 >= 0.5 → 类别 1
# - 概率 < 0.5 → 类别 0
# 
# torch.round(): 四舍五入
# - 0.5139 → 1
# - 0.4999 → 0

y_preds = torch.round(y_pred_probs)
print(f"预测标签: {y_preds.squeeze()}")
# 预测标签: tensor([1., 1., 1., 1., 1.])

# ==================== 完整转换（一行代码）====================
y_pred_labels = torch.round(torch.sigmoid(model_3(X_test.to(device))))

# ==================== 转换流程图 ====================
# 
# 原始输出      Sigmoid        Round
# (Logits)  →  (概率)    →   (标签)
#   0.5     →   0.62     →     1
#  -1.2     →   0.23     →     0
#   2.5     →   0.92     →     1
```

### 7.3 多分类：Logits → 概率 → 标签

```python
# ==================== 步骤 1: Logits → 预测概率 ====================
# 使用 Softmax 函数将 logits 转换为概率分布
# 
# Softmax 公式: softmax(x_i) = e^(x_i) / Σ(e^(x_j))
# 
# 特点：
# - 输出是一个概率分布
# - 所有输出值之和 = 1
# - 每个值在 (0, 1) 之间
# - 最大的 logit 对应最大的概率

y_logits = model_4(X_blob_test.to(device))[:5]
print(f"原始 logits:\n{y_logits}")
# tensor([[ 4.3377, 10.3539, -14.8948, -9.7642],
#         [ 5.0142, -12.0371,   3.3860, 10.6699],
#         ...])

y_pred_probs = torch.softmax(y_logits, dim=1)  # dim=1: 在类别维度上计算
print(f"\n预测概率:\n{y_pred_probs}")
# tensor([[0.0001, 0.9999, 0.0000, 0.0000],  <- 类别 1 概率最高
#         [0.0028, 0.0000, 0.0005, 0.9967],  <- 类别 3 概率最高
#         ...])

# 验证概率之和为 1
print(f"\n第一个样本的概率之和: {y_pred_probs[0].sum():.4f}")  # 1.0000

# ==================== 步骤 2: 预测概率 → 预测标签 ====================
# 使用 argmax 选择概率最大的类别
# 
# argmax 返回最大值的索引（即类别编号）
# dim=1: 在类别维度上找最大值

y_preds = y_pred_probs.argmax(dim=1)
print(f"\n预测标签: {y_preds}")
# tensor([1, 3, 2, 1, 0])

# ==================== 完整转换（一行代码）====================
y_pred_labels = torch.softmax(model_4(X_blob_test.to(device)), dim=1).argmax(dim=1)

# 或者直接使用 argmax（跳过 softmax，结果相同）
# 因为 softmax 保持相对大小关系
y_pred_labels = model_4(X_blob_test.to(device)).argmax(dim=1)

# ==================== 转换流程图 ====================
# 
# 原始输出 (4个类别)      Softmax           Argmax
#    (Logits)        →  (概率分布)    →   (类别索引)
# [2.0, 1.0, 0.5, 0.1] → [0.5, 0.2, 0.1, 0.1] →  0
# [0.1, 3.0, 1.0, 0.5] → [0.1, 0.7, 0.1, 0.1] →  1
```

### 7.4 转换流程总结

```
二分类转换流程：
┌─────────┐    Sigmoid    ┌─────────────┐    Round     ┌──────────┐
│ Logits  │ ────────────→ │ 概率 [0,1]  │ ──────────→ │ 标签 0/1 │
└─────────┘               └─────────────┘              └──────────┘
  例: 0.5      →           0.62          →              1

多分类转换流程：
┌─────────┐    Softmax    ┌─────────────┐    Argmax    ┌──────────┐
│ Logits  │ ────────────→ │ 概率分布    │ ──────────→ │ 类别索引 │
└─────────┘               └─────────────┘              └──────────┘
  [1.2, 0.5,    →         [0.5, 0.2,      →              0
   -0.3, 0.1]              0.1, 0.2]
```

| 步骤 | 二分类 | 多分类 |
|------|--------|--------|
| **原始输出** | Logits (1个值) | Logits (C个值) |
| **激活函数** | Sigmoid | Softmax |
| **输出范围** | [0, 1] | 概率分布，和为1 |
| **获取标签** | Round (阈值 0.5) | Argmax |
| **标签格式** | 0 或 1 | 0 到 C-1 |

---

## 8. 模型改进与非线性

### 8.1 为什么模型表现不好？

```python
# ==================== 问题分析 ====================
# 
# 现象：模型在圆形数据上只有 ~50% 的准确率（和随机猜测一样）
# 
# 原因分析：
# 1. 圆形数据是非线性的（内圆 vs 外圆）
# 2. 我们的模型只有线性层（nn.Linear）
# 3. 多个线性层的组合仍然是线性的！
# 4. 线性模型只能画直线来分割数据
# 5. 直线无法正确分割圆形数据
# 
# 数学证明：
# 两个线性层的组合：
# y = W2(W1*x + b1) + b2
#   = W2*W1*x + W2*b1 + b2
#   = W'*x + b'  （仍然是线性的！）
# 
# 无论堆叠多少线性层，结果都是线性的！
```

### 8.2 从模型角度改进的方法

```python
# ==================== 改进方法 ====================
# 
# 1. 添加更多层
#    - 增加模型的深度
#    - 但没有非线性，仍然是线性模型
# 
# 2. 添加更多神经元
#    - 增加每层的 hidden_units
#    - 增加模型的宽度
# 
# 3. 训练更长时间
#    - 增加 epochs
#    - 给模型更多学习机会
# 
# 4. 改变学习率
#    - 太大：可能跳过最优解
#    - 太小：收敛太慢
# 
# 5. 改变激活函数 ⭐ 关键！
#    - 添加非线性激活函数
#    - 使模型能够学习非线性关系
# 
# 机器学习的座右铭："实验，实验，实验！"
```

### 8.3 常用激活函数详解

```python
# ==================== ReLU 激活函数 ====================
# ReLU (Rectified Linear Unit) 是最常用的激活函数
# 
# 公式: ReLU(x) = max(0, x)
# 
# 特点：
# ✅ 负数 → 0
# ✅ 正数 → 保持不变
# ✅ 计算简单高效
# ✅ 解决梯度消失问题
# ✅ 稀疏激活（部分神经元输出为 0）
# 
# 缺点：
# ❌ "死亡 ReLU" 问题（神经元可能永远输出 0）

# 手动实现 ReLU
def relu(x):
    return torch.maximum(torch.tensor(0), x)

# 使用 PyTorch 内置
relu_layer = nn.ReLU()

# 可视化 ReLU
import matplotlib.pyplot as plt
A = torch.arange(-10, 10, 1, dtype=torch.float32)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(A, label='原始')
plt.plot(relu(A), label='ReLU')
plt.legend()
plt.title("ReLU 激活函数")

# ==================== Sigmoid 激活函数 ====================
# Sigmoid 将任意实数映射到 (0, 1)
# 
# 公式: σ(x) = 1 / (1 + e^(-x))
# 
# 特点：
# ✅ 输出范围 (0, 1)，可解释为概率
# ✅ 平滑可微
# 
# 缺点：
# ❌ 梯度消失问题（输入很大或很小时）
# ❌ 输出不是零中心的
# ❌ 计算相对较慢（指数运算）
# 
# 用途：
# - 二分类的输出层
# - 将 logits 转换为概率

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

plt.subplot(1, 2, 2)
plt.plot(A, label='原始')
plt.plot(sigmoid(A), label='Sigmoid')
plt.legend()
plt.title("Sigmoid 激活函数")
plt.show()

# ==================== 其他常用激活函数 ====================
# 
# | 激活函数 | 公式 | 用途 |
# |---------|------|------|
# | ReLU | max(0, x) | 隐藏层（最常用）|
# | LeakyReLU | max(0.01x, x) | 解决死亡 ReLU |
# | Sigmoid | 1/(1+e^(-x)) | 二分类输出 |
# | Softmax | e^(x_i)/Σe^(x_j) | 多分类输出 |
# | Tanh | (e^x - e^(-x))/(e^x + e^(-x)) | 隐藏层（输出零中心）|
# | GELU | x * Φ(x) | Transformer 中常用 |
```

### 8.4 激活函数放置位置

```python
# ==================== 激活函数应该放在哪里？====================
# 
# 一般规则：
# 1. 在隐藏层之间放置激活函数
# 2. 输出层后通常不放激活函数（或根据任务选择）
# 
# 常见模式：
# Input → Linear → ReLU → Linear → ReLU → Linear → Output
#                  ↑              ↑
#               激活函数        激活函数
# 
# 输出层激活函数选择：
# - 二分类: Sigmoid（或在损失函数中处理）
# - 多分类: Softmax（或在损失函数中处理）
# - 回归: 通常不需要激活函数
# 
# 注意：
# - BCEWithLogitsLoss 内置 Sigmoid
# - CrossEntropyLoss 内置 Softmax
# - 使用这些损失函数时，输出层不需要激活函数

# ==================== 为什么非线性如此重要？====================
# 
# 想象一下：你能用多少条直线和曲线画出什么图案？
# - 只用直线：只能画直线、三角形、多边形
# - 加上曲线：可以画任何形状！
# 
# 神经网络也是如此：
# - 只有线性层：只能学习线性边界
# - 加上非线性：可以学习任意复杂的边界
# 
# 这就是为什么深度学习如此强大！
```

---

## 9. 多分类问题

### 9.1 二分类 vs 多分类

```python
# ==================== 问题类型对比 ====================
# 
# 二分类 (Binary Classification):
# - 只有 2 个类别
# - 例如：猫 vs 狗、垃圾邮件 vs 正常邮件
# - 输出：1 个值（属于类别 1 的概率）
# 
# 多分类 (Multi-class Classification):
# - 有 > 2 个类别
# - 例如：手写数字 0-9、动物种类
# - 输出：C 个值（每个类别的概率）
# 
# 多标签分类 (Multi-label Classification):
# - 一个样本可以属于多个类别
# - 例如：一张图片可能同时包含猫和狗
# - 不在本教程范围内
```

### 9.2 Softmax 详解

```python
# ==================== Softmax 函数 ====================
# Softmax 将一组数值转换为概率分布
# 
# 公式: softmax(x_i) = e^(x_i) / Σ(e^(x_j))
# 
# 特点：
# 1. 所有输出值之和 = 1
# 2. 每个输出值在 (0, 1) 之间
# 3. 保持相对大小关系（最大的 logit → 最大的概率）
# 4. 对异常值敏感（指数放大差异）

# 示例
logits = torch.tensor([2.0, 1.0, 0.1])
probs = torch.softmax(logits, dim=0)

print(f"Logits: {logits}")
print(f"Softmax 概率: {probs}")
print(f"概率之和: {probs.sum():.4f}")

# 输出：
# Logits: tensor([2.0000, 1.0000, 0.1000])
# Softmax 概率: tensor([0.6590, 0.2424, 0.0986])
# 概率之和: 1.0000

# ==================== dim 参数 ====================
# dim 指定在哪个维度上计算 softmax
# 
# 对于 batch 数据 (batch_size, num_classes):
# - dim=0: 在 batch 维度上计算（通常不用）
# - dim=1: 在类别维度上计算（正确用法）

batch_logits = torch.tensor([
    [2.0, 1.0, 0.1],  # 样本 1
    [0.5, 2.5, 1.0]   # 样本 2
])

probs = torch.softmax(batch_logits, dim=1)
print(f"\n批量 Softmax:\n{probs}")
# tensor([[0.6590, 0.2424, 0.0986],  <- 样本 1 的概率分布
#         [0.1142, 0.8438, 0.1884]]) <- 样本 2 的概率分布
```

### 9.3 完整的多分类训练循环

```python
# ==================== 多分类训练循环 ====================
torch.manual_seed(42)
epochs = 100

# 移动数据到设备
X_blob_train = X_blob_train.to(device)
y_blob_train = y_blob_train.to(device)
X_blob_test = X_blob_test.to(device)
y_blob_test = y_blob_test.to(device)

for epoch in range(epochs):
    # ========== 训练阶段 ==========
    model_4.train()
    
    # 1. 前向传播（输出 logits）
    y_logits = model_4(X_blob_train)
    
    # 将 logits 转换为预测标签
    # softmax → 概率分布
    # argmax → 选择概率最大的类别
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    # 2. 计算损失和准确率
    # CrossEntropyLoss 直接使用 logits（内置 softmax）
    loss = loss_fn_multi(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)
    
    # 3. 清零梯度
    optimizer.zero_grad()
    
    # 4. 反向传播
    loss.backward()
    
    # 5. 更新参数
    optimizer.step()
    
    # ========== 测试阶段 ==========
    model_4.eval()
    
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        
        test_loss = loss_fn_multi(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_pred)
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch:3d} | "
              f"Train Loss: {loss:.4f}, Train Acc: {acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# ==================== 预期输出 ====================
# Epoch:   0 | Train Loss: 1.0432, Train Acc: 65.50% | Test Loss: 0.5786, Test Acc: 95.50%
# Epoch:  10 | Train Loss: 0.1440, Train Acc: 99.12% | Test Loss: 0.1304, Test Acc: 99.00%
# ...
# Epoch:  90 | Train Loss: 0.0330, Train Acc: 99.12% | Test Loss: 0.0242, Test Acc: 99.50%
# 
# 注意：多分类 Blob 数据可以用线性边界分割，所以准确率很高！
```

---

## 10. 分类评估指标

### 10.1 常用指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **准确率** | 正确数/总数 | 最直观的指标 |
| **精确率** | TP/(TP+FP) | 预测为正的准确性 |
| **召回率** | TP/(TP+FN) | 实际为正的覆盖率 |
| **F1 分数** | 2×P×R/(P+R) | 精确率和召回率的调和平均 |

### 10.2 使用 torchmetrics

```python
from torchmetrics import Accuracy

# 多分类准确率
accuracy = Accuracy(task='multiclass', num_classes=4).to(device)
acc = accuracy(y_preds, y_test)
print(f"准确率: {acc:.4f}")
```

---

## 总结

### 二分类 vs 多分类对比

| 特性 | 二分类 | 多分类 |
|-----|--------|--------|
| 类别数 | 2 | > 2 |
| 输出层大小 | 1 | 类别数 |
| 输出激活 | Sigmoid | Softmax |
| 损失函数 | BCEWithLogitsLoss | CrossEntropyLoss |
| 标签格式 | Float [0, 1] | Long [0, C-1] |
| 预测方式 | round(sigmoid(logits)) | argmax(softmax(logits)) |

### 关键要点

1. **形状很重要**：始终检查输入输出形状
2. **非线性是关键**：使用 ReLU 等激活函数处理非线性数据
3. **损失函数选择**：二分类用 BCEWithLogitsLoss，多分类用 CrossEntropyLoss
4. **Logits 转换**：二分类用 Sigmoid + Round，多分类用 Softmax + Argmax
5. **设备管理**：使用设备无关代码，确保数据和模型在同一设备上

---

## 11. 完整实战代码

### 11.1 二分类完整示例

```python
"""
PyTorch 二分类完整示例 - 圆形数据分类
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# ==================== 1. 准备数据 ====================
n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

# 转换为张量
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==================== 2. 设置设备 ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# ==================== 3. 构建模型 ====================
class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

model = CircleModel().to(device)

# ==================== 4. 损失函数和优化器 ====================
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    return (torch.eq(y_true, y_pred).sum().item() / len(y_pred)) * 100

# ==================== 5. 训练循环 ====================
torch.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    model.train()
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    loss = loss_fn(y_logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test).squeeze()
            test_acc = accuracy_fn(y_test, torch.round(torch.sigmoid(test_logits)))
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Test Acc: {test_acc:.2f}%")

print("✅ 二分类训练完成!")
```

### 11.2 多分类完整示例

```python
"""
PyTorch 多分类完整示例 - Blob 数据分类
"""
import torch
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# ==================== 1. 准备数据 ====================
NUM_CLASSES = 4
X_blob, y_blob = make_blobs(n_samples=1000, n_features=2, centers=NUM_CLASSES, 
                            cluster_std=1.5, random_state=42)

X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

X_train, X_test, y_train, y_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=42)

# ==================== 2. 设置设备 ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# ==================== 3. 构建模型 ====================
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features)
        )
    
    def forward(self, x):
        return self.layers(x)

model = BlobModel(input_features=2, output_features=NUM_CLASSES).to(device)

# ==================== 4. 损失函数和优化器 ====================
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    return (torch.eq(y_true, y_pred).sum().item() / len(y_pred)) * 100

# ==================== 5. 训练循环 ====================
torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model.train()
    y_logits = model(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    loss = loss_fn(y_logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_acc = accuracy_fn(y_test, test_pred)
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Test Acc: {test_acc:.2f}%")

print("✅ 多分类训练完成!")
```

---

## 12. 常见问题与解决方案

### 12.1 模型准确率只有 50%（随机猜测）

**原因**: 使用纯线性模型处理非线性数据

**解决方案**: 添加非线性激活函数（如 ReLU）

```python
# 错误：纯线性模型
self.layer_2(self.layer_1(x))

# 正确：添加 ReLU
self.layer_2(self.relu(self.layer_1(x)))
```

### 12.2 形状不匹配错误

**原因**: 输入输出维度不一致

**解决方案**: 
- 检查 `in_features` 和 `out_features`
- 使用 `.squeeze()` 去除多余维度
- 确保标签格式正确（二分类用 Float，多分类用 Long）

### 12.3 损失不下降

**可能原因**:
1. 学习率太小或太大
2. 模型容量不足
3. 数据问题

**解决方案**:
- 尝试不同的学习率
- 增加隐藏层或神经元
- 检查数据预处理

---

## 参考资源

- [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 损失函数](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [TorchMetrics 文档](https://torchmetrics.readthedocs.io/)
