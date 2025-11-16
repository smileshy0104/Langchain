# PyTorch 基础 - 完整代码示例集合

> 本文档包含 PyTorch 基础教程中的所有重要代码示例
> 原始教程: https://www.learnpytorch.io/00_pytorch_fundamentals/

## 目录

- [1. 导入和版本检查](#1-导入和版本检查)
- [2. 创建张量](#2-创建张量)
- [3. 张量属性和信息](#3-张量属性和信息)
- [4. 张量运算](#4-张量运算)
- [5. 矩阵乘法](#5-矩阵乘法)
- [6. 聚合操作](#6-聚合操作)
- [7. 重塑和变换](#7-重塑和变换)
- [8. 索引](#8-索引)
- [9. NumPy 互操作](#9-numpy-互操作)
- [10. 随机种子](#10-随机种子)
- [11. GPU 操作](#11-gpu-操作)

---

## 1. 导入和版本检查

```python
# 导入 PyTorch
import torch

# 检查 PyTorch 版本
print(torch.__version__)
# 输出示例: '1.13.1+cu116'
```

---

## 2. 创建张量

### 2.1 创建标量

```python
# 标量 - 0维张量
scalar = torch.tensor(7)
print(scalar)
# 输出: tensor(7)

# 检查维度
print(scalar.ndim)
# 输出: 0

# 获取 Python 数值
print(scalar.item())
# 输出: 7
```

### 2.2 创建向量

```python
# 向量 - 1维张量
vector = torch.tensor([7, 7])
print(vector)
# 输出: tensor([7, 7])

# 检查维度
print(vector.ndim)
# 输出: 1

# 检查形状
print(vector.shape)
# 输出: torch.Size([2])
```

### 2.3 创建矩阵

```python
# 矩阵 - 2维张量
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(MATRIX)
# 输出:
# tensor([[ 7,  8],
#         [ 9, 10]])

print(MATRIX.ndim)    # 输出: 2
print(MATRIX.shape)   # 输出: torch.Size([2, 2])
print(MATRIX[0])      # 输出: tensor([7, 8])
print(MATRIX[1])      # 输出: tensor([9, 10])
```

### 2.4 创建张量

```python
# 张量 - n维数组
TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])

print(TENSOR.ndim)    # 输出: 3
print(TENSOR.shape)   # 输出: torch.Size([1, 3, 3])
print(TENSOR[0])      # 输出: 第一个3x3矩阵
```

### 2.5 随机张量

```python
# 创建随机张量
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.shape)   # torch.Size([3, 4])
print(random_tensor.ndim)    # 2

# 创建类似图像的随机张量 [height, width, color_channels]
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape)   # torch.Size([224, 224, 3])
print(random_image_size_tensor.ndim)    # 3
```

### 2.6 全零和全一张量

```python
# 全零张量
zeros = torch.zeros(size=(3, 4))
print(zeros)
print(zeros * random_tensor)  # 乘以0得到0

# 全一张量
ones = torch.ones(size=(3, 4))
print(ones)
print(ones.dtype)  # torch.float32

# 创建特定数据类型的张量
ones_int = torch.ones(size=(3, 4), dtype=torch.int32)
print(ones_int.dtype)  # torch.int32
```

### 2.7 范围张量

```python
# 创建范围张量
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)
# 输出: tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 简化写法
zero_to_ten = torch.arange(0, 10)
print(zero_to_ten)
```

### 2.8 创建类似形状的张量

```python
# 创建形状相同的全零张量
ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros)
# 输出: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# 创建形状相同的全一张量
ten_ones = torch.ones_like(input=zero_to_ten)
print(ten_ones)
# 输出: tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

---

## 3. 张量属性和信息

```python
# 创建示例张量
some_tensor = torch.rand(3, 4)

# 获取张量信息
print(f"张量: \n{some_tensor}\n")
print(f"数据类型: {some_tensor.dtype}")
print(f"形状: {some_tensor.shape}")
print(f"维度: {some_tensor.ndim}")
print(f"设备: {some_tensor.device}")
print(f"元素总数: {some_tensor.numel()}")

# 输出示例:
# 数据类型: torch.float32
# 形状: torch.Size([3, 4])
# 维度: 2
# 设备: cpu
# 元素总数: 12
```

### 张量数据类型

```python
# 默认数据类型
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,  # 默认为 float32
                               device=None,  # 默认为 CPU
                               requires_grad=False)  # 是否跟踪梯度

print(float_32_tensor.dtype)  # torch.float32

# 创建 float16 张量
float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16)
print(float_16_tensor.dtype)  # torch.float16

# 创建 int8 张量
int_8_tensor = torch.tensor([3, 6, 9],
                            dtype=torch.int8)
print(int_8_tensor.dtype)  # torch.int8
```

---

## 4. 张量运算

### 4.1 基本算术运算

```python
# 创建张量
tensor = torch.tensor([1, 2, 3])

# 加法
print(tensor + 10)     # tensor([11, 12, 13])
print(torch.add(tensor, 10))  # 等同于上面

# 减法
print(tensor - 10)     # tensor([-9, -8, -7])
print(torch.subtract(tensor, 10))

# 乘法
print(tensor * 10)     # tensor([10, 20, 30])
print(torch.multiply(tensor, 10))

# 除法
print(tensor / 10)     # tensor([0.1000, 0.2000, 0.3000])
print(torch.divide(tensor, 10))
```

### 4.2 元素级乘法

```python
# 元素级乘法
tensor = torch.tensor([1, 2, 3])
print(tensor * tensor)
# 输出: tensor([1, 4, 9])

# 等同于
print(torch.mul(tensor, tensor))
```

---

## 5. 矩阵乘法

### 5.1 基本矩阵乘法

```python
import torch

tensor = torch.tensor([1, 2, 3])

# 矩阵乘法(点积)
print(torch.matmul(tensor, tensor))
# 输出: tensor(14)  # 1*1 + 2*2 + 3*3

# 等同写法
print(tensor @ tensor)
```

### 5.2 矩阵乘法规则演示

```python
# 创建两个矩阵
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])  # 形状: [3, 2]

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])  # 形状: [3, 2]

# 这会出错,因为内部维度不匹配
# torch.matmul(tensor_A, tensor_B)
# RuntimeError: mat1 and mat2 shapes cannot be multiplied

# 转置 tensor_B
print(tensor_B.T)
# 输出: 形状变为 [2, 3]

# 现在可以相乘
result = torch.matmul(tensor_A, tensor_B.T)
print(result)
print(result.shape)  # torch.Size([3, 3])
```

### 5.3 矩阵乘法的不同方法

```python
# 方法 1: torch.matmul()
result1 = torch.matmul(tensor_A, tensor_B.T)

# 方法 2: torch.mm() (仅适用于2D张量)
result2 = torch.mm(tensor_A, tensor_B.T)

# 方法 3: @ 运算符
result3 = tensor_A @ tensor_B.T

# 三种方法结果相同
print(result1 == result2)
print(result2 == result3)
```

### 5.4 神经网络线性层

```python
# 模拟神经网络的线性层
torch.manual_seed(42)

# 创建线性层: y = x·A^T + b
linear = torch.nn.Linear(in_features=2,  # 输入特征数
                         out_features=6)  # 输出特征数

# 输入
x = torch.tensor([[1., 2.]])
print(x.shape)  # torch.Size([1, 2])

# 前向传播
output = linear(x)
print(output)
print(output.shape)  # torch.Size([1, 6])
```

---

## 6. 聚合操作

### 6.1 最小值、最大值、平均值、总和

```python
# 创建张量
x = torch.arange(0, 100, 10)
print(x)
# tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# 最小值
print(torch.min(x))  # tensor(0)
print(x.min())       # tensor(0)

# 最大值
print(torch.max(x))  # tensor(90)
print(x.max())       # tensor(90)

# 平均值 (需要 float 类型)
print(torch.mean(x.type(torch.float32)))  # tensor(45.)
print(x.type(torch.float32).mean())

# 总和
print(torch.sum(x))  # tensor(450)
print(x.sum())       # tensor(450)
```

### 6.2 位置最小值/最大值

```python
# 查找最大值的索引位置
print(x.argmax())  # tensor(9)

# 查找最小值的索引位置
print(x.argmin())  # tensor(0)

# 验证
print(x[9])  # tensor(90) - 最大值
print(x[0])  # tensor(0)  - 最小值
```

---

## 7. 重塑和变换

### 7.1 Reshape

```python
# 创建张量
x = torch.arange(1., 10.)
print(x)
print(x.shape)  # torch.Size([9])

# 重塑为 3x3
x_reshaped = x.reshape(3, 3)
print(x_reshaped)
print(x_reshaped.shape)  # torch.Size([3, 3])
```

### 7.2 View

```python
# View 创建张量的视图(共享内存)
z = x.view(3, 3)
print(z)
print(z.shape)  # torch.Size([3, 3])

# 修改 z 会影响 x
z[0, 0] = 5
print(z)
print(x)  # x 也被修改了!
```

### 7.3 Stack

```python
# 堆叠张量
x = torch.arange(1., 10.)

# 在新维度上堆叠
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked)
print(x_stacked.shape)  # torch.Size([4, 9])

# 在不同维度堆叠
x_stacked_dim1 = torch.stack([x, x, x, x], dim=1)
print(x_stacked_dim1.shape)  # torch.Size([9, 4])
```

### 7.4 Squeeze 和 Unsqueeze

```python
# Squeeze - 移除维度为1的维度
x = torch.zeros(2, 1, 2, 1, 2)
print(x.shape)  # torch.Size([2, 1, 2, 1, 2])

x_squeezed = x.squeeze()
print(x_squeezed.shape)  # torch.Size([2, 2, 2])

# Unsqueeze - 添加维度为1的维度
x = torch.arange(1., 10.)
print(x.shape)  # torch.Size([9])

x_unsqueezed = x.unsqueeze(dim=0)
print(x_unsqueezed.shape)  # torch.Size([1, 9])

x_unsqueezed = x.unsqueeze(dim=1)
print(x_unsqueezed.shape)  # torch.Size([9, 1])
```

### 7.5 Permute

```python
# Permute - 重新排列维度
x_original = torch.rand(size=(224, 224, 3))  # [H, W, C]

# 改变维度顺序
x_permuted = x_original.permute(2, 0, 1)  # [C, H, W]
print(x_original.shape)  # torch.Size([224, 224, 3])
print(x_permuted.shape)  # torch.Size([3, 224, 224])
```

### 7.6 改变数据类型

```python
# 创建 float32 张量
tensor = torch.tensor([3., 6., 9.])
print(tensor.dtype)  # torch.float32

# 转换为 float16
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16.dtype)  # torch.float16

# 转换为 int8
tensor_int8 = tensor.type(torch.int8)
print(tensor_int8.dtype)  # torch.int8
print(tensor_int8)       # tensor([3, 6, 9], dtype=torch.int8)
```

---

## 8. 索引

### 8.1 基本索引

```python
# 创建张量
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)
print(x.shape)  # torch.Size([1, 3, 3])

# 索引第一个维度
print(x[0])
# 输出:
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])

# 索引到单个元素
print(x[0][0])      # tensor([1, 2, 3])
print(x[0][0][0])   # tensor(1)
print(x[0, 0, 0])   # tensor(1) - 更简洁的写法
```

### 8.2 使用冒号索引

```python
# 使用 : 选择所有
print(x[:, 0])
# 输出: tensor([[1, 2, 3]])

print(x[:, :, 0])
# 输出: tensor([[1, 4, 7]])

print(x[:, 1, 1])
# 输出: tensor([5])

# 获取所有值
print(x[:, :, :])  # 等同于 x
```

### 8.3 高级索引

```python
# 创建更大的张量
x = torch.arange(1, 25).reshape(2, 3, 4)
print(x.shape)  # torch.Size([2, 3, 4])

# 获取第一个批次
print(x[0])

# 获取第一个批次的第一行
print(x[0, 0])

# 获取特定元素
print(x[0, 0, 0])  # tensor(1)
print(x[1, 2, 3])  # tensor(24)
```

---

## 9. NumPy 互操作

### 9.1 NumPy 数组转 PyTorch 张量

```python
import torch
import numpy as np

# 创建 NumPy 数组
array = np.arange(1.0, 8.0)
print(array)
print(type(array))  # <class 'numpy.ndarray'>

# 转换为 PyTorch 张量
tensor = torch.from_numpy(array)
print(tensor)
print(type(tensor))  # <class 'torch.Tensor'>

# 检查数据类型
print(array.dtype)   # float64 (NumPy 默认)
print(tensor.dtype)  # torch.float64

# 转换为 float32
tensor_float32 = torch.from_numpy(array).type(torch.float32)
print(tensor_float32.dtype)  # torch.float32
```

### 9.2 PyTorch 张量转 NumPy 数组

```python
# 创建 PyTorch 张量
tensor = torch.ones(7)
print(tensor)
print(tensor.dtype)  # torch.float32

# 转换为 NumPy 数组
numpy_tensor = tensor.numpy()
print(numpy_tensor)
print(type(numpy_tensor))  # <class 'numpy.ndarray'>
print(numpy_tensor.dtype)  # float32
```

### 9.3 共享内存注意事项

```python
# 创建张量和数组
tensor = torch.arange(1., 8.)
numpy_array = tensor.numpy()

# 修改张量
tensor = tensor + 1

# numpy_array 不受影响(因为 tensor 被重新赋值)
print(tensor)       # tensor([2., 3., 4., 5., 6., 7., 8.])
print(numpy_array)  # [1. 2. 3. 4. 5. 6. 7.]
```

---

## 10. 随机种子

### 10.1 设置随机种子

```python
import torch

# 不设置种子 - 每次都不同
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(random_tensor_A == random_tensor_B)  # 大部分为 False

# 设置随机种子
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C == random_tensor_D)  # 全部为 True!
```

### 10.2 GPU 随机种子

```python
# 为 GPU 设置随机种子
torch.cuda.manual_seed(42)

# 为所有 GPU 设置随机种子
torch.cuda.manual_seed_all(42)
```

---

## 11. GPU 操作

### 11.1 检查 GPU 可用性

```python
import torch

# 检查 CUDA (NVIDIA GPU) 是否可用
print(torch.cuda.is_available())  # True 或 False

# 获取 CUDA 版本
if torch.cuda.is_available():
    print(torch.version.cuda)

# 获取可用 GPU 数量
print(torch.cuda.device_count())  # 0, 1, 2, ...

# 获取当前 GPU 名称
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### 11.2 设备无关代码

```python
# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 或者更详细的版本
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():  # Apple Silicon
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
```

### 11.3 将张量移到 GPU

```python
# 创建张量
tensor = torch.tensor([1, 2, 3])
print(f"Tensor: {tensor}")
print(f"Device: {tensor.device}")  # cpu

# 移到 GPU
if torch.cuda.is_available():
    tensor_on_gpu = tensor.to("cuda")
    print(f"Tensor on GPU: {tensor_on_gpu}")
    print(f"Device: {tensor_on_gpu.device}")  # cuda:0

# 使用设备变量
tensor_on_device = tensor.to(device)
print(tensor_on_device.device)
```

### 11.4 GPU 上的张量运算

```python
if torch.cuda.is_available():
    # 在 GPU 上创建张量
    tensor_A = torch.tensor([1, 2, 3], device="cuda")
    tensor_B = torch.tensor([4, 5, 6], device="cuda")

    # GPU 上的运算
    result = tensor_A + tensor_B
    print(result)
    print(result.device)  # cuda:0
```

### 11.5 将张量移回 CPU

```python
if torch.cuda.is_available():
    # GPU 张量
    tensor_on_gpu = torch.tensor([1, 2, 3], device="cuda")

    # 移回 CPU
    tensor_on_cpu = tensor_on_gpu.cpu()
    print(tensor_on_cpu.device)  # cpu

    # 现在可以用于 NumPy
    numpy_array = tensor_on_cpu.numpy()
    print(type(numpy_array))  # <class 'numpy.ndarray'>
```

### 11.6 Apple Silicon (MPS)

```python
# 检查 MPS 可用性
print(torch.backends.mps.is_available())

# 设置 MPS 设备
if torch.backends.mps.is_available():
    device = "mps"
    tensor = torch.tensor([1, 2, 3], device=device)
    print(tensor.device)  # mps:0
```

---

## 完整示例:综合应用

### 示例 1: 创建和操作张量

```python
import torch

# 设置随机种子
torch.manual_seed(42)

# 创建随机张量
A = torch.rand(3, 4)
B = torch.rand(3, 4)

print(f"Tensor A:\n{A}\n")
print(f"Tensor B:\n{B}\n")

# 元素级操作
print(f"A + B:\n{A + B}\n")
print(f"A * B:\n{A * B}\n")

# 矩阵乘法
print(f"A @ B.T:\n{torch.matmul(A, B.T)}\n")

# 聚合
print(f"A mean: {A.mean()}")
print(f"A max: {A.max()}")
print(f"A min: {A.min()}")
```

### 示例 2: 图像数据处理

```python
import torch

# 模拟图像数据 [batch, height, width, channels]
images = torch.rand(32, 224, 224, 3)

print(f"Images shape: {images.shape}")
print(f"Images dtype: {images.dtype}")
print(f"Images device: {images.device}")

# 转换为 [batch, channels, height, width] (PyTorch 标准格式)
images_permuted = images.permute(0, 3, 1, 2)
print(f"Permuted shape: {images_permuted.shape}")

# 计算每个通道的平均值
channel_means = images_permuted.mean(dim=[0, 2, 3])
print(f"Channel means: {channel_means}")
```

### 示例 3: 简单神经网络前向传播

```python
import torch
import torch.nn as nn

# 设置随机种子
torch.manual_seed(42)

# 创建简单的神经网络
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# 创建输入数据
x = torch.rand(1, 10)

# 前向传播
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output: {output}")
print(f"Output shape: {output.shape}")
```

### 示例 4: 批量处理和设备管理

```python
import torch

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 创建批量数据
batch_size = 32
input_features = 10
output_features = 5

# 创建数据和模型
X = torch.rand(batch_size, input_features).to(device)
model = torch.nn.Linear(input_features, output_features).to(device)

# 前向传播
with torch.no_grad():  # 不计算梯度
    predictions = model(X)

print(f"Input shape: {X.shape}")
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions device: {predictions.device}")

# 移回 CPU 用于进一步处理
predictions_cpu = predictions.cpu()
print(f"Predictions on CPU: {predictions_cpu.shape}")
```

---

## 常用代码片段

### 检查张量信息

```python
def print_tensor_info(tensor, name="Tensor"):
    """打印张量的详细信息"""
    print(f"\n{name} Information:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Requires grad: {tensor.requires_grad}")
    print(f"  Min: {tensor.min()}")
    print(f"  Max: {tensor.max()}")
    if tensor.dtype in [torch.float32, torch.float64]:
        print(f"  Mean: {tensor.mean()}")
    print(f"  First few elements: {tensor.flatten()[:5]}")

# 使用示例
tensor = torch.rand(3, 4)
print_tensor_info(tensor, "Random Tensor")
```

### 设置随机种子函数

```python
def set_seeds(seed=42):
    """设置所有随机种子以保证可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    # 可选: 设置 NumPy 种子
    import numpy as np
    np.random.seed(seed)

# 使用
set_seeds(42)
```

### 设备管理函数

```python
def get_device():
    """获取最佳可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# 使用
device = get_device()
print(f"Using device: {device}")

# 移动张量到设备
tensor = torch.rand(3, 4).to(device)
```

---

## 调试技巧

### 检查形状匹配

```python
# 矩阵乘法前检查形状
def check_matmul_shapes(A, B):
    """检查两个张量是否可以进行矩阵乘法"""
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")

    if A.shape[-1] != B.shape[-2]:
        print(f"❌ Cannot multiply: inner dimensions don't match")
        print(f"   {A.shape[-1]} != {B.shape[-2]}")
        return False
    else:
        result_shape = (*A.shape[:-1], B.shape[-1])
        print(f"✓ Can multiply! Result shape will be: {result_shape}")
        return True

# 使用
A = torch.rand(3, 4)
B = torch.rand(4, 5)
check_matmul_shapes(A, B)
```

### 梯度检查

```python
# 检查张量是否需要梯度
def check_requires_grad(tensor, name="Tensor"):
    """检查张量的梯度设置"""
    print(f"{name}:")
    print(f"  Requires grad: {tensor.requires_grad}")
    if tensor.grad is not None:
        print(f"  Has grad: True")
        print(f"  Grad shape: {tensor.grad.shape}")
    else:
        print(f"  Has grad: False")

# 使用
x = torch.rand(3, 4, requires_grad=True)
check_requires_grad(x, "Input tensor")
```

---

## 性能优化提示

### 1. 使用 in-place 操作

```python
# 常规操作(创建新张量)
x = torch.rand(1000, 1000)
y = x + 1

# In-place 操作(修改现有张量,更快)
x.add_(1)  # 注意下划线
```

### 2. 使用 torch.no_grad()

```python
# 推理时不需要梯度
model = torch.nn.Linear(10, 5)
x = torch.rand(1, 10)

with torch.no_grad():
    output = model(x)
# 更快,内存占用更少
```

### 3. 批量操作

```python
# 慢 - 逐个处理
results = []
for i in range(100):
    x = torch.rand(1, 10)
    result = x * 2
    results.append(result)

# 快 - 批量处理
x_batch = torch.rand(100, 10)
results_batch = x_batch * 2
```

---

**文档更新**: 2025-11-16
**PyTorch 版本**: 1.10.0+
