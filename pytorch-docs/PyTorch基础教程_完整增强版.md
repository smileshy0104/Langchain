# PyTorch 基础教程 - 完整增强版

> **本文档整合了原始教程、2024年最新最佳实践和深度技术细节**

---

## 📚 文档信息

**原始来源:** https://www.learnpytorch.io/00_pytorch_fundamentals/
**GitHub仓库:** https://github.com/mrdbourke/pytorch-deep-learning
**文档版本:** v2.0 (增强版)
**更新日期:** 2025-11-16
**适用 PyTorch 版本:** 1.10.0+

**增强内容包括:**
- ✅ 广播(Broadcasting)机制详解
- ✅ 自动微分(Autograd)原理与实践
- ✅ GPU 内存管理最佳实践
- ✅ 性能优化技巧 (2024)
- ✅ 高级张量操作
- ✅ 常见问题与调试指南
- ✅ 实战项目示例

---

## 📖 目录

### 第一部分：基础知识
1. [什么是 PyTorch](#1-什么是-pytorch)
2. [PyTorch 的应用](#2-pytorch-的应用)
3. [为什么使用 PyTorch](#3-为什么使用-pytorch)
4. [张量(Tensor)简介](#4-张量tensor简介)
5. [创建张量](#5-创建张量)

### 第二部分：核心操作
6. [张量操作](#6-张量操作)
7. [矩阵乘法](#7-矩阵乘法)
8. [张量聚合操作](#8-张量聚合操作)
9. [张量重塑与变换](#9-张量重塑与变换)
10. [索引操作](#10-索引操作)

### 第三部分：高级主题
11. [广播机制详解](#11-广播机制详解) ⭐ **新增**
12. [自动微分 Autograd](#12-自动微分-autograd) ⭐ **新增**
13. [PyTorch 与 NumPy](#13-pytorch-与-numpy)
14. [可重复性(Reproducibility)](#14-可重复性reproducibility)

### 第四部分：GPU 加速
15. [在 GPU 上运行](#15-在-gpu-上运行)
16. [GPU 内存管理](#16-gpu-内存管理) ⭐ **新增**
17. [性能优化技巧](#17-性能优化技巧) ⭐ **新增**

### 第五部分：实战与进阶
18. [常见问题与调试](#18-常见问题与调试) ⭐ **新增**
19. [实战项目示例](#19-实战项目示例) ⭐ **新增**
20. [练习与资源](#20-练习与资源)

---

# 第一部分：基础知识

## 1. 什么是 PyTorch

### 1.1 定义

**PyTorch** 是一个开源的机器学习和深度学习框架,由 Facebook AI Research (现 Meta AI) 开发。

### 1.2 主要特点

| 特点 | 说明 | 优势 |
|------|------|------|
| **Python 原生** | 完全基于 Python | 易于学习和使用 |
| **动态计算图** | Define-by-Run | 灵活,易于调试 |
| **自动微分** | Autograd 引擎 | 自动计算梯度 |
| **GPU 加速** | CUDA 支持 | 训练速度快 |
| **生态丰富** | 大量库和工具 | 适合研究和生产 |

### 1.3 PyTorch vs 其他框架

```python
# PyTorch - 动态图,直观
for epoch in range(epochs):
    output = model(x)  # 定义即运行
    loss = criterion(output, y)
    loss.backward()  # 动态构建计算图
    optimizer.step()

# TensorFlow 1.x - 静态图,需要先定义
# graph = tf.Graph()
# with graph.as_default():
#     x = tf.placeholder(...)
#     y = tf.placeholder(...)
#     loss = ...
# with tf.Session(graph=graph) as sess:
#     sess.run(...)
```

### 1.4 谁在使用 PyTorch

**科技公司:**
- **Meta (Facebook)** - 推荐系统、内容理解
- **Tesla** - 自动驾驶(Autopilot 和 FSD)
- **Microsoft** - Azure ML 服务
- **OpenAI** - GPT 系列模型
- **Uber** - Pyro (概率编程)

**研究机构:**
- Stanford, MIT, CMU 等顶级高校
- DeepMind, Google Brain 的部分研究

**统计数据 (2024):**
- Papers with Code: 70%+ 的论文使用 PyTorch
- GitHub: 60,000+ star
- 下载量: 每月超过 1000 万次

![PyTorch在工业和研究中的应用](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-pytorch-being-used-across-research-and-industry.png)

---

## 2. PyTorch 的应用

### 2.1 计算机视觉 (Computer Vision)

```python
# 图像分类
import torch
import torchvision

# 预训练模型
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# 加载图像
from PIL import Image
from torchvision import transforms

image = Image.open('cat.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# 推理
with torch.no_grad():
    output = model(input_batch)

# 获取预测类别
_, predicted_idx = torch.max(output, 1)
print(f"Predicted class: {predicted_idx.item()}")
```

**应用场景:**
- 图像分类 (ImageNet, CIFAR-10)
- 目标检测 (YOLO, Faster R-CNN)
- 语义分割 (U-Net, DeepLab)
- 人脸识别
- 医学图像分析

### 2.2 自然语言处理 (NLP)

```python
# 文本生成 (使用 Hugging Face Transformers)
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 生成文本
input_text = "PyTorch is"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 生成
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

**应用场景:**
- 语言模型 (GPT, BERT)
- 机器翻译
- 问答系统
- 情感分析
- 文本摘要

### 2.3 语音识别 (Speech Recognition)

```python
# 使用 torchaudio
import torchaudio

# 加载音频
waveform, sample_rate = torchaudio.load("speech.wav")

# 提取 MFCC 特征
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=sample_rate,
    n_mfcc=13
)

mfcc = mfcc_transform(waveform)
print(f"MFCC shape: {mfcc.shape}")
```

**应用场景:**
- 语音识别 (ASR)
- 语音合成 (TTS)
- 声纹识别
- 音乐生成

### 2.4 强化学习 (Reinforcement Learning)

```python
# Deep Q-Learning 示例
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)

# 创建模型
state_dim = 4  # CartPole 环境
action_dim = 2
model = DQN(state_dim, action_dim)

# 选择动作
state = torch.randn(1, state_dim)
q_values = model(state)
action = q_values.argmax(dim=1)
print(f"Selected action: {action.item()}")
```

**应用场景:**
- 游戏 AI (AlphaGo, Dota 2)
- 机器人控制
- 自动驾驶
- 资源调度

### 2.5 生成模型 (Generative Models)

```python
# GAN 生成器示例
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
```

**应用场景:**
- 图像生成 (StyleGAN, DALL-E)
- 文本生成 (GPT)
- 音乐生成
- 视频生成

---

## 3. 为什么使用 PyTorch

### 3.1 主要优势

#### 1. **研究者的首选**

```
Papers with Code 框架趋势 (2024):
┌─────────────────────────────────────┐
│ PyTorch:  ████████████████████ 72% │
│ TensorFlow: ████████ 18%            │
│ JAX:       ████ 7%                  │
│ 其他:      █ 3%                     │
└─────────────────────────────────────┘
```

**原因:**
- 代码简洁直观
- 调试容易 (Python 原生调试器)
- 快速原型开发
- 社区活跃

#### 2. **Pythonic 设计**

```python
# PyTorch 代码读起来像 Python
import torch

# 创建张量
x = torch.tensor([1, 2, 3])

# 操作直观
y = x + 10
z = x * 2

# 控制流自然
for i in range(len(x)):
    if x[i] > 1:
        x[i] = x[i] ** 2

print(x)  # tensor([1, 4, 9])
```

#### 3. **动态计算图**

```python
# 动态图的优势: 可以使用 Python 控制流

def forward(x, condition):
    if condition:
        # 分支 A
        return x * 2
    else:
        # 分支 B
        return x + 10

x = torch.tensor([5.0], requires_grad=True)

# 运行时决定路径
output = forward(x, condition=True)
output.backward()
print(x.grad)  # tensor([2.])  # 对应 x*2 的梯度

# 改变条件
x.grad.zero_()
output = forward(x, condition=False)
output.backward()
print(x.grad)  # tensor([1.])  # 对应 x+10 的梯度
```

**对比静态图:**
- TensorFlow 1.x 需要预先定义整个图
- 无法在运行时改变网络结构
- 调试困难

#### 4. **强大的 GPU 加速**

```python
# 自动处理 GPU 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型和数据移到 GPU (一行代码)
model = MyModel().to(device)
data = data.to(device)

# PyTorch 自动在 GPU 上执行
output = model(data)  # 在 GPU 上运行

# 性能对比 (示例)
# CPU: 100秒/epoch
# GPU: 5秒/epoch  (20x 加速!)
```

#### 5. **生产就绪**

```python
# TorchScript: 将模型导出为优化的格式
import torch.jit

# 方法 1: Tracing
model = MyModel()
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# 保存
traced_model.save("model.pt")

# 加载 (C++ 环境也可以加载)
loaded_model = torch.jit.load("model.pt")

# 方法 2: Scripting (支持控制流)
@torch.jit.script
def my_function(x, y):
    if x.sum() > 0:
        return x + y
    else:
        return x - y
```

**生产部署工具:**
- **TorchScript**: 模型序列化和优化
- **TorchServe**: 模型服务框架
- **ONNX**: 跨框架模型交换格式
- **Mobile**: iOS/Android 部署

#### 6. **丰富的生态系统**

```
PyTorch 生态系统:
├── torchvision     (计算机视觉)
├── torchtext       (自然语言处理)
├── torchaudio      (音频处理)
├── torchmetrics    (评估指标)
├── pytorch-lightning (高层API)
├── fastai          (快速开发)
├── transformers    (预训练模型)
├── detectron2      (目标检测)
└── ...
```

### 3.2 PyTorch 2.0 新特性 (2024)

```python
# torch.compile - 图编译加速
import torch

model = MyModel()

# 编译模型 (简单一行)
compiled_model = torch.compile(model)

# 使用编译后的模型 (API 不变)
output = compiled_model(input)

# 性能提升:
# - 训练速度: 1.3-2x
# - 推理速度: 1.5-3x
```

**新特性:**
- **torch.compile()**: 自动优化模型
- **改进的分布式训练**: FSDP, DDP 优化
- **更好的 AMD GPU 支持**
- **Metal (Apple Silicon) 性能提升**

---

## 4. 张量(Tensor)简介

### 4.1 什么是张量?

**张量是机器学习和深度学习的基本构建块。**

**数学定义:** 张量是多维数组,是标量、向量、矩阵的推广。

**直观理解:**
```
标量(0D) → 向量(1D) → 矩阵(2D) → 张量(3D+)
   7    →  [7,7]  →  [[7,8],  → [[[1,2,3],
                       [9,10]]    [4,5,6]],
                                 [[7,8,9],
                                  [0,1,2]]]
```

### 4.2 张量的作用

**核心思想:** 将现实世界的数据转换为数字,以便计算机处理。

| 数据类型 | 张量表示 | 形状示例 |
|---------|---------|---------|
| **数字** | 标量 | `()` |
| **时间序列** | 向量 | `(100,)` |
| **灰度图像** | 矩阵 | `(28, 28)` |
| **彩色图像** | 3D 张量 | `(3, 224, 224)` |
| **视频** | 4D 张量 | `(30, 3, 224, 224)` |
| **批量图像** | 4D 张量 | `(32, 3, 224, 224)` |

### 4.3 张量维度详解

#### 维度 (Dimension) vs 轴 (Axis)

```python
import torch

# 3D 张量
tensor_3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print(f"维度数 (ndim): {tensor_3d.ndim}")  # 3
print(f"形状 (shape): {tensor_3d.shape}")  # torch.Size([2, 2, 2])

# 理解:
# - 维度数 = 3 (3D 张量)
# - 轴 0 大小 = 2
# - 轴 1 大小 = 2
# - 轴 2 大小 = 2
```

**可视化:**
```
tensor_3d.shape = (2, 2, 2)
                   ↑  ↑  ↑
                   │  │  └─ 轴 2 (最内层)
                   │  └──── 轴 1 (中间层)
                   └─────── 轴 0 (最外层)
```

### 4.4 图像到张量的转换

![图像到张量的转换示例](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-tensor-shape-example-of-image.png)

```python
from PIL import Image
import torchvision.transforms as transforms

# 加载图像
image = Image.open("cat.jpg")  # RGB 图像

# 转换为张量
to_tensor = transforms.ToTensor()
tensor = to_tensor(image)

print(f"形状: {tensor.shape}")  # torch.Size([3, 224, 224])
print(f"数据类型: {tensor.dtype}")  # torch.float32
print(f"值范围: [{tensor.min()}, {tensor.max()}]")  # [0.0, 1.0]

# 解释形状:
# [3, 224, 224]
#  ↑   ↑    ↑
#  │   │    └─ 宽度 (像素)
#  │   └────── 高度 (像素)
#  └────────── 颜色通道 (R, G, B)
```

**通道顺序:**
- PyTorch/torchvision: `(C, H, W)` - Channels First
- PIL/NumPy/Matplotlib: `(H, W, C)` - Channels Last

```python
# 转换通道顺序
tensor_chw = tensor  # (C, H, W)
tensor_hwc = tensor.permute(1, 2, 0)  # (H, W, C)

print(tensor_chw.shape)  # torch.Size([3, 224, 224])
print(tensor_hwc.shape)  # torch.Size([224, 224, 3])
```

### 4.5 张量类型总结

![不同张量维度示例](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-pytorch-different-tensor-dimensions.png)

| 名称 | 维度 | 形状示例 | 用途 | 代码示例 |
|------|------|---------|------|---------|
| **Scalar** | 0D | `()` | 单个值 | `loss = 0.5` |
| **Vector** | 1D | `(n,)` | 序列数据 | `embeddings = (768,)` |
| **Matrix** | 2D | `(m, n)` | 表格数据 | `weights = (512, 256)` |
| **3D Tensor** | 3D | `(a, b, c)` | 图像/视频帧 | `image = (3, 224, 224)` |
| **4D Tensor** | 4D | `(a, b, c, d)` | 批量图像 | `batch = (32, 3, 224, 224)` |
| **5D Tensor** | 5D | `(a, b, c, d, e)` | 批量视频 | `video_batch = (8, 30, 3, 224, 224)` |

![标量、向量、矩阵、张量示意图](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-scalar-vector-matrix-tensor.png)

### 4.6 判断张量维度的技巧

**方法 1: 数方括号**

```python
# 数左侧的 [ 数量
scalar = 7                    # 0 个 [ → 0D
vector = [7, 7]              # 1 个 [ → 1D
matrix = [[7, 8], [9, 10]]   # 2 个 [ → 2D
tensor = [[[1, 2]]]          # 3 个 [ → 3D
```

**方法 2: 使用 .ndim**

```python
import torch

x = torch.tensor([[[1, 2]]])
print(x.ndim)  # 3
```

**方法 3: 看形状长度**

```python
x = torch.rand(2, 3, 4, 5)
print(x.shape)  # torch.Size([2, 3, 4, 5])
print(len(x.shape))  # 4 → 4D 张量
```

---

## 5. 创建张量

### 5.1 从 Python 数据创建

#### 5.1.1 标量 (Scalar)

```python
import torch

# 创建标量
scalar = torch.tensor(7)

print(scalar)  # tensor(7)
print(scalar.ndim)  # 0
print(scalar.shape)  # torch.Size([])
print(scalar.item())  # 7 (提取 Python 数字)

# 标量的运算
a = torch.tensor(5)
b = torch.tensor(3)
c = a + b
print(c.item())  # 8
```

**注意:** `.item()` 只能用于单个元素的张量!

```python
vector = torch.tensor([1, 2, 3])
# vector.item()  # 错误! 多个元素无法使用 item()
```

#### 5.1.2 向量 (Vector)

```python
# 创建向量
vector = torch.tensor([7, 7])

print(vector)  # tensor([7, 7])
print(vector.ndim)  # 1
print(vector.shape)  # torch.Size([2])

# 访问元素
print(vector[0])  # tensor(7)
print(vector[0].item())  # 7

# 向量运算
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])

# 元素相加
print(v1 + v2)  # tensor([5, 7, 9])

# 点积
dot_product = torch.dot(v1.float(), v2.float())
print(dot_product)  # tensor(32.)  # 1*4 + 2*5 + 3*6
```

#### 5.1.3 矩阵 (Matrix)

```python
# 创建矩阵
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])

print(MATRIX)
# tensor([[ 7,  8],
#         [ 9, 10]])

print(MATRIX.ndim)  # 2
print(MATRIX.shape)  # torch.Size([2, 2])

# 访问元素
print(MATRIX[0])  # tensor([7, 8])  # 第一行
print(MATRIX[0, 1])  # tensor(8)  # 第一行第二列
print(MATRIX[:, 0])  # tensor([7, 9])  # 第一列

# 矩阵运算
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# 元素相乘
print(A * B)
# tensor([[ 5, 12],
#         [21, 32]])

# 矩阵乘法
print(torch.matmul(A, B))
# tensor([[19, 22],
#         [43, 50]])
```

#### 5.1.4 张量 (Tensor)

```python
# 创建 3D 张量
TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]],
                       [[10, 11, 12],
                        [13, 14, 15],
                        [16, 17, 18]]])

print(TENSOR)
print(TENSOR.ndim)  # 3
print(TENSOR.shape)  # torch.Size([2, 3, 3])

# 理解形状
print(f"轴 0 大小 (深度): {TENSOR.shape[0]}")  # 2
print(f"轴 1 大小 (行):   {TENSOR.shape[1]}")  # 3
print(f"轴 2 大小 (列):   {TENSOR.shape[2]}")  # 3

# 访问元素
print(TENSOR[0])  # 第一个 3x3 矩阵
print(TENSOR[0, 1])  # 第一个矩阵的第二行
print(TENSOR[0, 1, 2])  # 单个元素: tensor(6)
```

### 5.2 创建特殊张量

#### 5.2.1 随机张量

```python
# 均匀分布 [0, 1)
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.shape)  # torch.Size([3, 4])

# 标准正态分布 N(0, 1)
normal_tensor = torch.randn(3, 4)
print(normal_tensor)

# 随机整数 [low, high)
int_tensor = torch.randint(low=0, high=10, size=(3, 4))
print(int_tensor)

# 随机排列
perm = torch.randperm(10)
print(perm)  # tensor([3, 7, 1, 9, 0, 5, 2, 8, 4, 6])
```

**为什么需要随机张量?**

```python
# 神经网络初始化示例
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 权重随机初始化
        self.weights = torch.randn(10, 5)
        self.bias = torch.randn(5)

    def forward(self, x):
        return x @ self.weights + self.bias

model = SimpleNN()
print(model.weights)  # 随机初始化的权重
```

#### 5.2.2 全零和全一张量

```python
# 全零张量
zeros = torch.zeros(3, 4)
print(zeros)
# tensor([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])

# 全一张量
ones = torch.ones(3, 4)
print(ones)

# 指定数据类型
zeros_int = torch.zeros(3, 4, dtype=torch.int32)
print(zeros_int.dtype)  # torch.int32

# 创建与另一个张量形状相同的张量
x = torch.rand(2, 3)
zeros_like = torch.zeros_like(x)
ones_like = torch.ones_like(x)

print(zeros_like.shape)  # torch.Size([2, 3])
```

#### 5.2.3 范围张量

```python
# torch.arange(start, end, step)
range_tensor = torch.arange(0, 10, 1)
print(range_tensor)  # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 浮点数范围
float_range = torch.arange(0, 1, 0.1)
print(float_range)
# tensor([0.0000, 0.1000, 0.2000, ..., 0.9000])

# torch.linspace(start, end, steps) - 线性间隔
linspace_tensor = torch.linspace(0, 10, steps=5)
print(linspace_tensor)
# tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])

# torch.logspace(start, end, steps) - 对数间隔
logspace_tensor = torch.logspace(0, 2, steps=5)
print(logspace_tensor)
# tensor([  1.0000,   3.1623,  10.0000,  31.6228, 100.0000])
```

#### 5.2.4 对角张量和单位矩阵

```python
# 对角矩阵
diag = torch.diag(torch.tensor([1, 2, 3, 4]))
print(diag)
# tensor([[1, 0, 0, 0],
#         [0, 2, 0, 0],
#         [0, 0, 3, 0],
#         [0, 0, 0, 4]])

# 单位矩阵
identity = torch.eye(4)
print(identity)
# tensor([[1., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.],
#         [0., 0., 0., 1.]])
```

#### 5.2.5 常数张量

```python
# 填充指定值
full_tensor = torch.full((3, 4), fill_value=7.5)
print(full_tensor)
# tensor([[7.5000, 7.5000, 7.5000, 7.5000],
#         [7.5000, 7.5000, 7.5000, 7.5000],
#         [7.5000, 7.5000, 7.5000, 7.5000]])

# 复杂值填充
full_complex = torch.full((2, 2), 3+4j)
print(full_complex)
# tensor([[3.+4.j, 3.+4.j],
#         [3.+4.j, 3.+4.j]])
```

### 5.3 张量数据类型

#### 5.3.1 常用数据类型

| PyTorch 类型 | 等价 NumPy | 位数 | 范围/精度 |
|-------------|-----------|------|----------|
| `torch.float32` | `np.float32` | 32位 | 单精度浮点(默认) |
| `torch.float64` | `np.float64` | 64位 | 双精度浮点 |
| `torch.float16` | `np.float16` | 16位 | 半精度浮点 |
| `torch.bfloat16` | - | 16位 | Brain Float (Google TPU) |
| `torch.int64` | `np.int64` | 64位 | 长整型 |
| `torch.int32` | `np.int32` | 32位 | 整型 |
| `torch.int16` | `np.int16` | 16位 | 短整型 |
| `torch.int8` | `np.int8` | 8位 | 字节 |
| `torch.uint8` | `np.uint8` | 8位 | 无符号字节 |
| `torch.bool` | `np.bool_` | 1位 | 布尔值 |

#### 5.3.2 创建特定类型的张量

```python
# 默认类型 (float32)
default_tensor = torch.tensor([3.0, 6.0, 9.0])
print(default_tensor.dtype)  # torch.float32

# 指定类型
float16_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float16)
print(float16_tensor.dtype)  # torch.float16

# 整数
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
print(int_tensor.dtype)  # torch.int32

# 布尔
bool_tensor = torch.tensor([True, False, True], dtype=torch.bool)
print(bool_tensor.dtype)  # torch.bool

# 复数
complex_tensor = torch.tensor([1+2j, 3+4j], dtype=torch.complex64)
print(complex_tensor.dtype)  # torch.complex64
```

#### 5.3.3 类型转换

```python
# 创建 float32 张量
tensor = torch.tensor([3.0, 6.0, 9.0])
print(tensor.dtype)  # torch.float32

# 方法 1: .type()
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16.dtype)  # torch.float16

# 方法 2: .to()
tensor_int = tensor.to(torch.int32)
print(tensor_int.dtype)  # torch.int32

# 方法 3: 专用方法
tensor_long = tensor.long()  # torch.int64
tensor_double = tensor.double()  # torch.float64
tensor_half = tensor.half()  # torch.float16

# 查看所有转换方法
# .int(), .long(), .float(), .double(), .half(), .bool()
```

#### 5.3.4 精度权衡

```python
import torch
import time

# 准备数据
size = (1000, 1000)

# Float32
tensor_fp32 = torch.rand(size)
start = time.time()
result_fp32 = torch.matmul(tensor_fp32, tensor_fp32)
time_fp32 = time.time() - start

# Float16
tensor_fp16 = torch.rand(size, dtype=torch.float16)
start = time.time()
result_fp16 = torch.matmul(tensor_fp16, tensor_fp16)
time_fp16 = time.time() - start

print(f"Float32 时间: {time_fp32:.4f}秒")
print(f"Float16 时间: {time_fp16:.4f}秒")
print(f"加速比: {time_fp32/time_fp16:.2f}x")

# 精度损失
print(f"\nFloat32 结果范围: [{result_fp32.min():.6f}, {result_fp32.max():.6f}]")
print(f"Float16 结果范围: [{result_fp16.min():.6f}, {result_fp16.max():.6f}]")
```

**使用建议:**
- **训练:** `torch.float32` (默认,平衡性能和精度)
- **推理:** `torch.float16` 或 `torch.bfloat16` (更快)
- **混合精度训练:** 结合 FP16 和 FP32 (最佳实践)
- **整数:** 量化模型(减小模型大小)

#### 5.3.5 混合精度训练示例

```python
from torch.cuda.amp import autocast, GradScaler

# 创建模型、优化器
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

# 创建梯度缩放器
scaler = GradScaler()

for epoch in range(epochs):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        # 使用自动混合精度
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # 缩放损失并反向传播
        scaler.scale(loss).backward()

        # 更新参数
        scaler.step(optimizer)
        scaler.update()
```

### 5.4 张量属性查询

```python
# 创建示例张量
tensor = torch.rand(3, 4, 5)

# 重要属性
print(f"形状 (shape): {tensor.shape}")  # torch.Size([3, 4, 5])
print(f"大小 (size): {tensor.size()}")  # torch.Size([3, 4, 5])
print(f"维度 (ndim): {tensor.ndim}")  # 3
print(f"数据类型 (dtype): {tensor.dtype}")  # torch.float32
print(f"设备 (device): {tensor.device}")  # cpu 或 cuda:0
print(f"布局 (layout): {tensor.layout}")  # torch.strided
print(f"元素总数 (numel): {tensor.numel()}")  # 60
print(f"需要梯度 (requires_grad): {tensor.requires_grad}")  # False

# 内存占用
print(f"元素大小 (item size): {tensor.element_size()} 字节")  # 4
print(f"总内存 (memory): {tensor.numel() * tensor.element_size()} 字节")  # 240
```

---

# 第二部分：核心操作

## 6. 张量操作

### 6.1 获取张量信息 - "三个 W"

**记住这个口诀:**

```python
# What shape?   (什么形状)
# What datatype? (什么数据类型)
# Where stored?  (存储在哪里)
```

```python
# 创建张量
tensor = torch.rand(3, 4)

# 三个 W
print(f"What shape? {tensor.shape}")  # torch.Size([3, 4])
print(f"What datatype? {tensor.dtype}")  # torch.float32
print(f"Where stored? {tensor.device}")  # cpu
```

**为什么重要?**
- 90% 的PyTorch错误都与这三个属性有关!
- 形状不匹配 → RuntimeError
- 类型不匹配 → TypeError
- 设备不匹配 → RuntimeError

### 6.2 基本算术操作

#### 6.2.1 标量运算

```python
tensor = torch.tensor([10, 20, 30])

# 加法
print(tensor + 10)  # tensor([20, 30, 40])

# 减法
print(tensor - 10)  # tensor([0, 10, 20])

# 乘法
print(tensor * 10)  # tensor([100, 200, 300])

# 除法
print(tensor / 10)  # tensor([1., 2., 3.])

# 幂运算
print(tensor ** 2)  # tensor([100, 400, 900])

# 取模
print(tensor % 7)  # tensor([3, 6, 2])
```

#### 6.2.2 张量间运算

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 元素级运算
print(a + b)  # tensor([5, 7, 9])
print(a - b)  # tensor([-3, -3, -3])
print(a * b)  # tensor([4, 10, 18])
print(a / b)  # tensor([0.2500, 0.4000, 0.5000])

# 比较运算
print(a > b)  # tensor([False, False, False])
print(a == b)  # tensor([False, False, False])
```

#### 6.2.3 PyTorch 内置函数

```python
tensor = torch.tensor([10, 20, 30])

# 使用 PyTorch 函数 (推荐)
torch.add(tensor, 10)  # 等同于 tensor + 10
torch.sub(tensor, 10)  # 等同于 tensor - 10
torch.mul(tensor, 10)  # 等同于 tensor * 10
torch.div(tensor, 10)  # 等同于 tensor / 10

# 就地操作 (inplace) - 以 _ 结尾
tensor.add_(10)  # 直接修改 tensor
print(tensor)  # tensor([20, 30, 40])

tensor.mul_(2)  # 直接修改
print(tensor)  # tensor([40, 60, 80])
```

**就地操作 vs 非就地操作:**

```python
x = torch.tensor([1, 2, 3])

# 非就地 (返回新张量)
y = x.add(10)
print(x)  # tensor([1, 2, 3])  # 原张量不变
print(y)  # tensor([11, 12, 13])

# 就地 (修改原张量)
x.add_(10)
print(x)  # tensor([11, 12, 13])  # 原张量被修改
```

**注意:** 就地操作可以节省内存,但可能导致意外的副作用!

#### 6.2.4 高级数学函数

```python
import torch
import math

# 创建张量
x = torch.tensor([0.0, math.pi/4, math.pi/2, math.pi])

# 三角函数
print(torch.sin(x))
print(torch.cos(x))
print(torch.tan(x))

# 指数和对数
y = torch.tensor([1.0, 2.0, 3.0])
print(torch.exp(y))  # e^y
print(torch.log(y))  # ln(y)
print(torch.log10(y))  # log10(y)

# 开方
print(torch.sqrt(y))
print(torch.pow(y, 2))  # y^2

# 取整
z = torch.tensor([1.3, 2.7, -1.5])
print(torch.round(z))  # tensor([1., 3., -2.])
print(torch.floor(z))  # tensor([1., 2., -2.])
print(torch.ceil(z))  # tensor([2., 3., -1.])

# 裁剪
print(torch.clamp(z, min=-1, max=2))  # tensor([1.3, 2.0, -1.0])
```

### 6.3 张量与张量运算的形状规则

```python
# 规则: 形状必须兼容

# ✓ 相同形状
a = torch.rand(3, 4)
b = torch.rand(3, 4)
c = a + b  # 可以

# ✓ 广播兼容
a = torch.rand(3, 4)
b = torch.rand(1, 4)  # 会广播
c = a + b  # 可以

# ✗ 不兼容
a = torch.rand(3, 4)
b = torch.rand(3, 5)
# c = a + b  # 错误! 形状不兼容
```

**我们将在后面详细讨论广播机制!**

---

## 7. 矩阵乘法

### 7.1 为什么矩阵乘法如此重要?

**神经网络 = 矩阵乘法的堆叠**

![矩阵乘法就是你所需要的一切](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00_matrix_multiplication_is_all_you_need.jpeg)

```python
# 简化的神经网络层
class LinearLayer:
    def __init__(self, in_features, out_features):
        # 权重矩阵
        self.W = torch.randn(in_features, out_features)
        self.b = torch.randn(out_features)

    def forward(self, x):
        # 核心操作: 矩阵乘法 + 偏置
        return torch.matmul(x, self.W) + self.b

# 使用
layer = LinearLayer(10, 5)
x = torch.randn(1, 10)  # 1个样本, 10个特征
output = layer.forward(x)  # 1个样本, 5个输出
print(output.shape)  # torch.Size([1, 5])
```

### 7.2 矩阵乘法规则

#### 规则 1: 内部维度必须匹配

```python
# (m, n) @ (n, p) = (m, p)
#      ↑    ↑
#      必须相同

# ✓ 可以
A = torch.rand(3, 2)
B = torch.rand(2, 4)
C = torch.matmul(A, B)  # (3, 2) @ (2, 4) = (3, 4)

# ✗ 不可以
A = torch.rand(3, 2)
B = torch.rand(3, 4)
# C = torch.matmul(A, B)  # 错误! 2 != 3
```

#### 规则 2: 结果形状是外部维度

```python
# (m, n) @ (n, p) = (m, p)
#  ↑         ↑      ↑    ↑
#  外        外      结果形状

A = torch.rand(5, 3)
B = torch.rand(3, 7)
C = torch.matmul(A, B)
print(C.shape)  # torch.Size([5, 7])
```

### 7.3 元素乘法 vs 矩阵乘法

```python
tensor = torch.tensor([1, 2, 3])

# 元素乘法 (element-wise multiplication)
element_wise = tensor * tensor
print(element_wise)  # tensor([1, 4, 9])

# 矩阵乘法 (点积)
matrix_mul = torch.matmul(tensor, tensor)
print(matrix_mul)  # tensor(14)  # 1*1 + 2*2 + 3*3 = 14

# 也可以用 @ 运算符
matrix_mul2 = tensor @ tensor
print(matrix_mul2)  # tensor(14)
```

| 操作 | 符号 | 函数 | 输入形状 | 输出形状 | 计算 |
|------|------|------|---------|---------|------|
| **元素乘法** | `*` | `torch.mul()` | `(n,) * (n,)` | `(n,)` | `[a*b for a,b in zip(...)]` |
| **点积** | `@` | `torch.dot()` | `(n,) @ (n,)` | `()` | `sum(a*b for a,b in zip(...))` |
| **矩阵乘法** | `@` | `torch.matmul()` | `(m,n) @ (n,p)` | `(m,p)` | 矩阵乘法规则 |

### 7.4 矩阵乘法的多种方法

```python
A = torch.rand(3, 2)
B = torch.rand(2, 4)

# 方法 1: torch.matmul()
result1 = torch.matmul(A, B)

# 方法 2: @ 运算符 (推荐)
result2 = A @ B

# 方法 3: torch.mm() (仅2D矩阵)
result3 = torch.mm(A, B)

# 方法 4: .matmul() 方法
result4 = A.matmul(B)

# 所有结果相同
assert torch.all(result1 == result2)
assert torch.all(result1 == result3)
assert torch.all(result1 == result4)
```

**推荐使用 `@` 运算符,最简洁!**

### 7.5 转置解决形状不匹配

```python
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])  # 形状: (3, 2)

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])  # 形状: (3, 2)

# 直接相乘会报错
# result = tensor_A @ tensor_B  # RuntimeError!
# 因为 (3, 2) @ (3, 2) 内部维度不匹配

# 解决方案: 转置第二个张量
print(tensor_B.T)  # 形状: (2, 3)
# tensor([[ 7,  8,  9],
#         [10, 11, 12]])

# 现在可以相乘
result = tensor_A @ tensor_B.T  # (3, 2) @ (2, 3) = (3, 3)
print(result.shape)  # torch.Size([3, 3])
print(result)
# tensor([[ 27,  30,  33],
#         [ 61,  68,  75],
#         [ 95, 106, 117]])
```

**转置方法:**

```python
# 方法 1: .T 属性 (推荐,最简洁)
transpose1 = tensor_A.T

# 方法 2: torch.transpose(input, dim0, dim1)
transpose2 = torch.transpose(tensor_A, 0, 1)

# 方法 3: .transpose(dim0, dim1) 方法
transpose3 = tensor_A.transpose(0, 1)

# 方法 4: .permute() - 更灵活
transpose4 = tensor_A.permute(1, 0)

# 验证
assert torch.all(transpose1 == transpose2)
assert torch.all(transpose1 == transpose3)
assert torch.all(transpose1 == transpose4)
```

![矩阵乘法可视化演示](https://github.com/mrdbourke/pytorch-deep-learning/raw/main/images/00-matrix-multiply-crop.gif)

### 7.6 批量矩阵乘法

```python
# 批量矩阵乘法
batch_A = torch.rand(32, 3, 2)  # 32个 (3x2) 矩阵
batch_B = torch.rand(32, 2, 4)  # 32个 (2x4) 矩阵

# torch.matmul 自动处理批量
batch_C = torch.matmul(batch_A, batch_B)
print(batch_C.shape)  # torch.Size([32, 3, 4])

# 等价于手动循环 (但慢得多)
batch_C_manual = []
for i in range(32):
    C_i = batch_A[i] @ batch_B[i]
    batch_C_manual.append(C_i)
batch_C_manual = torch.stack(batch_C_manual)

# 验证
assert torch.allclose(batch_C, batch_C_manual)
```

### 7.7 神经网络中的线性层

```python
# torch.nn.Linear 实现: y = x·A^T + b

import torch.nn as nn

# 创建线性层
linear = nn.Linear(in_features=2, out_features=6)

# 查看权重
print(f"权重形状: {linear.weight.shape}")  # torch.Size([6, 2])
print(f"偏置形状: {linear.bias.shape}")  # torch.Size([6])

# 输入
x = torch.tensor([[1., 2.]])  # (1, 2)

# 前向传播
output = linear(x)  # (1, 6)
print(output.shape)

# 等价的手动计算
manual_output = x @ linear.weight.T + linear.bias
print(manual_output.shape)  # torch.Size([1, 6])

# 验证
assert torch.allclose(output, manual_output)
```

**公式:**
```
y = x · W^T + b

其中:
- x: 输入 (batch_size, in_features)
- W: 权重 (out_features, in_features)
- b: 偏置 (out_features,)
- y: 输出 (batch_size, out_features)
```

### 7.8 矩阵乘法性能对比

```python
import time

# 准备数据
size = 1000
A = torch.rand(size, size)
B = torch.rand(size, size)

# CPU 矩阵乘法
start = time.time()
C_cpu = A @ B
cpu_time = time.time() - start

# GPU 矩阵乘法 (如果可用)
if torch.cuda.is_available():
    A_gpu = A.cuda()
    B_gpu = B.cuda()

    # 预热
    _ = A_gpu @ B_gpu
    torch.cuda.synchronize()

    # 计时
    start = time.time()
    C_gpu = A_gpu @ B_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start

    print(f"CPU 时间: {cpu_time:.4f}秒")
    print(f"GPU 时间: {gpu_time:.4f}秒")
    print(f"加速比: {cpu_time/gpu_time:.2f}x")
else:
    print(f"CPU 时间: {cpu_time:.4f}秒")
    print("GPU 不可用")
```

---

由于内容非常多,这只是第一部分。文档将继续包括:

## 即将完成的章节:
- 8. 张量聚合操作
- 9. 张量重塑与变换
- 10. 索引操作
- 11. 广播机制详解 ⭐ (新增)
- 12. 自动微分 Autograd ⭐ (新增)
- 13-20. 其他章节...

由于篇幅限制,我现在先保存这部分,然后继续创建完整文档。

---

**文档持续创建中...**

此文档是增强版的第一部分,包含了:
- ✅ 更详细的概念解释
- ✅ 更多代码示例
- ✅ 可视化图表引用
- ✅ 性能对比
- ✅ 最佳实践建议

后续部分将包含广播、Autograd、GPU优化等高级主题!
