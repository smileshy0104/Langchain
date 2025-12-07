# PyTorch 线性回归完整项目

这是一个完整的 PyTorch 项目示例，展示了从数据准备到模型训练、评估和部署的完整流程。

## 项目结构

```
pytorch_project/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后的数据
├── models/                  # 模型保存目录
│   └── checkpoints/        # 训练检查点
├── notebooks/               # Jupyter notebooks
│   └── exploration.ipynb   # 数据探索
├── src/                     # 源代码
│   ├── __init__.py         # 包初始化
│   ├── data.py             # 数据处理模块
│   ├── model.py            # 模型定义模块
│   ├── train.py            # 训练逻辑模块
│   ├── evaluate.py         # 评估逻辑模块
│   └── utils.py            # 工具函数模块
├── config.yaml              # 配置文件
├── requirements.txt         # 项目依赖
├── main.py                  # 主程序入口
└── README.md                # 项目说明
```

## 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 运行完整训练流程

```bash
python main.py
```

这将执行以下步骤：
1. 加载配置
2. 生成线性回归数据
3. 创建数据加载器
4. 构建模型
5. 训练模型
6. 评估模型
7. 保存模型和可视化结果

### 2. 修改配置

编辑 `config.yaml` 文件来调整超参数：

```yaml
# 训练配置
training:
  epochs: 100           # 训练轮数
  learning_rate: 0.01   # 学习率
  optimizer: "Adam"     # 优化器

# 数据配置
data:
  batch_size: 32        # 批次大小
  train_ratio: 0.7      # 训练集比例
```

### 3. 使用自己的数据

修改 `src/data.py` 中的数据加载函数：

```python
def load_custom_data(data_path):
    """加载自定义数据"""
    # 实现你的数据加载逻辑
    pass
```

## 模块说明

### data.py - 数据处理
- `create_linear_data()`: 生成线性回归数据
- `prepare_dataloaders()`: 创建训练/验证/测试数据加载器

### model.py - 模型定义
- `LinearRegressionModel`: 线性回归模型类
- `create_model()`: 模型工厂函数

### train.py - 训练逻辑
- `train_epoch()`: 单个 epoch 的训练
- `validate()`: 验证函数
- `train()`: 完整训练流程

### evaluate.py - 评估逻辑
- `evaluate_model()`: 计算评估指标（MAE, MSE, RMSE）

### utils.py - 工具函数
- `set_seed()`: 设置随机种子
- `plot_loss_curves()`: 绘制损失曲线
- `plot_predictions()`: 绘制预测结果
- `save_checkpoint()`: 保存训练检查点
- `load_checkpoint()`: 加载训练检查点
- `count_parameters()`: 统计模型参数

## 输出文件

运行 `main.py` 后，会在 `models/` 目录下生成：

- `final_model.pth`: 训练好的模型参数
- `loss_curves.png`: 训练和验证损失曲线
- `predictions.png`: 模型预测结果可视化

## 自定义扩展

### 添加新的模型

在 `src/model.py` 中定义新模型：

```python
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义你的模型结构
        
    def forward(self, x):
        # 定义前向传播
        return x
```

### 添加新的评估指标

在 `src/evaluate.py` 中添加：

```python
def calculate_r2_score(predictions, targets):
    """计算 R² 分数"""
    # 实现你的指标计算
    pass
```

### 使用学习率调度器

在 `src/train.py` 的 `train()` 函数中添加：

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(epochs):
    # 训练代码...
    scheduler.step()  # 更新学习率
```

## 常见问题

### Q: 如何使用 GPU 训练？

A: 在 `config.yaml` 中设置 `device: "cuda"`，程序会自动检测 GPU 可用性。

### Q: 如何保存训练检查点？

A: 使用 `utils.py` 中的 `save_checkpoint()` 函数：

```python
from src.utils import save_checkpoint

save_checkpoint(model, optimizer, epoch, loss, 'models/checkpoints/checkpoint.pth')
```

### Q: 如何恢复训练？

A: 使用 `utils.py` 中的 `load_checkpoint()` 函数：

```python
from src.utils import load_checkpoint

epoch, loss = load_checkpoint(model, optimizer, 'models/checkpoints/checkpoint.pth', device)
```

## 许可证

MIT License

## 作者

PyTorch 学习项目

## 参考资料

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [Learn PyTorch](https://www.learnpytorch.io/)
