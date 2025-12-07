# PyTorch 完整项目总览

## 项目简介

这是一个完整的 PyTorch 机器学习项目模板，实现了线性回归任务。项目采用模块化设计，代码结构清晰，易于扩展和维护。

## 核心特性

✅ **模块化设计** - 数据、模型、训练、评估分离  
✅ **配置驱动** - 使用 YAML 配置文件管理超参数  
✅ **详细注释** - 每个函数都有完整的中文注释  
✅ **完整流程** - 从数据准备到模型部署的完整流程  
✅ **可视化** - 自动生成损失曲线和预测结果图  
✅ **设备无关** - 自动检测并使用 GPU/CPU  
✅ **最佳实践** - 遵循 PyTorch 官方推荐的代码规范  

## 项目架构

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│                   (主程序入口)                           │
└────────────┬────────────────────────────────────────────┘
             │
             ├─── config.yaml (配置文件)
             │
             ├─── src/data.py (数据模块)
             │    ├── create_linear_data()
             │    └── prepare_dataloaders()
             │
             ├─── src/model.py (模型模块)
             │    ├── LinearRegressionModel
             │    └── create_model()
             │
             ├─── src/train.py (训练模块)
             │    ├── train_epoch()
             │    ├── validate()
             │    └── train()
             │
             ├─── src/evaluate.py (评估模块)
             │    └── evaluate_model()
             │
             └─── src/utils.py (工具模块)
                  ├── set_seed()
                  ├── plot_loss_curves()
                  ├── plot_predictions()
                  ├── save_checkpoint()
                  └── load_checkpoint()
```

## 数据流

```
原始数据
   ↓
create_linear_data() → 生成线性数据 (X, y)
   ↓
prepare_dataloaders() → 创建 DataLoader
   ↓
   ├── train_loader (训练集)
   ├── val_loader (验证集)
   └── test_loader (测试集)
```

## 训练流程

```
1. 加载配置 (config.yaml)
   ↓
2. 设置随机种子 (确保可重复性)
   ↓
3. 设置设备 (GPU/CPU)
   ↓
4. 准备数据
   ├── 生成数据
   └── 创建 DataLoader
   ↓
5. 创建模型
   ├── 定义模型架构
   └── 移动到设备
   ↓
6. 训练循环
   ├── 前向传播
   ├── 计算损失
   ├── 反向传播
   └── 更新参数
   ↓
7. 评估模型
   ├── MAE
   ├── MSE
   └── RMSE
   ↓
8. 保存结果
   ├── 模型参数
   ├── 损失曲线
   └── 预测结果
```

## 关键代码片段

### 1. 数据准备

```python
# 生成数据
X, y = create_linear_data(num_samples=1000)

# 创建数据加载器
train_loader, val_loader, test_loader = prepare_dataloaders(
    X, y,
    train_ratio=0.7,
    val_ratio=0.15,
    batch_size=32
)
```

### 2. 模型定义

```python
class LinearRegressionModel(nn.Module):
    def __init__(self, input_features=1, output_features=1):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features)
    
    def forward(self, x):
        return self.linear(x)
```

### 3. 训练循环

```python
for epoch in range(epochs):
    # 训练
    model.train()
    for X, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    with torch.inference_mode():
        val_loss = validate(model, val_loader, loss_fn, device)
```

### 4. 模型评估

```python
metrics = evaluate_model(model, test_loader, device)
# 返回: {'MAE': ..., 'MSE': ..., 'RMSE': ...}
```

## 配置说明

### config.yaml 结构

```yaml
data:           # 数据配置
  train_ratio: 0.7
  val_ratio: 0.15
  batch_size: 32

model:          # 模型配置
  input_features: 1
  output_features: 1

training:       # 训练配置
  epochs: 100
  learning_rate: 0.01
  optimizer: "Adam"

device: "cuda"  # 设备配置

save:           # 保存配置
  model_dir: "models"
```

## 使用场景

### 场景 1: 快速原型开发

```bash
# 1. 修改配置
vim config.yaml

# 2. 运行训练
python main.py

# 3. 查看结果
ls models/
```

### 场景 2: 实验对比

```bash
# 实验 1: 小学习率
# config.yaml: learning_rate: 0.001
python main.py

# 实验 2: 大学习率
# config.yaml: learning_rate: 0.1
python main.py
```

### 场景 3: 模型调试

```python
# 在 main.py 中添加调试代码
from src.utils import count_parameters

model = create_model(config, device)
count_parameters(model)  # 查看参数数量
```

## 扩展建议

### 1. 添加更多数据集

在 `src/data.py` 中添加：

```python
def load_csv_data(file_path):
    """从 CSV 加载数据"""
    import pandas as pd
    df = pd.read_csv(file_path)
    # 处理数据...
    return X, y
```

### 2. 实现更复杂的模型

在 `src/model.py` 中添加：

```python
class MLPModel(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
```

### 3. 添加早停机制

在 `src/train.py` 中添加：

```python
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

### 4. 添加 TensorBoard 支持

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.close()
```

## 性能优化建议

1. **数据加载优化**
   - 增加 `num_workers` 参数
   - 使用 `pin_memory=True`（GPU 训练时）

2. **训练优化**
   - 使用混合精度训练 (`torch.cuda.amp`)
   - 梯度累积（处理大批次）
   - 学习率调度器

3. **内存优化**
   - 使用 `torch.inference_mode()` 代替 `torch.no_grad()`
   - 及时释放不需要的张量
   - 使用梯度检查点（大模型）

## 常见错误及解决

### 错误 1: CUDA out of memory

**解决方案:**
```yaml
# 减小批次大小
data:
  batch_size: 16  # 从 32 改为 16
```

### 错误 2: 损失不下降

**解决方案:**
```yaml
# 调整学习率
training:
  learning_rate: 0.001  # 尝试更小的学习率
```

### 错误 3: 过拟合

**解决方案:**
```python
# 在模型中添加 Dropout
self.dropout = nn.Dropout(0.5)
```

## 项目检查清单

- [x] 配置文件完整
- [x] 所有模块有详细注释
- [x] 训练流程可运行
- [x] 评估指标正确
- [x] 结果可视化
- [x] README 文档完整
- [x] 依赖列表准确
- [x] .gitignore 配置

## 下一步

1. ✅ 运行 `python main.py` 测试项目
2. ✅ 查看生成的可视化结果
3. ✅ 尝试修改配置文件
4. ✅ 扩展到自己的数据集
5. ✅ 实现更复杂的模型

## 参考资源

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [Learn PyTorch](https://www.learnpytorch.io/)
- [PyTorch 最佳实践](https://pytorch.org/docs/stable/notes/best_practices.html)

---

**项目版本:** 1.0.0  
**最后更新:** 2024  
**维护者:** PyTorch 学习者
