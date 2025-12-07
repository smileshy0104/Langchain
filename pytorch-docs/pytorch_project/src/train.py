"""
训练模块

职责：
- 单个 epoch 的训练逻辑
- 验证逻辑
- 完整训练流程编排
"""
import torch
from torch import nn
from tqdm.auto import tqdm  # 进度条库，auto 版本自动适配环境（notebook/终端）


# ==================== 单 Epoch 训练函数 ====================
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个 epoch
    
    执行完整的训练循环：遍历所有批次，更新模型参数
    
    参数:
        model: 要训练的模型
        dataloader: 训练数据加载器
        loss_fn: 损失函数
        optimizer: 优化器
        device: 计算设备
    
    返回:
        float: 该 epoch 的平均损失
    """
    # 设置为训练模式
    # 启用 Dropout、BatchNorm 的训练行为
    model.train()
    total_loss = 0

    # 遍历所有批次
    for X, y in dataloader:
        # ========== 数据移动到设备 ==========
        # 确保数据和模型在同一设备上
        X, y = X.to(device), y.to(device)

        # ========== 前向传播 ==========
        y_pred = model(X)           # 模型预测
        loss = loss_fn(y_pred, y)   # 计算损失

        # ========== 反向传播 ==========
        optimizer.zero_grad()  # 清零梯度（PyTorch 默认累积梯度）
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数

        # 累积损失
        # .item(): 将单元素张量转换为 Python 数值
        # 这会将数据从 GPU 复制到 CPU，所以只在需要时使用
        total_loss += loss.item()

    # 返回平均损失
    return total_loss / len(dataloader)


# ==================== 验证函数 ====================
def validate(model, dataloader, loss_fn, device):
    """
    验证函数
    
    在验证集上评估模型性能，不更新参数
    
    参数:
        model: 要验证的模型
        dataloader: 验证数据加载器
        loss_fn: 损失函数
        device: 计算设备
    
    返回:
        float: 验证集的平均损失
    """
    # 设置为评估模式
    # 禁用 Dropout，BatchNorm 使用运行时统计量
    model.eval()
    total_loss = 0

    # torch.inference_mode(): 禁用梯度计算
    # 比 torch.no_grad() 更高效
    # 用于推理/评估，节省内存和计算
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# ==================== 完整训练流程 ====================
def train(model, train_loader, val_loader, config, device):
    """
    完整训练流程
    
    编排整个训练过程：设置优化器、执行训练循环、记录历史
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置字典
        device: 计算设备
    
    返回:
        dict: 训练历史，包含 'train_loss' 和 'val_loss' 列表
    """
    # ========== 设置损失函数和优化器 ==========
    # MSELoss: 均方误差，适用于回归问题
    # 公式: MSE = (1/n) * Σ(y_pred - y_true)²
    loss_fn = nn.MSELoss()
    
    # Adam 优化器: 自适应学习率，通常效果好
    # 从配置文件读取学习率，便于调参
    optimizer = torch.optim.Adam(
        model.parameters(),                      # 要优化的参数
        lr=config['training']['learning_rate']   # 学习率
    )

    # ========== 训练循环 ==========
    epochs = config['training']['epochs']
    
    # 记录训练历史，用于后续可视化和分析
    history = {'train_loss': [], 'val_loss': []}

    # tqdm: 显示进度条
    for epoch in tqdm(range(epochs), desc="Training"):
        # 训练一个 epoch
        train_loss = train_epoch(model, train_loader, loss_fn,
                                optimizer, device)
        # 验证
        val_loss = validate(model, val_loader, loss_fn, device)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # 每 10 个 epoch 打印一次进度
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}")

    return history
