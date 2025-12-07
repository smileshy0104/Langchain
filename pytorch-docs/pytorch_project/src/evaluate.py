"""
评估模块

职责：
- 在测试集上评估模型
- 计算各种评估指标
"""
import torch
import numpy as np


# ==================== 模型评估函数 ====================
def evaluate_model(model, dataloader, device):
    """
    评估模型
    
    在测试集上运行模型，计算回归评估指标
    
    参数:
        model: 要评估的模型
        dataloader: 测试数据加载器
        device: 计算设备
    
    返回:
        dict: 包含各种评估指标的字典
            - MAE: 平均绝对误差
            - MSE: 均方误差
            - RMSE: 均方根误差
    """
    # 设置为评估模式
    model.eval()

    # 存储所有批次的预测和目标值
    all_preds = []
    all_targets = []

    # 禁用梯度计算
    with torch.inference_mode():
        for X, y in dataloader:
            # 只需要将输入移动到设备
            X = X.to(device)
            y_pred = model(X)

            # 将预测结果移回 CPU 并存储
            # .cpu(): 将张量从 GPU 移动到 CPU
            # 这是必要的，因为最终需要在 CPU 上合并所有结果
            all_preds.append(y_pred.cpu())
            all_targets.append(y)  # y 本来就在 CPU 上

    # ========== 合并所有批次 ==========
    # torch.cat: 沿指定维度拼接张量列表
    # 将多个 (batch_size, 1) 的张量合并为 (total_samples, 1)
    predictions = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # ========== 计算评估指标 ==========
    
    # MAE (Mean Absolute Error) - 平均绝对误差
    # 公式: MAE = (1/n) * Σ|y_pred - y_true|
    # 特点: 对异常值不敏感，直观易理解
    mae = torch.mean(torch.abs(predictions - targets))
    
    # MSE (Mean Squared Error) - 均方误差
    # 公式: MSE = (1/n) * Σ(y_pred - y_true)²
    # 特点: 对大误差惩罚更重，常用于优化目标
    mse = torch.mean((predictions - targets) ** 2)
    
    # RMSE (Root Mean Squared Error) - 均方根误差
    # 公式: RMSE = √MSE
    # 特点: 与原始数据单位相同，更易解释
    rmse = torch.sqrt(mse)

    # .item(): 将单元素张量转换为 Python 数值
    return {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'RMSE': rmse.item()
    }
