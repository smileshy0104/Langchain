"""
工具函数模块

职责：
- 提供通用的辅助函数
- 可视化函数
- 其他工具函数
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def set_seed(seed=42):
    """
    设置随机种子，确保结果可重复
    
    参数:
        seed (int): 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def plot_loss_curves(history, save_path=None):
    """
    绘制训练和验证损失曲线
    
    参数:
        history (dict): 包含 'train_loss' 和 'val_loss' 的字典
        save_path (str, optional): 保存图片的路径
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curves saved to {save_path}")
    
    plt.show()


def plot_predictions(model, X, y, device, save_path=None):
    """
    绘制模型预测结果与真实值的对比
    
    参数:
        model: 训练好的模型
        X (Tensor): 输入数据
        y (Tensor): 真实标签
        device (str): 计算设备
        save_path (str, optional): 保存图片的路径
    """
    model.eval()
    
    with torch.inference_mode():
        X_device = X.to(device)
        y_pred = model(X_device).cpu()
    
    plt.figure(figsize=(10, 6))
    
    # 绘制真实值
    plt.scatter(X.numpy(), y.numpy(), c='blue', label='True Values', alpha=0.6)
    
    # 绘制预测值
    plt.scatter(X.numpy(), y_pred.numpy(), c='red', label='Predictions', alpha=0.6)
    
    plt.title('Model Predictions vs True Values')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Predictions plot saved to {save_path}")
    
    plt.show()


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    保存训练检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        epoch (int): 当前 epoch
        loss (float): 当前损失
        path (str): 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """
    加载训练检查点
    
    参数:
        model: 模型
        optimizer: 优化器
        path (str): 检查点路径
        device (str): 计算设备
    
    返回:
        epoch (int): 保存时的 epoch
        loss (float): 保存时的损失
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {path}")
    print(f"Resuming from epoch {epoch}, loss {loss:.4f}")
    
    return epoch, loss


def count_parameters(model):
    """
    统计模型参数数量
    
    参数:
        model: PyTorch 模型
    
    返回:
        total_params (int): 总参数数量
        trainable_params (int): 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params
