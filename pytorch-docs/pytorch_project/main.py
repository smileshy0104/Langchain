"""
主程序入口

职责：
- 加载配置
- 编排整个训练流程
- 协调各模块之间的调用
"""
import torch
import yaml
from pathlib import Path

# 从各模块导入函数
from src.data import create_linear_data, prepare_dataloaders
from src.model import create_model
from src.train import train
from src.evaluate import evaluate_model
from src.utils import set_seed, plot_loss_curves, plot_predictions, count_parameters


def main():
    """
    主函数 - 完整的机器学习流程
    
    流程:
    1. 加载配置
    2. 设置随机种子
    3. 设置设备
    4. 准备数据
    5. 创建模型
    6. 训练模型
    7. 评估模型
    8. 保存模型
    9. 可视化结果
    """
    
    # ========== 步骤 1: 加载配置 ==========
    # 使用 YAML 配置文件管理超参数
    # 优点: 易于修改、版本控制、实验追踪
    print("=" * 50)
    print("PyTorch 线性回归项目")
    print("=" * 50)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n配置加载成功:")
    print(f"  - Epochs: {config['training']['epochs']}")
    print(f"  - Learning Rate: {config['training']['learning_rate']}")
    print(f"  - Batch Size: {config['data']['batch_size']}")

    # ========== 步骤 2: 设置随机种子 ==========
    set_seed(42)
    print("\n随机种子已设置: 42")

    # ========== 步骤 3: 设置设备 ==========
    # 优先使用 GPU（如果可用）
    # torch.cuda.is_available(): 检查 CUDA 是否可用
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    if device == 'cuda':
        print(f"  - GPU 名称: {torch.cuda.get_device_name(0)}")

    # ========== 步骤 4: 准备数据 ==========
    print("\n" + "=" * 50)
    print("数据准备")
    print("=" * 50)
    
    print("\n创建线性回归数据...")
    X, y = create_linear_data(num_samples=1000)
    print(f"  - 数据形状: X={X.shape}, y={y.shape}")

    print("\n准备数据加载器...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        X, y,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    print(f"  - 训练集批次数: {len(train_loader)}")
    print(f"  - 验证集批次数: {len(val_loader)}")
    print(f"  - 测试集批次数: {len(test_loader)}")

    # ========== 步骤 5: 创建模型 ==========
    print("\n" + "=" * 50)
    print("模型创建")
    print("=" * 50)
    
    print("\n创建模型...")
    model = create_model(config, device)
    print(model)
    
    print("\n模型参数统计:")
    count_parameters(model)

    # ========== 步骤 6: 训练模型 ==========
    print("\n" + "=" * 50)
    print("模型训练")
    print("=" * 50)
    
    print("\n开始训练...")
    history = train(model, train_loader, val_loader, config, device)
    
    print("\n训练完成!")
    print(f"  - 最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"  - 最终验证损失: {history['val_loss'][-1]:.4f}")

    # ========== 步骤 7: 评估模型 ==========
    print("\n" + "=" * 50)
    print("模型评估")
    print("=" * 50)
    
    print("\n在测试集上评估模型...")
    metrics = evaluate_model(model, test_loader, device)
    
    print("\n测试集指标:")
    for name, value in metrics.items():
        print(f"  - {name}: {value:.4f}")

    # ========== 步骤 8: 保存模型 ==========
    print("\n" + "=" * 50)
    print("模型保存")
    print("=" * 50)
    
    # 使用 pathlib 处理路径（跨平台兼容）
    model_dir = Path(config['save']['model_dir'])
    model_dir.mkdir(exist_ok=True)  # 创建目录（如果不存在）

    model_path = model_dir / 'final_model.pth'
    
    # torch.save(): 保存模型
    # model.state_dict(): 只保存模型参数（推荐方式）
    # 优点: 文件小、加载灵活、不依赖模型类定义的位置
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")

    # ========== 步骤 9: 可视化结果 ==========
    print("\n" + "=" * 50)
    print("结果可视化")
    print("=" * 50)
    
    # 绘制损失曲线
    print("\n绘制损失曲线...")
    plot_loss_curves(history, save_path=model_dir / 'loss_curves.png')
    
    # 绘制预测结果
    print("\n绘制预测结果...")
    plot_predictions(model, X, y, device, save_path=model_dir / 'predictions.png')
    
    print("\n" + "=" * 50)
    print("项目执行完成！")
    print("=" * 50)


# ==================== 程序入口 ====================
# 当直接运行此文件时执行 main()
# 当被其他文件 import 时不执行
if __name__ == "__main__":
    main()
