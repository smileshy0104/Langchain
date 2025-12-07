#!/bin/bash

# PyTorch 项目运行脚本

echo "================================"
echo "PyTorch 线性回归项目"
echo "================================"

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3 未安装"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "检查并安装依赖..."
pip install -q -r requirements.txt

# 运行主程序
echo ""
echo "运行训练程序..."
python main.py

# 完成
echo ""
echo "================================"
echo "执行完成！"
echo "================================"
echo "查看结果:"
echo "  - 模型文件: models/final_model.pth"
echo "  - 损失曲线: models/loss_curves.png"
echo "  - 预测结果: models/predictions.png"
