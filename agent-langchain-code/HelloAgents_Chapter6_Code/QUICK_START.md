# 快速开始指南

本指南帮助您快速上手 LangChain v1.0 多智能体系统。

## 🚀 推荐学习路径

### 步骤 1: 环境验证（1分钟）

```bash
# 运行快速测试
python quick_test.py
```

确保看到：`🎉 所有测试通过！`

### 步骤 2: 运行简化演示（推荐新手）

我们为每个示例都提供了简化版本（`simple_demo.py`），使用简短任务避免长时间运行和超时问题。

#### 2.1 智能搜索助手（最简单，推荐优先）

```bash
python SearchAssistant/simple_demo.py
```

**预计时间**: 30-60秒
**特点**:
- 使用模拟搜索，不需要额外API
- 展示 LangGraph 状态图工作流
- 三步推理：理解 → 搜索 → 生成答案

#### 2.2 软件开发团队

```bash
python SoftwareTeam/simple_demo.py
```

**预计时间**: 1-2分钟
**特点**:
- 产品经理 → 工程师 → 代码审查员协作
- 开发一个简单的 Python 函数
- 展示多角色顺序协作

#### 2.3 角色扮演创作

```bash
python BookWriting/simple_demo.py
```

**预计时间**: 1-2分钟
**特点**:
- 专家 ↔ 执行者双角色对话
- 创建一个教程大纲
- 展示迭代式协作

### 步骤 3: 运行完整示例（可选）

完整示例使用更复杂的任务，运行时间较长（5-15分钟）：

```bash
# 软件团队完整示例
python SoftwareTeam/software_team_langchain.py

# 角色扮演完整示例
python BookWriting/role_playing_langchain.py

# 搜索助手完整示例
python SearchAssistant/search_assistant_langgraph.py
```

## ⚠️ 常见问题

### 1. 超时错误 (Timeout Error)

**现象**: `The read operation timed out`

**原因**:
- 任务太复杂，需要很多轮对话
- API 响应慢
- 网络问题

**解决方案**:
```bash
# 方案 1: 使用简化演示
python SearchAssistant/simple_demo.py  # 推荐

# 方案 2: 修改代码减少对话轮次
# 在示例中找到 max_turns 参数，改为更小的值
# 例如：max_turns=3
```

### 2. API 配额不足

**现象**: API 返回配额错误

**解决方案**:
- 检查智谱AI账号配额
- 使用简化演示减少 API 调用
- 增加 API 调用间隔（修改代码添加 time.sleep()）

### 3. 导入错误

**现象**: `ModuleNotFoundError: No module named 'utils'`

**解决方案**:
```bash
# 确保从正确的目录运行
cd agent-langchain-code/HelloAgents_Chapter6_Code

# 运行测试验证
python quick_test.py
```

## 🎯 选择合适的示例

| 场景 | 推荐示例 | 难度 | 时间 |
|------|---------|------|------|
| 首次测试 | `SearchAssistant/simple_demo.py` | ⭐ | 30秒 |
| 理解团队协作 | `SoftwareTeam/simple_demo.py` | ⭐⭐ | 1分钟 |
| 理解角色扮演 | `BookWriting/simple_demo.py` | ⭐⭐ | 1分钟 |
| 开发实际应用 | 完整示例 + 修改任务 | ⭐⭐⭐ | 5-15分钟 |

## 💡 性能优化建议

### 减少 API 调用次数

1. **减少对话轮次**: 设置较小的 `max_turns`
2. **简化任务**: 使用简短明确的任务描述
3. **降低温度**: 使用较低的 `temperature` 参数（0.1-0.3）获得更确定的输出

### 避免超时

1. **使用简化演示**: 专为快速测试设计
2. **添加超时处理**: 修改代码添加 timeout 参数
3. **分步执行**: 将复杂任务拆分为多个简单任务

## 📚 下一步

### 学习资源

- 📖 阅读 [README.md](README.md) 了解完整功能
- 🔧 查看各个 `.py` 文件的源码和注释
- 🌐 访问 [LangChain 文档](https://python.langchain.com/)

### 实战练习

1. **修改任务**: 在 `simple_demo.py` 中修改任务描述
2. **调整参数**: 尝试不同的 `temperature` 和 `max_turns`
3. **添加工具**: 在软件团队示例中添加新工具
4. **集成真实API**: 在搜索助手中集成真实搜索API

## 🆘 获取帮助

- 查看 README.md 的"常见问题"部分
- 检查代码中的注释和文档字符串
- 在 GitHub 提交 Issue

---

**提示**: 推荐从 `SearchAssistant/simple_demo.py` 开始，它最简单且运行最快！
