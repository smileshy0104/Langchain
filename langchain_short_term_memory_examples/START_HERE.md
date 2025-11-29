# 🚀 从这里开始！

欢迎来到 LangChain 短期记忆示例项目！

## ⚡ 5分钟快速开始

### 步骤1：设置环境变量

```bash
export ZHIPUAI_API_KEY="your-api-key-here"
```

### 步骤2：运行第一个示例

```bash
python 01_basic_memory.py
```

### 步骤3：查看效果

你应该会看到类似这样的输出：

```
============================================================
示例1：基础短期记忆
============================================================

【第一轮对话】
👤 用户: 你好！我叫张三，我喜欢编程。
🤖 助手: 你好，张三！很高兴认识你...

【第二轮对话】
👤 用户: 我叫什么名字？
🤖 助手: 你叫张三。

【第三轮对话】
👤 用户: 我喜欢什么？
🤖 助手: 你喜欢编程。
```

恭喜！你已经成功运行了第一个短期记忆示例！🎉

---

## 📚 接下来做什么？

### 新手路径（推荐）

1. **阅读文档**（30分钟）
   - 打开 [README.md](README.md)
   - 理解核心概念
   - 查看示例说明

2. **运行示例**（2小时）
   ```bash
   python 01_basic_memory.py       # 基础记忆
   python 02_multi_thread.py       # 多用户
   python 03_trim_messages.py      # 消息修剪
   ```

3. **深入学习**（2天）
   - 打开 [LEARNING_GUIDE.md](LEARNING_GUIDE.md)
   - 按照4天学习计划
   - 完成练习题

### 进阶路径

1. **研究高级示例**
   ```bash
   python 04_summarization.py      # 消息摘要
   python 05_custom_state.py       # 自定义状态
   python 06_tool_state_access.py  # 工具集成
   ```

2. **查看快速参考**
   - 打开 [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
   - 查找常用代码片段
   - 学习最佳实践

3. **构建实战项目**
   - 参考 [LEARNING_GUIDE.md](LEARNING_GUIDE.md) 中的项目建议
   - 应用到自己的业务场景

---

## 🎯 学习目标

完成本项目后，你将掌握：

✅ 短期记忆的基本概念  
✅ 多用户会话管理  
✅ 消息管理策略（修剪/摘要）  
✅ 自定义状态设计  
✅ 工具与状态集成  
✅ 生产环境部署

---

## 📁 文件导航

| 文件 | 用途 | 适合 |
|------|------|------|
| [README.md](README.md) | 项目总览 | 所有人 |
| [LEARNING_GUIDE.md](LEARNING_GUIDE.md) | 学习指南 | 新手 |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | 快速参考 | 进阶 |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | 项目总结 | 贡献者 |

---

## 🆘 遇到问题？

### 常见错误

**错误：ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**错误：API Key 错误**
```bash
export ZHIPUAI_API_KEY="your-api-key"
```

**错误：记忆不工作**
- 确保使用了相同的 `thread_id`
- 检查是否添加了 `checkpointer`

### 获取帮助

1. 查看 [README.md](README.md) 的"常见问题"
2. 搜索错误信息
3. 查阅官方文档

---

## 🎁 推荐学习顺序

```
第1步（必须）  → 运行 01_basic_memory.py
              → 理解 thread_id 和 checkpointer

第2步（推荐）  → 运行 02_multi_thread.py
              → 理解会话隔离

第3步（重要）  → 运行 03_trim_messages.py
              → 学习消息管理

第4步（进阶）  → 运行其余示例
              → 完成练习题

第5步（实战）  → 构建自己的项目
              → 应用到生产
```

---

## 💡 小贴士

- 📝 **边学边做**：每个示例都可以修改和实验
- 🔍 **仔细阅读**：代码注释包含重要信息
- 🧪 **多做实验**：尝试修改参数，观察变化
- 📚 **查阅文档**：遇到问题先看文档
- 💬 **记录笔记**：记下关键概念和心得

---

**准备好了吗？开始你的学习之旅吧！🚀**

```bash
# 运行第一个示例
python 01_basic_memory.py

# 或使用批量运行工具
python run_all_examples.py
```

**Happy Learning! 🎉**
