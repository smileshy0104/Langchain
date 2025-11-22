# 第四章：智能体经典范式构建

> Hello-Agents 第四章完整学习资源集合

## 📑 目录结构

```
Chapter4-Agent-Patterns/
│
├── README.md                                    # 本文档（总索引）
│
├── 📖 理论文档/
│   ├── HelloAgents_Chapter4_通俗总结.md          # 第四章通俗易懂总结
│   └── 原始代码到LangChain_v1.0_转换指南.md      # 代码转换详细指南
│
└── 💻 代码实现/
    └── code-langchain-v1/                       # LangChain v1.0 实现
        ├── README.md                            # 项目完整说明
        ├── QUICKSTART.md                        # 5分钟快速开始
        ├── PROJECT_STATUS.md                    # 项目状态报告
        ├── requirements.txt                     # Python 依赖
        ├── .env.example                        # 环境配置模板
        │
        ├── utils.py                            # LLM 初始化
        ├── tools.py                            # 自定义工具
        │
        ├── 01_react_agent.py                   # ReAct 范式
        ├── 02_plan_and_solve.py                # Plan-and-Solve 范式
        └── 03_reflection_agent.py              # Reflection 范式
```

---

## 🎯 快速导航

### 我想...

**📖 理解三种智能体范式**
→ [HelloAgents_Chapter4_通俗总结.md](HelloAgents_Chapter4_通俗总结.md)
- ReAct = 侦探破案（边想边做）
- Plan-and-Solve = 建筑师盖房（先谋后动）
- Reflection = 匠人打磨（自我优化）

**🚀 快速运行代码**
→ [code-langchain-v1/QUICKSTART.md](code-langchain-v1/QUICKSTART.md)
- 5分钟配置环境
- 立即运行示例
- 基于 GLM-4 模型

**🔄 学习如何转换代码**
→ [原始代码到LangChain_v1.0_转换指南.md](原始代码到LangChain_v1.0_转换指南.md)
- 从原始实现到 LangChain
- 详细代码对比
- 转换步骤清单

**💻 查看完整代码**
→ [code-langchain-v1/README.md](code-langchain-v1/README.md)
- 17 个可运行示例
- 完整文档和注释
- 生产就绪代码

---

## 📚 学习路线

### 🌱 新手路线（推荐，2-3小时）

```
第1步（30分钟）- 理解概念
└─ 阅读: HelloAgents_Chapter4_通俗总结.md（重点部分）
   └─ 理解三种范式的核心思想

第2步（30分钟）- 配置环境
└─ 跟随: code-langchain-v1/QUICKSTART.md
   └─ 配置 API 密钥和依赖

第3步（1-2小时）- 动手实践
├─ 运行: 01_react_agent.py（ReAct 示例）
├─ 运行: 02_plan_and_solve.py（Plan-and-Solve 示例）
└─ 运行: 03_reflection_agent.py（Reflection 示例）
```

### ⚡ 快速上手（急用，30分钟）

```
第1步（10分钟）- 快速浏览
└─ 浏览: HelloAgents_Chapter4_通俗总结.md § 三种范式对比

第2步（10分钟）- 配置运行
└─ 跟随: code-langchain-v1/QUICKSTART.md
   └─ 配置 ZHIPUAI_API_KEY

第3步（10分钟）- 选择并运行
└─ 根据需求选择一个范式运行
   ├─ ReAct: 需要调用工具
   ├─ Plan-and-Solve: 需要分步骤
   └─ Reflection: 需要高质量输出
```

### 🔬 深度学习（1-2天）

```
第1天 - 理论深化
├─ 精读: HelloAgents_Chapter4_通俗总结.md
├─ 精读: 原始代码到LangChain_v1.0_转换指南.md
└─ 对比: 原教程手工实现 vs LangChain 实现

第2天 - 实践扩展
├─ 修改提示词测试效果
├─ 添加自定义工具
├─ 解决实际问题
└─ 集成到项目中
```

---

## 🎓 三种范式详解

### 1. ReAct - 边想边做 🕵️

**核心思想**: Thought → Action → Observation → 循环

**适用场景**:
- ✅ 需要查询实时信息（天气、新闻、股票）
- ✅ 需要使用外部工具（搜索、计算器）
- ✅ 问题需要动态决策

**示例代码**:
```python
from langchain.agents import create_agent
from utils import get_llm
from tools import get_weather, calculator

llm = get_llm(provider="zhipuai", model="glm-4")
agent = create_agent(model=llm, tools=[get_weather, calculator])

result = agent.invoke({
    "messages": [{"role": "user", "content": "北京天气如何？"}]
})
```

**文档**: [通俗总结 § ReAct](HelloAgents_Chapter4_通俗总结.md#react-范式---侦探破案)
**代码**: [01_react_agent.py](code-langchain-v1/01_react_agent.py)

---

### 2. Plan-and-Solve - 先谋后动 🏗️

**核心思想**: Plan (规划) → Solve (执行)

**适用场景**:
- ✅ 复杂的多步骤问题
- ✅ 可以提前规划的任务
- ✅ 需要结构化解决方案

**示例代码**:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 规划链
plan_chain = plan_prompt | llm | JsonOutputParser()
plan = plan_chain.invoke({"question": "数学应用题"})

# 执行链
for step in plan['steps']:
    result = execute_chain.invoke({"current_step": step})
```

**文档**: [通俗总结 § Plan-and-Solve](HelloAgents_Chapter4_通俗总结.md#plan-and-solve-范式---建筑师盖房)
**代码**: [02_plan_and_solve.py](code-langchain-v1/02_plan_and_solve.py)

---

### 3. Reflection - 自我优化 🎨

**核心思想**: Generate → Reflect → Refine → 循环

**适用场景**:
- ✅ 需要高质量输出（代码、文章）
- ✅ 可以通过迭代改进的任务
- ✅ 有明确质量标准

**示例代码**:
```python
# 初始生成
code = initial_chain.invoke({"task": "编写素数函数"})

# 迭代优化
for i in range(max_iterations):
    feedback = reflect_chain.invoke({"code": code})
    if "无需改进" in feedback:
        break
    code = refine_chain.invoke({"code": code, "feedback": feedback})
```

**文档**: [通俗总结 § Reflection](HelloAgents_Chapter4_通俗总结.md#reflection-范式---匠人打磨)
**代码**: [03_reflection_agent.py](code-langchain-v1/03_reflection_agent.py)

---

## 📊 实现对比

### 原教程 vs LangChain v1.0

| 特性 | 原教程（手工） | LangChain v1.0 |
|------|--------------|---------------|
| **代码量** | ~570 行 | ~200 行 (-65%) |
| **实现方式** | 手动循环 + 正则 | `create_agent` API |
| **错误处理** | 手动 try-except | 自动内置 |
| **工具调用** | 字典查找 | 自动路由 |
| **学习价值** | ⭐⭐⭐⭐⭐ 理解原理 | ⭐⭐⭐⭐ 工程实践 |
| **生产就绪** | ❌ 需完善 | ✅ 可直接用 |

**建议**: 两种都学！先理解原理，再掌握工具。

详细对比: [原始代码到LangChain_v1.0_转换指南.md](原始代码到LangChain_v1.0_转换指南.md)

---

## 🛠️ 技术栈

### 核心技术

- **LangChain v1.0**: 最新的智能体框架
- **智谱AI GLM-4**: 中文优化的大语言模型
- **LCEL**: LangChain Expression Language（可组合链）
- **Pydantic**: 数据验证和解析

### 优势

| 技术 | 优势 |
|------|------|
| **GLM-4** | 中文优化，成本更低（-70%），国内访问快 |
| **LangChain v1.0** | 代码简洁，错误处理完善，生产就绪 |
| **LCEL** | 可组合，可追踪，易于调试 |
| **BaseTool** | 标准化工具接口，自动文档生成 |

---

## 🔧 快速开始

### 最小化示例（复制即用）

```python
#!/usr/bin/env python3
"""最简 ReAct Agent 示例"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# 1. 加载环境变量
load_dotenv()

# 2. 定义工具
@tool
def calculator(expression: str) -> str:
    """执行数学计算

    Args:
        expression: 数学表达式，如 '2 + 3 * 4'
    """
    return f"结果: {eval(expression)}"

# 3. 创建 LLM
llm = ChatZhipuAI(
    model="glm-4",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    temperature=0.3
)

# 4. 创建 Agent
agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt="你是智能助手，可以调用工具。"
)

# 5. 运行
result = agent.invoke({
    "messages": [HumanMessage(content="计算 15 * 23 + 7")]
})

print(result["messages"][-1].content)
```

保存为 `minimal_example.py`，运行:
```bash
python minimal_example.py
```

---

## 📖 文档说明

### 理论文档

#### HelloAgents_Chapter4_通俗总结.md

**内容**:
- ✅ 三种范式通俗易懂的解释
- ✅ 生活化类比（侦探、建筑师、匠人）
- ✅ 适用场景和选择指南
- ✅ 完整的 FAQ 问答

**适合**: 初学者快速理解概念

**阅读时间**: 60分钟

---

#### 原始代码到LangChain_v1.0_转换指南.md

**内容**:
- ✅ LangChain v1.0 重要变化说明
- ✅ 逐行代码对比分析
- ✅ 详细的转换步骤
- ✅ 常见问题解答（8个问题）

**适合**: 想深入理解实现差异的开发者

**阅读时间**: 90分钟

---

### 代码文档

#### code-langchain-v1/README.md

**内容**:
- ✅ 项目完整介绍
- ✅ 安装和配置指南
- ✅ API 使用说明
- ✅ 17 个示例说明
- ✅ 常见问题解答

**适合**: 所有用户

---

#### code-langchain-v1/QUICKSTART.md

**内容**:
- ✅ 3 步快速开始
- ✅ 最小化代码示例
- ✅ 常见问题速查

**适合**: 需要快速上手的用户

**阅读时间**: 5分钟

---

## 🎯 使用建议

### 按目标选择

#### 目标：快速了解概念

```
1. 阅读: HelloAgents_Chapter4_通俗总结.md（前半部分）
2. 运行: 任选一个示例体验
时间: 30分钟
```

#### 目标：实际项目应用

```
1. 阅读: code-langchain-v1/README.md
2. 配置: 按 QUICKSTART.md 操作
3. 运行: 所有示例，选择合适的范式
4. 修改: 适配到自己的场景
时间: 2-3小时
```

#### 目标：深入理解原理

```
1. 阅读: HelloAgents_Chapter4_通俗总结.md（完整）
2. 阅读: 原始代码到LangChain_v1.0_转换指南.md
3. 对比: 原教程手工实现
4. 实践: 运行和修改所有示例
时间: 1-2天
```

---

## 💡 实战技巧

### 1. 选择合适的范式

**决策树**:
```
你的任务特点是？
├─ 需要调用外部工具/API？ → ReAct
├─ 可以分解为明确步骤？ → Plan-and-Solve
└─ 需要高质量迭代优化？ → Reflection
```

### 2. 调试技巧

```python
# 启用 debug 模式查看执行过程
agent = create_agent(..., debug=True)

# 使用 LangSmith 追踪（可选）
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_key"
```

### 3. 优化提示词

```python
# 针对中文任务优化
system_prompt = """你是一个强大的中文AI助手。

重要原则:
1. 使用中文回答
2. 回答简洁准确
3. 主动使用工具查询信息
4. 步骤清晰，逻辑严密
"""
```

### 4. 自定义工具

```python
from langchain_core.tools import tool

@tool
def my_tool(input: str) -> str:
    """工具描述（LLM 会看到）

    Args:
        input: 参数描述
    """
    # 你的实现
    return result

tools = [my_tool]
```

---

## ❓ 常见问题

### Q1: 我应该先学原教程还是直接用 LangChain？

**A**: 建议两者都学:
1. **先读理论**: [HelloAgents_Chapter4_通俗总结.md](HelloAgents_Chapter4_通俗总结.md) 理解原理
2. **再用框架**: [code-langchain-v1/](code-langchain-v1/) 工程实践

这样既理解底层，又掌握工具。

---

### Q2: GLM-4 和 GPT-4 该怎么选？

**A**:

| 场景 | 推荐 | 原因 |
|------|------|------|
| **中文任务** | GLM-4 | 中文理解更好 |
| **低成本** | GLM-4 | 价格约为 GPT-4 的 1/3 |
| **国内使用** | GLM-4 | 访问速度更快 |
| **英文任务** | GPT-4 | 英文能力更强 |
| **复杂推理** | GPT-4 | 逻辑推理更强 |

代码同时支持两者，可以随时切换。

---

### Q3: 如何切换到 GPT-4？

**A**: 修改 `.env`:
```bash
# 使用 OpenAI
LLM_MODEL_ID=gpt-4
LLM_API_KEY=your_openai_key
LLM_BASE_URL=https://api.openai.com/v1
```

代码中:
```python
llm = get_llm(provider="openai", model="gpt-4")
```

---

### Q4: 遇到错误怎么办？

**A**:
1. **检查 API 密钥**: 确保 `.env` 配置正确
2. **查看错误信息**: 通常很明确
3. **启用 debug**: `debug=True` 查看执行过程
4. **参考文档**: README 中的常见问题部分

---

## 📞 获取帮助

- 📖 **文档**: 查看各个 README 和指南
- 💻 **代码**: 所有示例都有详细注释
- 🔍 **搜索**: 使用本文档的目录快速定位
- 📚 **原教程**: https://datawhalechina.github.io/hello-agents/

---

## 🎉 开始学习

### 推荐起点

**第一次接触智能体？**
→ 从 [HelloAgents_Chapter4_通俗总结.md](HelloAgents_Chapter4_通俗总结.md) 开始

**想快速运行代码？**
→ 直接看 [code-langchain-v1/QUICKSTART.md](code-langchain-v1/QUICKSTART.md)

**想深入学习转换？**
→ 阅读 [原始代码到LangChain_v1.0_转换指南.md](原始代码到LangChain_v1.0_转换指南.md)

**想看完整项目？**
→ 浏览 [code-langchain-v1/README.md](code-langchain-v1/README.md)

---

## 📊 资源统计

```
文档数量: 6 篇
代码文件: 8 个
示例数量: 17 个
工具数量: 5 个
总代码行数: ~1,300 行
学习时间: 2小时-2天（根据目标）
```

---

**祝学习顺利！** 🎉

如果觉得有帮助，欢迎分享给更多人！

---

**最后更新**: 2025-11-22
**版本**: v1.0
**维护**: 持续更新中
