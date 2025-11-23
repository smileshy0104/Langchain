# LangChain v1.0 多智能体系统实现 - 第六章

本目录包含使用 **LangChain v1.0** 实现的多智能体协作系统示例，对应 Hello-Agents 教程第六章的框架实践内容。

## 📁 目录结构

```
agent-langchain-code/HelloAgents_Chapter6_Code/
├── README.md                                    # 本文档
├── quick_test.py                                # 快速测试脚本
├── SoftwareTeam/                                # 软件开发团队协作
│   └── software_team_langchain.py
├── BookWriting/                                 # 角色扮演内容创作
│   └── role_playing_langchain.py
└── SearchAssistant/                             # LangGraph 智能搜索
    └── search_assistant_langgraph.py
```

**依赖关系**: 本目录下的代码依赖 `HelloAgents_Chapter4_Code/utils.py` 和 `tools.py`

## 🎯 核心特性

### LangChain v1.0 技术栈

本章节示例全面使用 LangChain v1.0 的最新特性：

- ✅ **create_agent API**: 简化的智能体创建接口
- ✅ **LCEL (LangChain Expression Language)**: 声明式链组合
- ✅ **LangGraph**: 状态图和工作流编排
- ✅ **智谱AI GLM-4.6**: 中文优化的大语言模型
- ✅ **多智能体协作**: 多个智能体并行/顺序协作

### 三大核心场景

| 场景 | 文件 | 特点 | 适用场景 |
|------|------|------|----------|
| **软件开发团队** | `SoftwareTeam/software_team_langchain.py` | 多角色顺序协作 | 需要分工明确的复杂任务 |
| **角色扮演创作** | `BookWriting/role_playing_langchain.py` | 双角色迭代对话 | 内容创作、教学辅导 |
| **智能搜索助手** | `SearchAssistant/search_assistant_langgraph.py` | LangGraph 状态机 | 多步推理、信息检索 |

## 🚀 快速开始

### 1. 环境准备

安装依赖：

```bash
# 基础依赖
pip install langchain langchain-community langchain-core
pip install langgraph
pip install python-dotenv

# 智谱AI SDK
pip install zhipuai
```

### 2. 配置 API 密钥

创建 `.env` 文件：

```bash
# 智谱AI API Key（必需）
ZHIPUAI_API_KEY=your-api-key-here

# 可选：搜索API（用于真实网络搜索）
SERPAPI_API_KEY=your-serpapi-key
TAVILY_API_KEY=your-tavily-key
```

获取 API 密钥：
- 智谱AI: https://open.bigmodel.cn/
- SerpAPI: https://serpapi.com/
- Tavily: https://tavily.com/

### 3. 快速验证

首先运行快速测试脚本，确保所有依赖和配置正确：

```bash
cd agent-langchain-code/HelloAgents_Chapter6_Code
python quick_test.py
```

如果所有测试通过，将看到：

```
🎉 所有测试通过！您可以开始使用 Chapter 6 示例了。
```

### 4. 运行示例

```bash
# 软件开发团队协作
python SoftwareTeam/software_team_langchain.py

# 角色扮演内容创作
python BookWriting/role_playing_langchain.py

# LangGraph 智能搜索
python SearchAssistant/search_assistant_langgraph.py
```

## 📌 详细示例

### 示例1: 软件开发团队协作

**文件**: `SoftwareTeam/software_team_langchain.py`

**场景**: 多智能体软件开发流程

#### 核心架构

```
用户任务
   ↓
产品经理 (需求分析)
   ↓
软件工程师 (代码实现)
   ↓
代码审查员 (质量检查)
   ↓
最终交付
```

#### 团队成员

| 角色 | 职责 | 技能 |
|------|------|------|
| **ProductManager** | 需求分析、技术规划 | 需求理解、功能划分、验收标准定义 |
| **Engineer** | 代码实现 | Python、Web开发、API集成、错误处理 |
| **CodeReviewer** | 代码审查 | 代码质量、安全性、最佳实践检查 |

#### 使用示例

```python
from software_team_langchain import SoftwareTeamAgent

# 创建团队
team = SoftwareTeamAgent(
    model="glm-4.6",
    temperature=0.3,
    debug=True
)

# 开发任务
task = """开发一个简单的天气查询命令行应用。

需求：
1. 用户输入城市名称
2. 调用天气 API 获取天气信息
3. 显示温度、湿度、天气状况
4. 提供友好的错误处理

技术栈：Python + requests 库"""

# 执行协作
results = team.run(task)

# 查看结果
team.print_summary(results)
```

#### 多轮迭代开发

支持代码审查反馈后的迭代优化：

```python
from software_team_langchain import MultiRoundCollaboration

collab = MultiRoundCollaboration(
    max_iterations=2,  # 最多迭代2次
    debug=True
)

final_code = collab.run(task)
```

---

### 示例2: 角色扮演内容创作

**文件**: `BookWriting/role_playing_langchain.py`

**场景**: 双角色协作创作电子书、教程、商业计划书等

#### 核心机制

```
任务定义
   ↓
专家角色 (提供指导)  ←→  执行角色 (完成创作)
   ↓                      ↓
迭代对话直到任务完成
   ↓
导出对话历史
```

#### 角色配置示例

| 任务类型 | 专家角色 | 执行角色 | 温度 |
|---------|---------|---------|------|
| 电子书创作 | 心理学家 | 作家 | 0.7 |
| 技术教程 | Python讲师 | 技术作家 | 0.6 |
| 商业计划 | 投资顾问 | 创业者 | 0.5 |
| 学术论文 | 教授 | 研究生 | 0.4 |
| 故事创作 | 作家导师 | 新人作家 | 0.8 |

#### 使用示例

```python
from role_playing_langchain import RolePlayingSession

# 定义协作任务
task = """创作一本关于"拖延症心理学"的短篇电子书。

要求：
1. 内容科学严谨，基于实证研究
2. 语言通俗易懂
3. 包含实用的改善建议和案例分析
4. 篇幅控制在8000-10000字
5. 结构清晰"""

# 创建会话
session = RolePlayingSession(
    assistant_role="心理学家",      # 专家角色
    user_role="作家",                # 执行角色
    task=task,
    temperature=0.7,                 # 创作温度
    max_turns=30,                    # 最大对话轮次
    debug=True
)

# 运行协作
conversation = session.run()

# 导出对话历史
session.export_conversation("book_conversation.txt")
```

#### 协作流程

1. **初始化**: 执行角色发起任务，请求专家指导
2. **迭代对话**:
   - 专家提供建议和反馈
   - 执行者根据反馈进行创作
3. **任务完成**: 当达到满意结果时，双方确认 `TASK_DONE`
4. **导出结果**: 完整对话历史保存到文件

---

### 示例3: LangGraph 智能搜索助手

**文件**: `SearchAssistant/search_assistant_langgraph.py`

**场景**: 基于状态图的多步推理搜索系统

#### LangGraph 状态图

```
START
  ↓
[understand_query] - 理解用户意图，优化搜索关键词
  ↓
[search_information] - 执行搜索（真实或模拟）
  ↓
[generate_answer] - 基于搜索结果生成答案
  ↓
END
```

#### 状态定义

```python
class SearchState(TypedDict):
    messages: List[BaseMessage]     # 对话历史
    user_query: str                 # 用户原始查询
    search_query: str               # 优化后的搜索查询
    search_results: str             # 搜索结果
    final_answer: str               # 最终答案
    step: str                       # 当前步骤
```

#### 使用示例

**基础搜索**:

```python
from search_assistant_langgraph import SearchAssistant

# 创建搜索助手
assistant = SearchAssistant(
    model="glm-4.6",
    temperature=0.7,
    use_memory=True,  # 启用记忆功能
    debug=True
)

# 执行搜索
answer = assistant.search("什么是 LangChain？")
print(answer)
```

**多轮对话**（带记忆）:

```python
assistant = SearchAssistant(use_memory=True)

conversation = [
    "什么是智谱AI？",
    "它有哪些主要产品？",           # 理解上下文
    "这些产品可以应用在哪些场景？"
]

thread_id = "conversation_1"

for user_input in conversation:
    response = assistant.chat(user_input, thread_id=thread_id)
    print(f"用户: {user_input}")
    print(f"助手: {response}\n")
```

#### 集成真实搜索

当前使用模拟搜索，可以轻松替换为真实搜索API：

```python
# 在 search_information_node 中替换
from langchain_community.tools import TavilySearchResults

def search_information_node(state: SearchState) -> dict:
    search_query = state["search_query"]

    # 使用 Tavily 搜索
    search = TavilySearchResults()
    results = search.invoke(search_query)

    return {
        "search_results": results,
        "step": "searched"
    }
```

## 🔧 高级功能

### 1. 自定义智能体角色

所有示例都支持自定义角色提示词：

```python
# 软件团队 - 自定义工程师角色
team = SoftwareTeamAgent()
team.engineer_prompt = """你是一位前端专家...
专精：React、Vue、TypeScript
..."""

# 角色扮演 - 自定义专家角色
session = RolePlayingSession(
    assistant_role="资深投资人",
    user_role="创业者",
    task="撰写商业计划书"
)
```

### 2. 温度参数调优

不同任务类型建议的温度参数：

| 任务类型 | 推荐温度 | 说明 |
|---------|---------|------|
| 代码生成 | 0.1-0.3 | 需要确定性和准确性 |
| 技术文档 | 0.3-0.5 | 平衡严谨性和可读性 |
| 创意写作 | 0.7-0.9 | 需要创造性和多样性 |
| 逻辑推理 | 0.0-0.2 | 需要最大确定性 |

### 3. 调试模式

所有示例支持详细的调试输出：

```python
agent = SearchAssistant(debug=True)

# 输出示例：
# 🤔 理解查询: 用户想了解 LangChain 的基本概念
# 🔍 搜索关键词: LangChain 框架
# 📄 搜索结果: ...
# 💡 生成答案: ...
```

## 📚 与其他框架的对比

### LangChain vs AutoGen

| 特性 | LangChain v1.0 | AutoGen |
|------|---------------|---------|
| **核心范式** | 链式组合 + 状态图 | 对话驱动 |
| **多智能体** | 手动编排（灵活） | 自动轮转（简单） |
| **工具集成** | 丰富的工具生态 | 需要自己实现 |
| **学习曲线** | 中等 | 较陡 |
| **中文支持** | 完善（本示例使用GLM） | 依赖底层模型 |

### LangChain vs CAMEL

| 特性 | LangChain v1.0 | CAMEL |
|------|---------------|-------|
| **协作模式** | 灵活的工作流 | 角色扮演范式 |
| **实现复杂度** | 需要手动编排 | 内置协作机制 |
| **扩展性** | 极强（模块化） | 受限于框架设计 |
| **最佳场景** | 复杂工作流 | 双角色对话 |

### LangGraph vs AgentScope

| 特性 | LangGraph | AgentScope |
|------|-----------|------------|
| **架构** | 显式状态图 | 消息驱动 |
| **可视化** | 支持图可视化 | MsgHub 架构图 |
| **分布式** | 需要自己实现 | 内置分布式支持 |
| **调试** | Checkpointer 状态回溯 | Pydantic 结构化 |

## 🎓 学习路径

### 初学者路径

1. **入门**: 运行 `SearchAssistant` 示例，理解基础的状态图概念
2. **进阶**: 修改 `RolePlayingSession` 的角色和任务
3. **实战**: 使用 `SoftwareTeamAgent` 开发真实项目

### 进阶学习

1. **自定义节点**: 在 LangGraph 中添加新的处理节点
2. **集成工具**: 添加真实的搜索、数据库等工具
3. **优化提示词**: 针对特定领域优化智能体提示词
4. **部署生产**: 添加错误处理、日志、监控

## 🔗 相关资源

### 官方文档

- [LangChain v1.0 文档](https://python.langchain.com/)
- [LangGraph 指南](https://langchain-ai.github.io/langgraph/)
- [智谱AI GLM](https://open.bigmodel.cn/)

### Hello-Agents 教程

- [第四章: 智能体经典范式](../HelloAgents_Chapter4_Code/)
- [第六章: 框架开发实践](https://github.com/datawhalechina/hello-agents/tree/V1.0.0/code/chapter6)

### 示例代码参考

- AutoGen 示例: https://github.com/datawhalechina/hello-agents/tree/V1.0.0/code/chapter6/AutoGenDemo
- CAMEL 示例: https://github.com/datawhalechina/hello-agents/tree/V1.0.0/code/chapter6/CAMEL
- LangGraph 示例: https://github.com/datawhalechina/hello-agents/tree/V1.0.0/code/chapter6/Langgraph

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 改进方向

- [ ] 集成真实搜索API（Tavily、SerpAPI）
- [ ] 添加更多智能体角色模板
- [ ] 支持流式输出
- [ ] 添加 Streamlit Web 界面
- [ ] 性能优化和缓存机制
- [ ] 多语言支持

## 📝 许可证

本项目遵循 MIT 许可证。

## ⚠️ 注意事项

1. **API 费用**: 使用智谱AI API 会产生费用，请注意控制调用次数
2. **速率限制**: 注意 API 的调用速率限制
3. **数据隐私**: 不要在提示词中包含敏感信息
4. **模型限制**: GLM-4.6 有上下文长度限制（约128K tokens）

## 🆘 常见问题

### Q: 如何切换到其他模型（如GPT-4）？

A: 修改 `utils.py` 中的 `get_llm` 函数：

```python
llm = get_llm(provider="openai", model="gpt-4")
```

### Q: 如何增加最大对话轮次？

A: 在创建会话时设置 `max_turns` 参数：

```python
session = RolePlayingSession(max_turns=50)  # 增加到50轮
```

### Q: 如何保存智能体输出？

A: 使用内置的导出功能：

```python
# 角色扮演会话
session.export_conversation("output.txt")

# 软件团队结果
with open("output.txt", "w") as f:
    f.write(results["engineer_code"])
```

### Q: 如何调试工作流？

A: 启用 `debug=True` 并查看详细输出：

```python
agent = SearchAssistant(debug=True)
```

---

**版本**: v1.0.0
**更新日期**: 2025-11-23
**作者**: LangChain Multi-Agent Examples Contributors
