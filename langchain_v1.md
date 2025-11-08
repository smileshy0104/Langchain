# LangChain v1.0 完整指南

> 面向生产环境的 AI Agent 构建框架 - 核心变化、优势与迁移指南

---

## 📋 目录

- [版本概况](#版本概况)
- [核心设计理念](#核心设计理念)  
- [架构层面的重大变化](#架构层面的重大变化)
- [功能层面的核心改进](#功能层面的核心改进)
- [生产就绪性提升](#生产就绪性提升)
- [主要破坏性变更](#主要破坏性变更)
- [核心优势总结](#核心优势总结)
- [设计哲学演变](#设计哲学演变)
- [迁移指南](#迁移指南)
- [最佳实践](#最佳实践)
- [参考资源](#参考资源)

---

## 版本概况

| 项目 | 内容 |
|------|------|
| **发布时间** | 2025年10月20日 |
| **版本号** | v1.0.0 |
| **定位** | 面向生产环境的 Agent 构建基础框架 |
| **核心理念** | 最简单的 LLM 应用起点，同时兼具灵活性和生产就绪能力 |
| **稳定性承诺** | 在 2.0 之前无破坏性变更 |
| **采用企业** | Uber、LinkedIn、Klarna 等 |

---

## 核心设计理念

LangChain v1.0 构建在以下核心信念之上：

### 1. 数据融合是关键
> LLM 与外部数据源结合时最具变革性

- 支持多种数据源集成
- 标准化的数据访问接口
- 高效的数据检索和处理

### 2. Agent 化是趋势
> 未来应用将越来越 Agent 化

- 自主决策能力
- 动态工具调用
- 上下文感知的行为

### 3. 生产可靠性挑战
> 尽管原型开发容易，但构建可靠的生产级 Agent 仍具挑战性

- 持久化执行机制
- 错误处理和恢复
- 状态管理和检查点

### 4. 编排优先于生成
> 模型应编排复杂流程，而非仅生成文本

- 工作流编排能力
- 多步骤任务执行
- 确定性与 Agent 化的结合

---

## 架构层面的重大变化

### 1. 统一的 Agent 抽象 🎯

#### 核心变化
- **旧架构**: 多种 chains 和 agents（ConversationalRetrievalChain、ReActAgent 等）
- **新架构**: 单一的高层抽象 - 基于 LangGraph 构建的统一 Agent

#### 新 API: `create_agent`

**替代关系**:
```python
# ❌ 旧方式 (已弃用)
from langgraph.prebuilt import create_react_agent

# ✅ 新方式 (v1.0 推荐)
from langchain.agents import create_agent
```

**核心特性**:
- 更清晰、更强大的 API 接口
- 10 行代码内构建基础 Agent
- 支持通过中间件（middleware）进行深度自定义
- 内置工具调用、状态管理、错误处理

**基础示例**:
```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

# 定义工具
def get_weather(location: str) -> str:
    """获取指定位置的天气信息"""
    return f"{location} 的天气是晴朗的"

# 创建 Agent（少于 10 行代码）
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个有帮助的 AI 助手"
)

# 执行
result = agent.invoke({"messages": [{"role": "user", "content": "北京天气如何？"}]})
```

### 2. 简化的包结构 📦

#### 命名空间精简

**核心原则**: `langchain` 包聚焦于 Agent 构建的核心组件

**包结构变化**:

| 功能模块 | v0.x 位置 | v1.0 位置 |
|---------|----------|----------|
| Agent 创建 | `langchain.agents` | `langchain.agents` ✅ |
| 模型集成 | `langchain.chat_models` | `langchain_*` (独立包) |
| Retrievers | `langchain.retrievers` | `langchain-classic` |
| Embeddings | `langchain.embeddings` | `langchain-classic` |
| Chains | `langchain.chains` | `langchain-classic` |
| Indexing | `langchain.indexes` | `langchain-classic` |
| Hub | `langchain.hub` | `langchain-classic` |

**优势**:
- ✅ 更容易发现核心功能
- ✅ 减少学习曲线
- ✅ 更清晰的依赖关系
- ✅ 更小的包体积

### 3. 深度集成 LangGraph 🔗

#### 双层架构

```
┌─────────────────────────────────────┐
│      LangChain v1.0 (高层抽象)       │
│   - create_agent                    │
│   - 简单易用（<10 行代码）            │
│   - 快速原型开发                     │
└───────────────┬─────────────────────┘
                │ 构建在
                ↓
┌─────────────────────────────────────┐
│      LangGraph v1.0 (底层运行时)     │
│   - 图原语（状态、节点、边）          │
│   - 持久化执行                       │
│   - 精细控制                        │
└─────────────────────────────────────┘
```

#### LangGraph v1.0 核心特性

| 特性 | 说明 | 应用场景 |
|-----|------|---------|
| **持久化执行** | 状态自动保存和恢复 | 长时间运行的任务 |
| **检查点** | 任意时刻保存/加载状态 | 错误恢复、调试 |
| **流式处理** | 实时输出中间结果 | 交互式应用 |
| **人机协作** | 暂停等待人工输入 | 审批流程、复杂决策 |
| **稳定 API** | 图原语保持不变 | 长期维护的项目 |

---

## 功能层面的核心改进

### 1. 标准化的内容块（Content Blocks） 📝

#### 背景问题

**模型 API 演变**:
- **早期**: 简单的文本字符串输出
- **现在**: 复杂的结构化输出（推理、引用、工具调用等）
- **问题**: 不同提供商格式不一致

#### v1.0 解决方案

**新属性**: `content_blocks`

**特性**:
- ✅ 统一访问现代 LLM 特性
- ✅ 响应不再是不透明字符串
- ✅ 结构化的内容分类
- ✅ 跨提供商标准化

**支持的内容类型**:
- 文本（text）
- 推理过程（reasoning）
- 引用（citation）
- 工具调用（tool_call）
- 多模态输入（文件、图像、视频等）

### 2. 增强的结构化输出 🎯

#### 改进方案

**旧方式**: 两步流程
1. Agent 生成响应
2. 额外 LLM 调用提取结构化数据

**新方式**: 集成到主循环
- 将结构化输出生成集成到主模型-工具循环
- 减少额外的 LLM 调用
- 降低延迟和成本

**优势**:
- ⚡ 更低延迟（减少 LLM 调用）
- 💰 更低成本（减少 token 使用）
- 🎯 更高准确性（模型直接生成结构化数据）

### 3. 中间件系统（Middleware） 🔧

#### 概述

**新功能**: 可在模型调用、工具调用周围组合中间件

**核心价值**:
- 横切关注点（日志、监控、错误处理）
- 动态行为注入
- 可复用的逻辑组件

#### 支持的钩子

```python
from langchain.agents import Middleware

class CustomMiddleware(Middleware):
    """自定义中间件示例"""

    def before_model(self, state, config):
        """模型调用前执行"""
        print(f"准备调用模型，当前状态: {state}")
        return state

    def after_model(self, state, response, config):
        """模型调用后执行"""
        print(f"模型调用完成")
        return response

    def wrap_tool_call(self, tool_call, state, config):
        """包装工具调用"""
        try:
            result = tool_call()
            return result
        except Exception as e:
            print(f"工具调用失败: {e}")
            return {"error": str(e)}
```

### 4. 标准化的模型接口 🔌

#### 核心问题

**提供商差异**:
- OpenAI: `chat.completions.create()`
- Anthropic: `messages.create()`
- Google: `generateContent()`
- 不同的参数名、响应格式

#### LangChain 统一方案

**统一 API**:
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# 所有模型使用相同接口
models = [
    ChatOpenAI(model="gpt-4"),
    ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    ChatGoogleGenerativeAI(model="gemini-pro")
]

for model in models:
    # 统一的调用方式
    response = model.invoke([
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"}
    ])
    print(response.content)  # 统一的响应格式
```

---

## 生产就绪性提升

### 1. 稳定性承诺 🛡️

#### 版本策略

**承诺**:
- ✅ 在 2.0 之前无破坏性变更
- ✅ 公共 API 保持稳定
- ✅ 弃用前有充分通知

**发布节奏**:
- 🗓️ Minor 版本: 每 2-3 个月
- 🐛 Patch 版本: 按需发布
- 📢 Major 版本: 提前至少 6 个月通知

### 2. 调试增强 🔍

#### LangSmith 深度集成

**核心功能**:

**1. 执行追踪**
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# 自动追踪所有执行
agent.invoke({"messages": [...]})

# 在 LangSmith UI 查看:
# - 每一步的输入输出
# - 工具调用详情
# - 模型推理时间
# - Token 使用统计
```

**2. 状态可视化**
- 图形化展示 Agent 执行流程
- 节点间的数据流向
- 条件分支的决策路径

### 3. 文档重构 📚

#### 新文档站点特性

**导航优化**:
- 🧭 直观的分类结构
- 🔍 强大的搜索功能
- 🏷️ 标签和过滤器

**内容改进**:
- 📖 深入的概念指南
- 💻 可运行的代码示例
- 🎓 端到端教程
- 🏗️ 常见架构模式

---

## 主要破坏性变更

### 1. 已弃用方法移除 ❌

#### 移除列表

| 移除项 | 弃用于 | 替代方案 |
|-------|-------|---------|
| `AgentExecutor` | v0.2 | `create_agent` |
| `LLMChain` | v0.1 | 直接使用模型 |
| `ConversationalRetrievalChain` | v0.2 | 自定义 Agent + 工具 |
| `load_tools()` | v0.1 | 显式导入工具 |

#### 迁移示例

**AgentExecutor → create_agent**:
```python
# ❌ 旧代码
from langchain.agents import AgentExecutor, create_react_agent

agent_executor = AgentExecutor(
    agent=create_react_agent(llm, tools, prompt),
    tools=tools,
    verbose=True
)

# ✅ 新代码
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=prompt
)
```

### 2. 消息对象变更 💬

**`.text()` 方法 → `.text` 属性**:
```python
from langchain_core.messages import AIMessage

message = AIMessage(content="你好")

# ❌ 旧方式（仍可用但有警告）
text = message.text()

# ✅ 新方式
text = message.text
```

### 3. AIMessage 变更 🤖

**`example` 参数被移除**:
```python
# ❌ 旧代码
message = AIMessage(
    content="你好",
    example=True  # ❌ 已移除
)

# ✅ 新代码
message = AIMessage(
    content="你好",
    additional_kwargs={"example": True}  # ✅ 使用 additional_kwargs
)
```

### 4. Agent 定义方式 🏗️

**规则**: 从 v1.0 起，只能在 LangChain 内定义 Agent

```python
# ❌ 不推荐：直接使用 LangGraph
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model, tools)

# ✅ 推荐：使用 LangChain
from langchain.agents import create_agent
agent = create_agent(model, tools)
```

### 5. 流式处理变更 🌊

**节点名称更改**:
```python
# ❌ v0.x
async for chunk in agent.astream(...):
    if chunk.get("node") == "agent":  # 旧名称
        print(chunk)

# ✅ v1.0
async for chunk in agent.astream(...):
    if chunk.get("node") == "model":  # 新名称
        print(chunk)
```

---

## 核心优势总结

### 1. 简化开发 ⚡

**量化指标**:
- 📉 代码量减少 60-80%
- ⏱️ 上手时间从天降至小时
- 🎯 API 数量减少 70%

**具体体现**:

**旧方式 (~50 行)**:
```python
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

memory = ConversationBufferMemory(return_messages=True)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    agent_kwargs={
        "system_message": "...",
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
)
```

**新方式 (~10 行)**:
```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是有帮助的助手"
)
```

### 2. 生产就绪 🏭

**关键特性**:

| 特性 | 说明 | 生产价值 |
|-----|------|---------|
| **持久化执行** | 自动状态保存 | 服务重启不丢失进度 |
| **错误恢复** | 检查点机制 | 从失败点继续执行 |
| **并发控制** | 内置限流 | 避免 API 配额超限 |
| **监控集成** | LangSmith | 实时性能监控 |
| **版本管理** | 语义化版本 | 安全的依赖升级 |

**企业采用案例**:
- **Uber**: 客服自动化 Agent
- **LinkedIn**: 内容推荐 Agent
- **Klarna**: 购物助手 Agent

### 3. 灵活性 🔧

**多层次抽象**:

```
高抽象层 ──┐
          │  create_agent (快速开发)
          │  ↓
          │  Middleware (自定义行为)
          │  ↓
          │  LangGraph (完全控制)
          │  ↓
低抽象层 ──┘  自定义图原语
```

**适应不同场景**:
1. **原型验证** (1-2 天): 使用 `create_agent`
2. **MVP 开发** (1-2 周): 添加 Middleware
3. **生产优化** (1-2 月): 迁移到 LangGraph
4. **定制需求** (持续): 自定义图结构

### 4. 现代化 🚀

**最新 LLM 特性支持**:

| 特性 | 支持模型 | LangChain 抽象 |
|-----|---------|---------------|
| **推理块** | Claude, GPT-4o | `content_blocks` |
| **引用** | Gemini, Claude | `content_blocks` |
| **视觉输入** | GPT-4V, Claude | `content=[image, text]` |
| **结构化输出** | All | `with_structured_output()` |
| **函数调用** | All | `bind_tools()` |

### 5. 可观测性 👁️

**多维度监控**:

**1. 性能维度**
```python
# LangSmith 自动收集
{
    "latency_ms": 1234,
    "token_usage": {"input": 500, "output": 200},
    "cost_usd": 0.05,
    "cache_hit_rate": 0.75
}
```

---

## 设计哲学演变

### 时间线 📅

```
2022.10 ────┬──── LangChain 诞生
            │     - Chains 概念
            │     - 预定义流程
            │
2023.06 ────┼──── 快速增长
            │     - 大量 Chain 类型
            │     - 社区贡献
            │
2024.02 ────┼──── LangGraph 发布
            │     - 低层编排控制
            │     - 图原语
            │
2024.10 ────┼──── LangGraph 成为主流
            │     - 生产案例增多
            │     - 灵活性需求
            │
2025.10 ────┴──── v1.0 发布
                  - 统一架构
                  - 生产就绪
```

### 架构演变 🏗️

#### 第一代: Chains (2022-2023)

**设计思路**: 预定义的处理链

**局限性**:
- ❌ 灵活性差
- ❌ 难以定制
- ❌ 无法处理复杂流程

#### 第二代: LangGraph (2024)

**设计思路**: 图状态机

**优势**:
- ✅ 完全灵活
- ✅ 可视化流程
- ✅ 复杂逻辑支持

**挑战**:
- ❌ 学习曲线陡峭
- ❌ 简单场景过度工程

#### 第三代: 统一架构 (2025)

**设计思路**: 双层抽象

**核心价值**:
- ✅ 简单场景快速开发 (`create_agent`)
- ✅ 复杂场景完全控制 (`LangGraph`)
- ✅ 渐进式过渡路径
- ✅ 生产级可靠性

### 核心信念 💡

#### 1. 组合优于配置

**反模式** (配置驱动):
```python
agent = create_agent(
    config={
        "retry": True,
        "retry_max": 3,
        "retry_delay": 1,
        "log": True,
        "log_level": "INFO",
        # ... 无尽的配置
    }
)
```

**推荐模式** (组合驱动):
```python
agent = create_agent(model, tools, middleware=[
    RetryMiddleware(max_retries=3),
    LoggingMiddleware(level="INFO")
])
```

#### 2. 显式优于隐式

**反模式** (隐式行为):
```python
# 魔法字符串触发特殊行为
prompt = "Use tool: search"  # ❌ 隐式
```

**推荐模式** (显式调用):
```python
# 明确的工具绑定
model.bind_tools([search_tool])  # ✅ 显式
```

#### 3. 标准优于专有

**反模式** (专有格式):
```python
# 绑定特定提供商
openai_specific_feature()  # ❌ 锁定
```

**推荐模式** (标准接口):
```python
# 通用抽象
model.with_structured_output(Schema)  # ✅ 可移植
```

---

## 迁移指南

### 前置准备 📋

#### 1. 环境检查

**检查当前版本**:
```bash
pip show langchain langchain-core langgraph
```

**依赖审计**:
```bash
# 列出所有 langchain 相关包
pip list | grep langchain

# 检查使用的已弃用 API
python -W default::DeprecationWarning your_app.py
```

#### 2. 备份

**代码备份**:
```bash
git checkout -b pre-v1-backup
git commit -am "Backup before LangChain v1.0 migration"
```

**依赖固定**:
```bash
pip freeze > requirements-old.txt
```

### 迁移步骤 🔄

#### 步骤 1: 升级依赖

**保守升级** (保留兼容性):
```bash
# 安装 v1.0 + classic
pip install -U langchain langchain-core langgraph
pip install langchain-classic
```

**激进升级** (完全迁移):
```bash
# 仅安装 v1.0
pip install -U langchain langchain-core langgraph
# 不安装 langchain-classic
```

#### 步骤 2: 迁移 Agent

**示例: 基础 Agent**
```python
# ❌ 旧代码
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
result = agent.run("查询天气")

# ✅ 新代码
from langchain.agents import create_agent

agent = create_agent(model=model, tools=tools)
result = agent.invoke({"messages": [{"role": "user", "content": "查询天气"}]})
```

#### 步骤 3: 迁移 Chains

**简单 Chain → 直接使用模型**:
```python
# ❌ LLMChain
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)

# ✅ LCEL (LangChain Expression Language)
chain = prompt | model
```

#### 步骤 4: 测试

**单元测试模板**:
```python
import pytest
from langchain.agents import create_agent

@pytest.fixture
def agent():
    return create_agent(model=model, tools=tools)

def test_basic_query(agent):
    result = agent.invoke({
        "messages": [{"role": "user", "content": "测试查询"}]
    })
    assert "messages" in result
    assert len(result["messages"]) > 0
```

### 常见问题处理 🔧

#### 问题 1: 导入错误

**错误信息**:
```
ImportError: cannot import name 'AgentExecutor' from 'langchain.agents'
```

**解决方案**:
```bash
# 选项 1: 安装 classic
pip install langchain-classic

# 选项 2: 迁移到新 API
# 见上文迁移示例
```

#### 问题 2: 工具格式不兼容

**错误信息**:
```
TypeError: Tool must have 'name' and 'func' attributes
```

**解决方案**:
```python
# ❌ 旧格式
from langchain.agents import Tool

tool = Tool(
    name="search",
    func=search_function,
    description="搜索工具"
)

# ✅ 新格式
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索工具"""
    return search_function(query)
```

### 迁移检查清单 ✅

**代码层面**:
- [ ] 所有导入路径已更新
- [ ] `AgentExecutor` 已替换为 `create_agent`
- [ ] `LLMChain` 已迁移或移除
- [ ] 消息对象使用 `.text` 属性而非 `.text()`
- [ ] 运行时配置使用 `context` 而非 `config["configurable"]`
- [ ] 工具定义使用新格式
- [ ] 流式处理适配新节点名称

**测试层面**:
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 性能测试对比（确保无退化）
- [ ] 边界案例测试

**文档层面**:
- [ ] README 更新
- [ ] API 文档更新
- [ ] 迁移说明编写
- [ ] 依赖版本固定

**运维层面**:
- [ ] 依赖更新到生产环境
- [ ] 监控指标更新（适配 LangSmith）
- [ ] 回滚方案准备
- [ ] 团队培训完成

---

## 最佳实践

### 1. Agent 设计模式 🎨

#### 模式 1: 单一职责 Agent

**原则**: 每个 Agent 专注一个领域

```python
# ✅ 好的设计
customer_service_agent = create_agent(
    model=model,
    tools=[search_kb, create_ticket, escalate],
    system_prompt="你是客服专员，专注于解决客户问题"
)

sales_agent = create_agent(
    model=model,
    tools=[search_products, calculate_price, create_order],
    system_prompt="你是销售专员，专注于产品推荐和订单"
)

# ❌ 不好的设计
everything_agent = create_agent(
    model=model,
    tools=[...100 个工具],  # 过多职责
    system_prompt="你什么都能做"
)
```

#### 模式 2: 分层 Agent 架构

**场景**: 复杂业务流程

```python
# 顶层：路由 Agent
router_agent = create_agent(
    model=model,
    tools=[
        delegate_to_customer_service,
        delegate_to_sales,
        delegate_to_technical
    ],
    system_prompt="根据用户需求路由到专门的 Agent"
)

# 底层：专业 Agent
def delegate_to_customer_service(query: str) -> str:
    """委托给客服 Agent"""
    result = customer_service_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1]["content"]
```

#### 模式 3: 工具优先设计

**原则**: 复杂逻辑封装为工具

```python
# ✅ 好的设计：逻辑在工具中
@tool
def complex_calculation(params: dict) -> float:
    """复杂的业务计算逻辑"""
    # 100 行计算逻辑
    return result

agent = create_agent(
    model=model,
    tools=[complex_calculation],
    system_prompt="使用工具进行计算"
)

# ❌ 不好的设计：依赖 LLM 计算
agent = create_agent(
    model=model,
    tools=[],
    system_prompt="请计算以下复杂公式: ..."  # LLM 容易出错
)
```

### 2. 工具开发最佳实践 🔨

#### 规范的工具定义

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """搜索输入参数"""
    query: str = Field(description="搜索查询词")
    limit: int = Field(default=10, description="返回结果数量")
    filters: dict = Field(default={}, description="过滤条件")

@tool(args_schema=SearchInput)
def search_database(query: str, limit: int = 10, filters: dict = {}) -> list:
    """
    在数据库中搜索内容

    使用场景：
    - 查找用户信息
    - 搜索历史记录
    - 检索产品数据

    Args:
        query: 搜索关键词，支持模糊匹配
        limit: 最多返回多少条结果，默认 10
        filters: 额外的过滤条件，如 {"status": "active"}

    Returns:
        匹配的记录列表

    Examples:
        >>> search_database("张三", limit=5)
        [{"name": "张三", "id": 1}, ...]
    """
    # 实现搜索逻辑
    results = db.query(query, limit=limit, **filters)
    return results
```

#### 工具错误处理

```python
from langchain_core.tools import ToolException

@tool
def risky_operation(param: str) -> str:
    """可能失败的操作"""
    try:
        result = external_api_call(param)
        return result
    except ExternalAPIError as e:
        # 转换为 ToolException，Agent 可以理解
        raise ToolException(
            f"API 调用失败: {e}. 请稍后重试或联系管理员。"
        )
    except ValueError as e:
        # 参数错误
        raise ToolException(f"参数无效: {e}")
```

### 3. 提示词工程 📝

#### 结构化系统提示

```python
SYSTEM_PROMPT = """
# 角色定义
你是一个专业的数据分析助手。

# 核心能力
- 数据查询和分析
- 生成可视化图表
- 提供业务洞察

# 工作流程
1. 理解用户需求
2. 选择合适的工具
3. 分析数据结果
4. 生成清晰的报告

# 输出格式
- 使用 Markdown 格式
- 数据用表格展示
- 结论要简明扼要

# 限制
- 不要编造数据
- 不确定时明确说明
- 超出能力范围时建议人工介入
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)
```

### 4. 性能优化 ⚡

#### 并行工具调用

```python
# LangChain 会自动并行执行独立的工具调用
# 确保工具是线程安全的

import asyncio
from langchain_core.tools import tool

@tool
async def async_search(query: str) -> str:
    """异步搜索工具"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/search?q={query}") as resp:
            return await resp.text()

# Agent 会自动并行调用多个 async_search
```

#### 缓存策略

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# 开发环境：内存缓存
set_llm_cache(InMemoryCache())

# 生产环境：持久化缓存
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# 自动缓存相同的 LLM 调用
agent.invoke({"messages": [...]})  # 首次调用 API
agent.invoke({"messages": [...]})  # 从缓存读取
```

### 5. 监控和调试 🔍

#### 结构化日志

```python
import structlog

logger = structlog.get_logger()

class LoggingMiddleware(Middleware):
    def before_model(self, state, config):
        logger.info(
            "model_call_start",
            message_count=len(state.get("messages", [])),
            user_id=config.get("context", {}).get("user_id")
        )
        return state

    def after_model(self, state, response, config):
        logger.info(
            "model_call_end",
            response_length=len(response.content),
            tool_calls=len(response.tool_calls or [])
        )
        return response
```

### 6. 安全最佳实践 🔒

#### 输入验证

```python
from pydantic import validator

class SecureInput(BaseModel):
    user_query: str

    @validator("user_query")
    def validate_query(cls, v):
        # 检查注入攻击
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT"]
        if any(word in v.upper() for word in forbidden):
            raise ValueError("检测到潜在的 SQL 注入")

        # 长度限制
        if len(v) > 1000:
            raise ValueError("查询过长")

        return v
```

#### 工具权限控制

```python
from functools import wraps

def require_permission(permission: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, config=None, **kwargs):
            user_permissions = config.get("context", {}).get("permissions", [])
            if permission not in user_permissions:
                raise PermissionError(f"需要权限: {permission}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@tool
@require_permission("database.write")
def delete_record(record_id: str) -> str:
    """删除记录（需要写权限）"""
    db.delete(record_id)
    return "已删除"
```

---

## 参考资源

### 官方文档 📚

| 资源 | 链接 | 说明 |
|-----|------|------|
| **LangChain 文档** | https://docs.langchain.com/ | 完整的官方文档 |
| **v1.0 发布说明** | https://docs.langchain.com/oss/python/releases/langchain-v1 | 新特性介绍 |
| **迁移指南** | https://docs.langchain.com/oss/python/migrate/langchain-v1 | 详细的迁移步骤 |
| **设计哲学** | https://docs.langchain.com/oss/python/langchain/philosophy | 架构理念 |
| **API 参考** | https://api.python.langchain.com/ | 完整的 API 文档 |

### 博客文章 📝

| 标题 | 链接 |
|-----|------|
| LangChain & LangGraph 1.0 发布 | https://blog.langchain.com/langchain-langgraph-1dot0/ |
| LangChain v1.0 Alpha | https://blog.langchain.com/langchain-langchain-1-0-alpha-releases/ |

### 社区资源 👥

| 资源 | 链接 |
|-----|------|
| **GitHub 仓库** | https://github.com/langchain-ai/langchain |
| **Discord 社区** | https://discord.gg/langchain |
| **Twitter** | https://twitter.com/LangChainAI |
| **YouTube** | https://www.youtube.com/@LangChain |

### 学习路径 🎓

#### 初学者
1. 快速开始 - 15 分钟
2. 构建第一个 Agent - 30 分钟
3. 工具使用教程 - 1 小时

#### 进阶
1. 中间件深入 - 2 小时
2. LangGraph 基础 - 3 小时
3. 生产部署指南 - 4 小时

#### 专家
1. 自定义图架构 - 1 天
2. 性能优化 - 2 天
3. 企业最佳实践 - 1 周

---

## 总结 🎯

LangChain v1.0 标志着该框架从**实验性工具**到**生产级平台**的重要转变：

### 关键成就 ✨

1. **架构统一**: 单一 Agent 抽象 + LangGraph 双层设计
2. **开发简化**: 10 行代码构建 Agent，学习曲线大幅降低
3. **生产就绪**: 持久化、监控、错误处理等企业级特性
4. **标准化**: 跨提供商的统一接口，避免锁定
5. **灵活性**: 从高层抽象到底层控制的渐进式路径

### 适用场景 📊

| 场景 | 推荐方案 | 开发周期 |
|-----|---------|---------|
| 快速原型 | `create_agent` | 1-3 天 |
| MVP 产品 | `create_agent` + 中间件 | 1-2 周 |
| 生产应用 | LangGraph | 1-3 月 |
| 复杂系统 | 自定义图 + 多 Agent | 3+ 月 |

### 下一步行动 🚀

**如果你是新用户**:
1. ✅ 安装 LangChain v1.0
2. ✅ 完成快速开始教程
3. ✅ 构建第一个 Agent
4. ✅ 尝试添加自定义工具

**如果你要迁移**:
1. ✅ 阅读迁移指南
2. ✅ 在测试环境升级
3. ✅ 逐步迁移代码
4. ✅ 完整测试后部署

**持续学习**:
- 📚 订阅 LangChain 博客
- 💬 加入 Discord 社区
- 🎥 关注 YouTube 频道
- 🔔 GitHub Star 并 Watch 仓库

---

**文档版本**: 1.0.0  
**最后更新**: 2025-11-08  
**维护者**: LangChain 中文社区  
**许可**: MIT License

如有问题或建议，欢迎提交 Issue 或 PR！
