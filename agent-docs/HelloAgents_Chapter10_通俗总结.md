# Hello Agents 第十章：智能体通信协议（通俗总结）

> **本章核心思想**：单个智能体能力再强也是"孤胆英雄"。要构建复杂的 AI 系统，必须让智能体学会"社交"。本章引入了三大通信协议：**MCP**（连接工具）、**A2A**（连接队友）、**ANP**（连接世界），为智能体搭建起沟通的桥梁。

---

## 📖 目录

- [1. 为什么智能体需要"通信协议"？](#1-为什么智能体需要通信协议)
- [2. 三大协议天团：MCP、A2A、ANP](#2-三大协议天团mcp-a2a-anp)
- [3. MCP 实战：给智能体装上"万能插座"](#3-mcp-实战给智能体装上万能插座)
- [4. A2A 实战：智能体之间的"团队通话"](#4-a2a-实战智能体之间的团队通话)
- [5. ANP 实战：构建智能体"互联网"](#5-anp-实战构建智能体互联网)
- [6. 本章总结](#6-本章总结)

---

## 1. 为什么智能体需要"通信协议"？

### 🤔 传统开发的痛点

以前我们给 Agent 加功能（比如查天气、读数据库），每加一个都要写一堆适配代码：
*   写 GitHub 适配器... ✍️
*   写数据库适配器... ✍️
*   写 Slack 适配器... ✍️

这就像早期的手机充电器，每个品牌都有自己的接口，互不通用。

### 💡 通信协议的价值

通信协议就是智能体世界的 **USB-C 接口** 和 **TCP/IP 协议**。
*   **标准化**：写一次代码，连接所有服务。
*   **互操作**：你的 Agent 可以直接用我写的工具。
*   **动态发现**：Agent 可以自己去网上找"谁能帮我干活"。

---

## 2. 三大协议天团：MCP、A2A、ANP

HelloAgents 引入了三种不同层级的协议，分别解决不同的沟通问题：

| 协议 | 全称 | 核心作用 | 形象比喻 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **MCP** | Model Context Protocol | **Agent ↔ 工具** | **USB-C 接口**<br>统一连接各种外设（工具/数据） | 读文件、查库、调 API |
| **A2A** | Agent-to-Agent | **Agent ↔ Agent** | **电话/微信**<br>智能体之间点对点沟通协作 | 研究员与写手协作 |
| **ANP** | Agent Network Protocol | **Agent ↔ 网络** | **互联网/黄页**<br>在大规模网络中发现服务 | 寻找"最便宜的算力节点" |

---

## 3. MCP 实战：给智能体装上"万能插座"

MCP (Model Context Protocol) 是由 Anthropic 提出的标准，旨在统一 AI 模型与外部数据的连接方式。

### 🔌 MCP 的核心能力
1.  **Tools (工具)**：主动执行操作（如：`read_file`, `execute_sql`）。
2.  **Resources (资源)**：被动读取数据（如：读取 API 返回的 JSON）。
3.  **Prompts (提示)**：提供预设的 Prompt 模板。

### 💻 代码演示：一键连接 GitHub

以前要写几十行代码对接 GitHub API，现在只需要一行配置：

```python
from hello_agents.tools import MCPTool

# 直接使用社区提供的 GitHub MCP 服务
github_tool = MCPTool(
    name="gh", 
    server_command=["npx", "-y", "@modelcontextprotocol/server-github"]
)

# Agent 瞬间拥有了搜索仓库、查看 Issue、提交代码的能力
agent.add_tool(github_tool)
agent.run("在 GitHub 上搜索关于 'AI Agents' 的 Python 项目")
```

### 🚀 自动展开机制

HelloAgents 的 `MCPTool` 有个黑科技叫**自动展开**。你只加了一个 `github_tool`，Agent 实际上会自动获得 `gh_search_repositories`, `gh_read_file` 等十几个具体工具，模型可以根据描述自动选择用哪个。

---

## 4. A2A 实战：智能体之间的"团队通话"

A2A (Agent-to-Agent) 让智能体不再是单打独斗，而是像人类团队一样协作。

### 🤝 核心概念
*   **Task (任务)**：我要在这个时间点前完成这件事。
*   **Artifact (工件)**：这是我做完东西（文档、代码）。
*   **协商**：Agent A 说"我要 100 块"，Agent B 说"太贵了，50 行不行"。

### 💻 代码演示：研究员与撰写员的协作

```python
from hello_agents.protocols import A2AServer, A2AClient

# 1. 启动一个"研究员" Agent 服务
researcher = A2AServer(name="researcher")
@researcher.skill("research")
def do_research(topic):
    return f"关于 {topic} 的详细资料..."
researcher.run(port=5000)

# 2. 另一个"主编" Agent 调用它
client = A2AClient("http://localhost:5000")
# 直接像调用本地函数一样调用远程 Agent 的技能
result = client.execute_skill("research", "AI 发展史")
```

---

## 5. ANP 实战：构建智能体"互联网"

ANP (Agent Network Protocol) 解决了"去哪里找人"的问题。当网络里有成千上万个 Agent 时，ANP 充当了服务���现中心。

### 🌍 核心功能
*   **服务注册**：Agent 上线后大喊一声："我是做图的，价格便宜！"
*   **服务发现**：用户搜索："我要找个能做图的，评价最高的。"
*   **智能路由**：系统自动把任务派发给当前负载最低的 Agent。

### 💻 代码演示：智能算力调度

```python
from hello_agents.protocols import ANPDiscovery, register_service

# 1. 注册 10 个计算节点 Agent
for i in range(10):
    register_service(
        service_name=f"节点{i}", 
        capabilities=["计算", "渲染"],
        metadata={"load": random.random()}  # 模拟负载
    )

# 2. 调度器 Agent 寻找最佳节点
scheduler.run("帮我找一个当前负载最低的计算节点，执行渲染任务")
# Agent 会自动查询 ANP 发现服务，对比 metadata，选出最优解
```

---

## 6. 本章总结

### 🌟 核心收获

1.  **协议 > 代码**：掌握了协议，你的 Agent 就能连接全世界的工具和其他 Agent，而不需要自己重造轮子。
2.  **MCP 是当下主流**：目前最实用的工具连接协议，生态最丰富（GitHub, SQLite, Google Drive 等都有现成服务）。
3.  **从单体到群体**：A2A 和 ANP 为构建"多智能体社会"（Multi-Agent Society）打下了基础。

### 🚀 下一步是什么？

现在我们的 Agent 已经非常强大了：
*   有大脑（LLM）
*   有手脚（Tools）
*   有记忆（Memory）
*   有书房（RAG）
*   懂社交（Protocols）

但是，怎么知道它到底有多强？怎么教它变得更强？

下一章（第十一章），我们将进入**Agentic-RL（智能体强化学习）**的领域，学习如何通过 SFT（监督微调）和 RL（强化学习）来训练和优化我们自己的 Agent 模型！

---

### 🔗 快速传送门
- **GitHub 源码**: [hello-agents/chapter10](https://github.com/jjyaoao/helloagents)
- **安装命令**: `pip install "hello-agents[protocol]"`
