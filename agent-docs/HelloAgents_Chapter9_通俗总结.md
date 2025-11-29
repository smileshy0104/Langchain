# Hello Agents 第九章：上下文工程（通俗总结）

> **本章核心思想**：如果说 Prompt Engineering 是在教模型"怎么说话"，那么 Context Engineering 就是在教模型"怎么思考"。我们要构建一个从"收集"到"压缩"的自动化流水线，让 Agent 在有限的 Token 预算内，始终拥有最关键的"短期记忆"。

---

## 📖 目录

- [1. 什么是上下文工程？](#1-什么是上下文工程)
- [2. 核心理论：有限的注意力](#2-核心理论有限的注意力)
- [3. 核心组件：ContextBuilder](#3-核心组件contextbuilder)
- [4. 新增工具：NoteTool & TerminalTool](#4-新增工具notetool--terminaltool)
- [5. 实战：��建上下文感知的 Agent](#5-实战构建上下文感知的-agent)
- [6. 本章总结](#6-本章总结)

---

## 1. 什么是上下文工程？

### 🤔 为什么之前的 Agent 聊久了就"变笨"？

你是否发现，和 AI 聊得越久，它越容易：
*   忘记你一开始说的话（**遗忘**）
*   把不相关的信息混在一起（**幻觉**）
*   反应变慢，甚至报错（**Token 超限**）

这是因为 LLM 的"大脑容量"（Context Window）是有限的。

### 💡 上下文工程 (Context Engineering)

上下文工程不仅仅是写好 Prompt，而是一套**动态管理信息**的系统工程：
*   **筛选**：从海量历史中挑出最有用的信息。
*   **压缩**：把长篇大论浓缩成精华。
*   **结构化**：把信息分门别类地喂给模型。

> **比喻**：这就像考试前的复习。书（数据）很厚，但你进考场（Context Window）只能带一张小纸条。上下文工程就是**决定在这张小纸条上写什么**的艺术。

---

## 2. 核心理论：有限的注意力

LLM 存在一个**上下文腐蚀 (Context Rot)** 现象：**随着上下文越长，模型从中提取准确信息的能力反而下降**。

### 🎯 "针堆找针"理论

如果把有用的信息比作一根针，扔进一堆干草（无关信息）里：
*   **干��越少**，针越好找。
*   **干草越多**，针越难找，模型越容易"走神"。

所以，我们的目标是：**用最少的 Token，提供最高密度的信息**。

### 🏗️ 上下文的解剖学

一个优秀的 Context 应该包含：
1.  **系统提示 (System Prompt)**：不仅要有指令，还要有清晰的边界（Role & Policies）。
2.  **工具 (Tools)**：给模型一把好用的"瑞士军刀"，而不是一堆生锈的铁片。
3.  **示例 (Few-shot)**：给几个好例子，胜过千言万语的解释。
4.  **动态信息**：按需加载的"即时上下文"（Just-in-Time Context）。

---

## 3. 核心组件：ContextBuilder

HelloAgents 引入了一个全新的核心组件 `ContextBuilder`，它像一个高效的图书管理员，专门负责整理要喂给 LLM 的信息。

### 🔄 GSSC 流水线

ContextBuilder 遵循 **GSSC** 流程，把杂乱的信息变成高质量的上下文：

1.  **Gather (收集)** 📥
    *   把所有可能有关的信息（对话历史、记忆、RAG 检索结果、笔记）都抓过来。
    *   *就像把书架上所有相关的书都搬到桌子上。*

2.  **Select (选择)** 🔍
    *   根据**相关性**（和当前问题有多大关系）和**新近性**（是不是最近发生的）打分。
    *   只保留分数最高的那些。
    *   *就像从书堆里挑出最核心的几本。*

3.  **Structure (结构化)** 📑
    *   把选出来的信按固定模板排好队：`[Role]`, `[Task]`, `[Evidence]`, `[Context]`, `[Output]`。
    *   *就像把书的内容整理成条理清晰的笔记。*

4.  **Compress (压缩)** 🗜️
    *   如果还是超长了，就进行智能截断或摘要，确不爆 Token。
    *   *就像把笔记再缩写一遍，确保能写在小纸条上。*

### 💻 代码演示

```python
from hello_agents.context import ContextBuilder, ContextConfig

# 1. 初始化构建器
builder = ContextBuilder(
    memory_tool=memory_tool,
    rag_tool=rag_tool,
    config=ContextConfig(max_tokens=4000)  # 设定预算
)

# 2. 构建上下文（一键搞定 GSSC 流程）
context = builder.build(
    user_query="如何优化 Pandas 内存？",
    conversation_history=history,
    system_instructions="你是一位数据专家..."
)

# 3. 得到的 context 是结构化、优化过的，直接喂给 LLM
llm.invoke(context)
```

---

## 4. 新增工具：NoteTool & TerminalTool

为了配合长时程任务和即时探索，本章新增了两个强力工具。

### 📝 NoteTool：结构化笔记

Agent 也是会"忘事"的，尤其是跨越多天的任务。`NoteTool` 让 Agent 像人一样记笔记。

*   **特点**：Markdown + YAML 格式，人机都能读。
*   **用途**：记录任务状态 (`task_state`)、阻塞点 (`blocker`)、结论 (`conclusion`)。

```python
# Agent 觉得这个信息很重要，记下来
note_tool.run({
    "action": "create",
    "title": "项目重构-第一阶段状态",
    "content": "已完成数据层重构，遇到依赖冲突问题...",
    "note_type": "task_state",
    "tags": ["blocker", "refactoring"]
})
```

### 💻 TerminalTool：即时文件系统访问

有时候，RAG 预先索引太慢了，或者 Agent 需要看最新的日志。`TerminalTool` 允许 Agent 安全地执行命令行。

*   **安全第一**：有白名单（只能 `ls`, `cat`, `grep` 等），有沙箱（只能访问工作目录），有超时限制。
*   **用途**：即时查看代码结构、分析日志文件、预览数据。

```python
# Agent 想看看项目结构
terminal.run({"command": "tree -L 2 src"})

# Agent 想找报错日志
terminal.run({"command": "grep 'ERROR' app.log | tail -n 10"})
```

---

## 5. 实战：构建上下文感知的 Agent

结合以上所有内容，我们构建了一个**长期项目助手**。它不再是"聊完即忘"的聊天机器人，而是一个能陪你一起做项目的伙伴。

### 🤖 这个 Agent 能做什么？

1.  **自动回溯**��每次你说话，它会自动去 RAG 和 Memory 里找相关信息，拼装进 Context。
2.  **记笔记**：聊到关键点，它会自动调用 `NoteTool` 记下来。
3.  **即时探索**：遇到不确定的文件，它会用 `TerminalTool` 去看一眼。
4.  **永远"在线"**：通过 ContextBuilder，它永远清楚项目的当前状态、阻塞点和下一步计划。

### 🌟 效果对比

*   **普通 Agent**：
    *   用户："那个依赖冲突解决了吗？"
    *   Agent："什么依赖冲突？请提供更多细节。"（遗忘）

*   **上下文感知 Agent**：
    *   用户："那个依赖冲突解决了吗？"
    *   Agent："（检索笔记 -> 发现 blocker -> 构建上下文）根据昨天的笔记，我们在重构业务层时遇到了 `pandas` 版本冲突。目前还没有记录解决结果，您尝试过锁定版本吗？"（智能、连贯）

---

## 6. 本章总结

### 🌟 核心收获

1.  **Context Engineering > Prompt Engineering**：我们在运行时动态管理信息，而不仅是静态地写 Prompt。
2.  **ContextBuilder**：我们有了一个自动化的流水线，把"找信息"和"塞给模型"这件事标准化了。
3.  **长期记忆与即时探索**：通过 NoteTool 和 TerminalTool，Agent 的能力边界大幅扩展。

### 🚀 下一步是��么？

现在 Agent 已经很聪明了，能记忆、能检索、能记笔记。但是，如果我们要让**多个** Agent 协同工作呢？比如一个负责写代码，一个负责测试，一个负责写文档？

下一章（第十章），我们将探索**智能体通信协议 (Agent Protocols)**，学习 MCP、A2A 等协议，让 Agent 们像团队一样协作！

---

### 🔗 快速传送门
- **GitHub 源码**: [hello-agents/chapter9](https://github.com/jjyaoao/helloagents)
- **安装命令**: `pip install "hello-agents[all]"`
