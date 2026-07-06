# Deep Agents Event Streaming、Streaming 与 Grading Rubrics 详细指南

> 基于 Deep Agents 官方 Event streaming、Streaming、Grading rubrics 文档整理。本文聚焦 Deep Agents 如何实时输出 coordinator、subagent、tool call、LLM token、custom event，以及如何用 `RubricMiddleware` 在运行时引入 LLM-as-a-judge 评分闭环，让 agent 按 rubric 自检、修订并收敛到可验收结果。

## 目录

1. [整体理解](#整体理解)
2. [Event Streaming 与 Streaming 的区别](#event-streaming-与-streaming-的区别)
3. [Event Streaming 核心模型](#event-streaming-核心模型)
4. [Subagent Streams](#subagent-streams)
5. [Stream Messages](#stream-messages)
6. [Stream Tool Calls](#stream-tool-calls)
7. [Nested Work 与并发消费](#nested-work-与并发消费)
8. [Streaming 旧式 Chunk 模型](#streaming-旧式-chunk-模型)
9. [Namespaces](#namespaces)
10. [Updates、Tokens、Tool Calls 与 Custom Updates](#updatestokenstool-calls-与-custom-updates)
11. [v2 Streaming Format](#v2-streaming-format)
12. [Grading Rubrics 是什么](#grading-rubrics-是什么)
13. [RubricMiddleware 配置](#rubricmiddleware-配置)
14. [调用时传入 Rubric](#调用时传入-rubric)
15. [Rubric Verdicts](#rubric-verdicts)
16. [观察评分迭代进度](#观察评分迭代进度)
17. [持久化 Rubric 与恢复执行](#持久化-rubric-与恢复执行)
18. [评分工具与证据收集](#评分工具与证据收集)
19. [三者如何协同](#三者如何协同)
20. [典型使用场景](#典型使用场景)
21. [最佳实践](#最佳实践)
22. [故障排查](#故障排查)
23. [快速参考](#快速参考)
24. [资料来源](#资料来源)

---

## 整体理解

这三篇文档解决的是 Deep Agents 在生产体验中的两个问题：

| 主题 | 解决的问题 | 推荐入口 |
|------|------------|----------|
| Event streaming | 用 typed projection 方式分别消费 subagents、messages、tool calls、values、output | `agent.stream_events(..., version="v3")` |
| Streaming | 用 LangGraph streaming chunk 方式消费 updates、messages、custom 等流 | `agent.stream(..., version="v2")` |
| Grading rubrics | 让 agent 在运行时根据 rubric 被评分、收到反馈并继续修订 | `RubricMiddleware` |

一句话：

```text
Event streaming / Streaming 负责“看见 agent 正在做什么”；
Grading rubrics 负责“判断 agent 做得够不够好，并让它继续改”。
```

从产品视角看，二者经常一起使用：

```text
用户请求
  ↓
Deep Agent coordinator
  ├─ 实时输出主 agent 消息、token、tool call
  ├─ 委派 subagents，并实时输出每个 subagent 的进度
  └─ 生成结果
       ↓
Rubric grader sub-agent
  ├─ 按 rubric 检查结果
  ├─ 可调用测试/检查工具收集证据
  ├─ 输出 satisfied / needs_revision / failed / grader_error
  └─ 如需修订，将逐条反馈注入对话，驱动 agent 再跑一轮
```

---

## Event Streaming 与 Streaming 的区别

官方文档把二者分成两类接口。

| 维度 | Event streaming | Streaming |
|------|-----------------|-----------|
| API | `stream_events()` / `astream_events()` | `stream()` / `astream()` |
| 推荐状态 | 新应用推荐，Deep Agents v0.6 引入 typed-projection API | 仍可使用，基于 LangGraph streaming |
| 典型版本 | `version="v3"` | `version="v2"` |
| 消费方式 | `stream.messages`、`stream.subagents`、`stream.tool_calls` 等独立 iterator | 读取 chunk，再按 `chunk["type"]`、`chunk["ns"]`、`chunk["data"]` 分支 |
| 主要优势 | 不需要在同一个流里手动判断各种 `stream_mode` chunk | 与 LangGraph 传统 streaming 机制一致，适合已有代码 |
| Subagent 抽象 | 有 `stream.subagents` 产品级 projection | 通过 `subgraphs=True` 和 namespace 识别 subagent |
| UI 适配 | 更适合拆成 coordinator 区域、subagent 卡片、工具调用面板 | 更适合已有 LangGraph streaming UI |

选择建议：

| 需求 | 推荐 |
|------|------|
| 新项目要构建实时 UI | 优先使用 `stream_events(..., version="v3")` |
| 只关心 subagent 启动、完成、失败 | `stream.subagents` |
| 要分别展示主 agent 和子 agent 文本 | `stream.messages` + `subagent.messages` |
| 已经基于 `stream_mode` 处理 LangGraph 流 | 保留 `agent.stream(..., version="v2")` |
| 需要同时消费 updates、tokens、custom signals | `stream(..., stream_mode=[...], subgraphs=True, version="v2")` 或用 event streaming projection 拆开 |
| 需要监听 rubric 评分事件 | `stream_events(..., transformers=[CustomTransformer])` + `stream.custom` |

---

## Event Streaming 核心模型

Deep Agents 的 event streaming 在 LangGraph streaming 之上加了一层更贴近 Deep Agents 产品概念的 projection。最重要的是 `stream.subagents`：它把 coordinator 通过 `task` 工具委派出去的子任务，变成一个个可迭代、可观察的 subagent handle。

典型调用：

```python
stream = agent.stream_events(
    {
        "messages": [
            {"role": "user", "content": "Research AI safety and summarize findings"}
        ]
    },
    version="v3",
)

for subagent in stream.subagents:
    print(subagent.name, subagent.path, subagent.status)
```

Event streaming 的 projection 是惰性的：

| Projection | 说明 |
|------------|------|
| `stream.messages` | 顶层 coordinator 的消息流 |
| `stream.tool_calls` | 顶层 coordinator 的工具调用流 |
| `stream.subagents` | delegated `task` 调用对应的 subagent 流 |
| `stream.values` | 顶层状态值更新 |
| `stream.output` | 顶层最终输出 |
| `stream.custom` | 自定义事件，例如 rubric 评分事件 |

“惰性”意味着：你访问某个 projection 时才打开并消费对应流。例如只遍历 `stream.subagents` 时，不必同时遍历 `stream.messages`、`stream.tool_calls`、`stream.values`，也不必消费每个 subagent 的 token 或 tool call。

更准确地说，惰性发生在“客户端消费事件”的层面，而不是“agent 执行”的层面：

| 层面 | 含义 |
|------|------|
| Agent 运行本身 | coordinator、subagent、tool call、model call 仍会按任务需要正常执行 |
| Event stream projection | 客户端访问哪个 projection，就读取哪个 projection 的事件 |

例如下面这段代码只适合做“子任务状态面板”：

```python
stream = agent.stream_events(input, version="v3")

for subagent in stream.subagents:
    print(subagent.name, subagent.status)
```

如果还想展示子 agent 的实时文本，需要显式访问 `subagent.messages`：

```python
for subagent in stream.subagents:
    for message in subagent.messages:
        print(f"[{subagent.name}]", message.text)
```

如果还想展示子 agent 的工具调用，则再访问 `subagent.tool_calls`：

```python
for subagent in stream.subagents:
    for call in subagent.tool_calls:
        print(call.tool_name, call.input)
```

因此可以把惰性理解成：

```text
不必为了拿 subagent 状态而遍历 message/tool/value 等其他流；
需要哪个 projection，再读取哪个 projection。
```

---

## Subagent Streams

`stream.subagents` 适合展示“哪些子智能体正在工作、状态是什么、结果什么时候返回”。

每个 subagent handle 常见字段：

| 字段 | 含义 |
|------|------|
| `name` | 子智能体名称，来自 coordinator 调用 `task` 时传入的 `subagent_type` |
| `path` | 该 subagent 在 agent 运行树中的 namespace path |
| `status` | 生命周期状态，如 `started`、`completed`、`failed`、`interrupted` |
| `messages` | 该 subagent 产生的消息流 |
| `tool_calls` | 该 subagent 内部的工具调用流 |
| `subagents` | 该 subagent 再次委派出的嵌套 subagents |
| `values` | 该 subagent 的状态值更新 |
| `output` | 该 subagent 的最终状态或完成信号 |

只跟踪生命周期：

```python
stream = agent.stream_events(input, version="v3")

running = 0
completed = 0
failed = 0

for subagent in stream.subagents:
    running += 1
    print(f"{subagent.name}: started")

    try:
        _ = subagent.output
        running -= 1
        completed += 1
        print(f"{subagent.name}: completed")
    except Exception:
        running -= 1
        failed += 1
        print(f"{subagent.name}: failed")
```

适合 UI：

```text
Coordinator
  ├─ message stream
  └─ subagents
       ├─ researcher card
       │   ├─ status
       │   ├─ messages
       │   └─ tool calls
       └─ reviewer card
           ├─ status
           ├─ messages
           └─ output
```

---

## Stream Messages

Deep Agents 可以同时从 coordinator 和 delegated subagents 输出消息。

| 消息来源 | 读取方式 |
|----------|----------|
| 顶层 coordinator | `stream.messages` |
| 某个 subagent | `subagent.messages` |
| 嵌套 subagent | 递归访问 `subagent.subagents` 后读取其 `messages` |

示例：

```python
stream = agent.stream_events(input, version="v3")

for message in stream.messages:
    print("[coordinator]", message.text)

for subagent in stream.subagents:
    for message in subagent.messages:
        print(f"[{subagent.name}]", message.text)
```

注意点：

| 注意点 | 说明 |
|--------|------|
| coordinator 和 subagent 输出可能交错 | 如果要实时 UI，通常需要并发消费 |
| `subagent.name` 是产品语义 | 适合展示 `researcher`、`critic`、`planner` 等名称 |
| 原始事件仍可访问 | 如果要绝对到达顺序，可以直接遍历 raw protocol events |

---

## Stream Tool Calls

工具调用也分层：

| 工具调用位置 | 读取方式 |
|--------------|----------|
| coordinator 调用的工具 | `stream.tool_calls` |
| subagent 内部工具 | `subagent.tool_calls` |
| nested subagent 内部工具 | 递归读取 nested handle 的 `tool_calls` |

工具调用可用于展示：

| 展示信息 | 用途 |
|----------|------|
| `tool_name` | 告诉用户当前调用了什么能力 |
| `input` | 展示工具入参或调试信息 |
| `output_deltas` | 流式展示工具输出 |
| `output` | 工具完成后的完整结果 |
| `completed` | 判断工具是否结束 |
| `error` | 展示失败原因或做降级处理 |

示例：

```python
stream = agent.stream_events(input, version="v3")

for call in stream.tool_calls:
    print("[coordinator tool]", call.tool_name, call.input)
    print(call.completed, call.error)

for subagent in stream.subagents:
    for call in subagent.tool_calls:
        print(f"[{subagent.name} tool]", call.tool_name, call.input)
        for delta in call.output_deltas:
            print(delta, end="", flush=True)
```

工具调用面板可以按 agent tree 分组：

```text
main
  └─ task(researcher)
researcher
  ├─ web_search(...)
  └─ summarize(...)
reviewer
  └─ check_claims(...)
```

---

## Nested Work 与并发消费

Subagent 可以继续委派 nested subagents（子智能体）。Event streaming 支持递归观察：

```python
stream = agent.stream_events(input, version="v3")

for subagent in stream.subagents:
    print(f"subagent {subagent.name}: {subagent.status}")

    for tool_call in subagent.tool_calls:
        print(f"{tool_call.tool_name}({tool_call.input})")

    for nested in subagent.subagents:
        print(f"nested subagent {nested.name}: {nested.status}")
```

并发消费的原因：

| 情况 | 原因 |
|------|------|
| coordinator 正在输出总结 | subagent 可能同时输出中间结果 |
| 多个 subagent 并行 | 单线程顺序消费可能导致某些输出延迟显示 |
| UI 需要 live update | 需要把消息、工具调用、状态分别推给不同组件 |

异步代码可用 `astream_events()` 和 `asyncio.gather()` 分别消费 coordinator 与 subagents：

```python
import asyncio

stream = await agent.astream_events(input, version="v3")

async def consume_coordinator():
    async for message in stream.messages:
        print("[coordinator]", await message.text)

async def consume_subagents():
    async for subagent in stream.subagents:
        async for message in subagent.messages:
            print(f"[{subagent.name}]", await message.text)

await asyncio.gather(consume_coordinator(), consume_subagents())
```

同步代码可用 `stream.interleave(...)` 同时消费多个 projection：

```python
stream = agent.stream_events(input, version="v3")

for name, item in stream.interleave("messages", "subagents"):
    if name == "messages":
        print("[coordinator]", item.text)
    else:
        for message in item.messages:
            print(f"[{item.name}]", message.text)
```

如果要严格按事件到达顺序渲染，需要遍历 raw events，并用 `event["params"]["namespace"]` 判断来源。

---

## Streaming 旧式 Chunk 模型

`agent.stream()` 是基于 LangGraph streaming 的传统接口。Deep Agents 通过 `subgraphs=True` 把 subagent 执行事件暴露出来。

基本形态：

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Research quantum computing"}]},
    stream_mode="updates",
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "updates":
        if chunk["ns"]:
            print(f"[subagent: {chunk['ns']}]")
        else:
            print("[main agent]")
        print(chunk["data"])
```

关键参数：

| 参数 | 作用 |
|------|------|
| `stream_mode` | 指定流类型，如 `updates`、`messages`、`custom`，也可以传列表 |
| `subgraphs=True` | 让 subagent/subgraph 内部事件也被输出 |
| `version="v2"` | 使用统一 chunk 格式 |

适合使用 `agent.stream()` 的情况：

| 场景 | 说明 |
|------|------|
| 已有 LangGraph streaming 代码 | 迁移成本低 |
| 需要按 `stream_mode` 统一处理 | chunk 模式天然按模式分支 |
| 对内部 graph 节点也感兴趣 | `subgraphs=True` 暴露 namespace path |

---

## Namespaces

当 `subgraphs=True` 时，每个 chunk 都带 namespace。Namespace 是一个 path，用于说明事件来自主 agent 还是某个 subagent/subgraph。

| Namespace 示例 | 来源 |
|----------------|------|
| `()` | 顶层 main agent |
| `("tools:abc123",)` | main agent 的某个 `task` 工具调用启动的 subagent |
| `("tools:abc123", "model_request:def456")` | subagent 内部的模型请求节点 |

判断事件是否来自 subagent：

```python
is_subagent = any(
    segment.startswith("tools:")
    for segment in chunk["ns"]
)
```

提取 subagent 标识：

```python
tool_call_id = next(
    segment.split(":")[1]
    for segment in chunk["ns"]
    if segment.startswith("tools:")
)
```

与 event streaming 的区别：

| 方式 | 用户界面含义 |
|------|--------------|
| `stream.subagents` | 直接拿到 subagent handle，更贴近产品概念 |
| `chunk["ns"]` | 拿到执行路径，更贴近 graph 运行结构 |

因此，用户可见 UI 更推荐 `stream.subagents`；调试内部图执行时，namespace 仍然有价值。

---

## Updates、Tokens、Tool Calls 与 Custom Updates

### Updates

`stream_mode="updates"` 用于观察每个节点完成后的状态更新，适合展示步骤进度。

可展示：

| 信息 | 示例 |
|------|------|
| main agent 进入模型请求 | `[main agent] step: model_request` |
| subagent 开始运行 | `[tools:call_xxx] step: model_request` |
| subagent 调用工具 | `[tools:call_xxx] step: tools` |
| subagent 返回结果 | main agent 的 `tools` 节点出现 tool message |

### LLM Tokens

`stream_mode="messages"` 可流式输出 main agent 和 subagent 的 token。

关键点：

| 处理 | 说明 |
|------|------|
| 读取 `token, metadata = chunk["data"]` | token 是消息片段，metadata 是元数据 |
| 通过 `chunk["ns"]` 区分来源 | 空 namespace 是 main agent，包含 `tools:` 是 subagent |
| source 切换时打印 header | 便于 UI 把 token 路由到不同区域 |

### Tool Calls

工具调用也出现在 `messages` stream 中，通常通过 `token.tool_call_chunks` 检测。

可处理：

| 数据 | 说明 |
|------|------|
| `tc["name"]` | 工具名 |
| `tc["args"]` | 流式传入参数片段 |
| `token.type == "tool"` | 工具结果消息 |
| 普通 `ai` content | 非工具调用文本输出 |

### Custom Updates

工具内部可以用 `get_stream_writer()` 输出自定义进度。

示例形态：

```python
from langgraph.config import get_stream_writer

def analyze_data(topic: str) -> str:
    writer = get_stream_writer()
    writer({"status": "starting", "topic": topic, "progress": 0})
    writer({"status": "analyzing", "progress": 50})
    writer({"status": "complete", "progress": 100})
    return "analysis result"
```

消费方式：

```python
for chunk in agent.stream(
    input,
    stream_mode="custom",
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "custom":
        print(chunk["ns"], chunk["data"])
```

### 多模式同时消费

`stream_mode` 可以传列表：

```python
for chunk in agent.stream(
    input,
    stream_mode=["updates", "messages", "custom"],
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "updates":
        ...
    elif chunk["type"] == "messages":
        ...
    elif chunk["type"] == "custom":
        ...
```

这适合构建完整运行面板：

```text
Execution panel
  ├─ updates: 当前节点和步骤
  ├─ messages: token 流
  ├─ custom: 工具自定义进度
  └─ namespace: 路由到 main 或 subagent
```

---

## v2 Streaming Format

Deep Agents streaming 文档推荐 `version="v2"`。v2 的 chunk 统一为：

```python
{
    "type": "updates" | "messages" | "custom",
    "ns": (),
    "data": ...
}
```

相比旧格式，v2 避免了嵌套 tuple 拆包：

| 格式 | 处理方式 |
|------|----------|
| v2 | `chunk["type"]`、`chunk["ns"]`、`chunk["data"]` |
| v1 | 需要处理 `(namespace, (mode, data))` 这类嵌套结构 |

建议：

```text
新代码使用 version="v2" 的 agent.stream()
新应用优先考虑 version="v3" 的 agent.stream_events()
```

---

## Grading Rubrics 是什么

Grading rubrics 用 `RubricMiddleware` 给 Deep Agent 增加运行时评分闭环。它使用 LLM-as-a-judge 模式：一个专门的 grader 模型检查工作模型的输出是否满足 rubric。

适合场景：

| 场景 | Rubric 示例 |
|------|-------------|
| 格式严格 | 必须三行、必须 JSON、必须包含指定字段 |
| 内容覆盖 | 报告必须包含背景、方法、风险、结论 |
| 代码生成 | 所有测试通过、函数签名正确、处理边界条件 |
| 重构任务 | 行为不变、测试通过、删除重复代码 |
| 文案任务 | 语气符合品牌、长度限制、包含 CTA |

工作流：

```text
1. Agent 根据用户消息生成输出
2. Grader sub-agent 读取 transcript 与 rubric
3. Grader 输出 verdict
4. 如果 satisfied，结束
5. 如果 needs_revision，逐条反馈注入对话
6. Agent 根据反馈继续修订
7. 循环直到 satisfied、failed、grader_error 或达到 max_iterations
```

与 LangSmith 离线评估的区别：

| 维度 | LangSmith evaluations | RubricMiddleware |
|------|-----------------------|------------------|
| 发生时间 | 离线或批量评估 | agent 运行时 |
| 目的 | 衡量应用质量、比较实验 | 驱动当前任务自动修订 |
| 输出 | score / feedback / experiment result | verdict / per-criterion feedback / 继续迭代 |
| 是否影响当前回答 | 通常不影响 | 直接影响，可能触发再生成 |

---

## RubricMiddleware 配置

基本配置：

```python
from deepagents import RubricMiddleware, create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_deep_agent(
    model="openai:gpt-5.5",
    middleware=[
        RubricMiddleware(
            model="anthropic:claude-haiku-4-5",
            max_iterations=3,
        ),
    ],
    checkpointer=InMemorySaver(),
)
```

核心参数：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | 是 | `None` | grader 使用的 chat model，可传 `"provider:model-id"` 或 `BaseChatModel` |
| `system_prompt` | 否 | 内置 grader prompt | 自定义评分说明 |
| `tools` | 否 | `None` | grader 可调用的工具，用于跑测试、读文件、统计字数、检查输出等 |
| `max_iterations` | 否 | `3` | 每次 rubric attempt 的最大评分迭代次数，输入上限为 20 |
| `on_evaluation` | 否 | `None` | 每次评分后调用的 callback，用于日志、指标、UI 或数据集记录 |

模型选择建议：

| 选择 | 说明 |
|------|------|
| grader 可比 worker 更小更便宜 | 只要能稳定检查 rubric 即可 |
| 复杂任务可用更强 grader | 例如代码审查、安全审查、法律文本检查 |
| 需要客观判断时给 grader 工具 | 例如测试工具、schema validator、文件读取工具 |
| 不要让 grader 只凭感觉判断可测试条件 | 可测试条件应尽量工具化 |

---

## 调用时传入 Rubric

`RubricMiddleware` 配好后，并不是每次都会运行。只有调用 state 中传入 `rubric` 时，才会启动评分闭环。

阻塞调用：

```python
from langchain.messages import HumanMessage

result = agent.invoke(
    {
        "messages": [HumanMessage("Write a haiku about spring.")],
        "rubric": (
            "- The poem has three lines\n"
            "- Lines follow a 5-7-5 syllable pattern\n"
            "- The theme is spring"
        ),
    },
    config={"configurable": {"thread_id": "rubric-thread"}},
)
```

流式观察评分事件：

```python
from langgraph.stream import CustomTransformer

stream = agent.stream_events(
    {
        "messages": [HumanMessage("Write a haiku about spring.")],
        "rubric": (
            "- The poem has three lines\n"
            "- Lines follow a 5-7-5 syllable pattern\n"
            "- The theme is spring"
        ),
    },
    config={"configurable": {"thread_id": "rubric-thread"}},
    version="v3",
    transformers=[CustomTransformer],
)

for event in stream.custom:
    if event.get("type") == "rubric_evaluation_start":
        print("grading starts", event["iteration"])
    elif event.get("type") == "rubric_evaluation_end":
        print("verdict", event["result"], event.get("explanation", ""))
```

Rubric 写法建议：

| 好的 rubric | 不好的 rubric |
|-------------|---------------|
| 每条标准可判断 | “答案要好” |
| 每条标准尽量单一 | “准确、完整、简洁、专业” 合在一条 |
| 包含可测试条件 | “所有 `run_test_suite` 测试通过” |
| 明确失败时应反馈什么 | “若缺少字段，指出缺少字段名” |
| 与用户目标对齐 | 与任务无关的审美要求 |

---

## Rubric Verdicts

Grader 会给出以下结果：

| Status | 含义 | 是否继续循环 |
|--------|------|--------------|
| `satisfied` | 所有 rubric criteria 都通过 | 否 |
| `needs_revision` | 至少一个 criterion 失败，反馈会注入给 agent | 是 |
| `max_iterations_reached` | 仍需修订，但达到迭代上限 | 否 |
| `failed` | rubric 本身 malformed，或无法基于 transcript 评估 | 否 |
| `grader_error` | grader 运行异常，如 provider timeout、凭证缺失、结构化输出异常 | 否 |

状态理解：

```text
satisfied = 任务按 rubric 达标
needs_revision = 任务未达标，但还有机会改
max_iterations_reached = 任务可能未达标，系统停止自动重试
failed = rubric/评估条件本身有问题
grader_error = 评估基础设施有问题
```

注意：当达到 `max_iterations` 时，单次 callback 中 grader 的 verdict 可能仍是 `needs_revision`，而最终运行状态会记录为 `max_iterations_reached`。如果业务需要区分“还差一点但触顶”，应在 `invoke()` 完成后检查私有状态或记录最后一次 evaluation 与迭代计数。

---

## 观察评分迭代进度

有三种方式观察 rubric 评分：

| 方式 | 适合场景 |
|------|----------|
| `on_evaluation` callback | 简单日志、指标上报、写入数据集 |
| `stream.custom` | 实时 UI 显示评分开始/结束、每轮 verdict |
| LangSmith tracing | 调试完整运行链路、查看 grader sub-agent 细节 |

`on_evaluation` 示例：

```python
from deepagents.middleware.rubric import RubricEvaluation

def log_evaluation(ev: RubricEvaluation) -> None:
    print(f"iteration {ev['iteration']}: {ev['result']} - {ev['explanation']}")

agent = create_deep_agent(
    model="openai:gpt-5.5",
    middleware=[
        RubricMiddleware(
            model="anthropic:claude-haiku-4-5",
            on_evaluation=log_evaluation,
        )
    ],
    checkpointer=InMemorySaver(),
)
```

`RubricEvaluation` 结构：

| 字段 | 类型 | 说明 |
|------|------|------|
| `grading_run_id` | `str` | 同一次 rubric attempt 共享的 ID |
| `iteration` | `int` | 当前 grader pass 的 0-based 序号 |
| `result` | `str` | `satisfied`、`needs_revision`、`failed` 或 `grader_error` |
| `explanation` | `str` | grader 的摘要说明，异常时包含异常信息 |
| `criteria` | `list` | 每条 criterion 的通过/失败信息，失败项包含可执行反馈 |

评分事件：

| 事件 | 说明 |
|------|------|
| `rubric_evaluation_start` | grader 开始运行前发出 |
| `rubric_evaluation_end` | grader 返回 verdict 或异常后发出 |

Callback 注意事项：

| 注意点 | 说明 |
|--------|------|
| callback 异常会被记录并抑制 | 不要靠 raise 中断评分流程 |
| callback 不适合作控制流 | 它更适合观测、日志、指标 |
| `criteria` 是最有价值的数据 | 可用于 UI 展示“哪些要求还没满足” |

---

## 持久化 Rubric 与恢复执行

单次 `agent.invoke()` 或 `agent.stream_events()` 会把 rubric loop 跑到终态。

终态包括：

```text
satisfied
failed
max_iterations_reached
grader_error
```

如果希望后续调用延续同一个 rubric，需要：

1. 给 agent 配置 checkpointer。
2. 调用时传入相同 `thread_id`。
3. 后续 invocation 不传新 rubric 时，沿用该 thread 中持久化的 rubric。

示例：

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
    middleware=[rubric_middleware],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "code-generation-session"}}

agent.invoke(
    {
        "messages": [HumanMessage("Write the function.")],
        "rubric": "- All tests pass\n- Function signature is correct",
    },
    config=config,
)

agent.invoke(
    {"messages": [HumanMessage("Now optimize it without changing behavior.")]},
    config=config,
)
```

中断行为：

| 中断类型 | 行为 |
|----------|------|
| `KeyboardInterrupt` | 向外传播，不被 middleware 吞掉 |
| `asyncio.CancelledError` | 向外传播 |
| checkpointed thread 后续恢复 | 使用同一 rubric 时可恢复 in-flight grading run |

---

## 评分工具与证据收集

Rubric grader 可以只基于 transcript 推理，也可以调用工具收集证据。对代码、数据、格式这类可验证任务，建议提供工具。

代码生成示例中的模式：

```python
from langchain.tools import tool

@tool
def run_test_suite(code: str) -> dict:
    """Run tests against generated Python source code."""
    ...
    return {"ok": True, "failures": []}

rubric_middleware = RubricMiddleware(
    model="openai:gpt-5.5",
    system_prompt="You are a code reviewer grading generated code against a rubric.",
    tools=[run_test_suite],
    max_iterations=5,
)
```

工具化评分的价值：

| 价值 | 说明 |
|------|------|
| 降低主观误判 | 让 grader 用测试、schema、文件检查等证据判断 |
| 反馈更可执行 | 失败信息可直接来自工具结果 |
| 更适合自动修订 | agent 收到具体错误后更容易修复 |
| 更适合生产质量门禁 | 可把 “done” 变成可验证条件 |

常见 grader tools：

| 工具 | 用途 |
|------|------|
| `run_test_suite` | 验证代码正确性 |
| `validate_json_schema` | 检查 JSON 输出结构 |
| `count_words` / `count_tokens` | 检查长度限制 |
| `read_file` | 检查文件修改结果 |
| `lint` / `typecheck` | 检查代码质量 |
| `diff_check` | 确认只改了允许范围 |

Rubric 示例：

```text
- All tests pass in run_test_suite
- The function is named find_duplicates and accepts a single list argument
- Duplicate elements are returned in first-appearance order
- The implementation handles unhashable list elements
```

---

## 三者如何协同

Event streaming、Streaming、Grading rubrics 可以组成一个完整的可观测、可自修复 agent 体验。

```text
stream_events(version="v3")
  ├─ stream.messages
  │   └─ 展示 coordinator 输出
  ├─ stream.subagents
  │   ├─ subagent.messages
  │   ├─ subagent.tool_calls
  │   └─ subagent.output
  ├─ stream.tool_calls
  │   └─ 展示主 agent 工具调用
  └─ stream.custom
      └─ 展示 rubric_evaluation_start / rubric_evaluation_end
```

推荐 UI 结构：

```text
Main answer area
  ├─ coordinator token stream
  └─ final answer

Work panel
  ├─ subagent cards
  │   ├─ status
  │   ├─ latest message
  │   └─ tool call list
  └─ custom progress events

Quality panel
  ├─ rubric criteria
  ├─ current grading iteration
  ├─ verdict
  └─ failed criteria feedback
```

运行阶段：

| 阶段 | 可观察数据 |
|------|------------|
| coordinator planning | `stream.messages` 或 `messages` chunks |
| task delegation | `stream.subagents` 或 `updates` + namespace |
| subagent execution | `subagent.messages`、`subagent.tool_calls` |
| tool execution | `tool_calls` projection 或 `token.tool_call_chunks` |
| initial answer | final messages/output |
| rubric grading | `stream.custom` rubric events 或 `on_evaluation` |
| revision loop | 新一轮 messages/subagents/tool calls |
| terminal verdict | `satisfied`、`failed`、`max_iterations_reached`、`grader_error` |

---

## 典型使用场景

### 研究助手

```text
需求：
用户想看研究过程，而不是只等最终答案。

实现：
- coordinator 委派 researcher、fact_checker、summarizer
- UI 用 stream.subagents 展示每个 subagent 卡片
- subagent.tool_calls 展示搜索、读取、检查动作
- rubric 检查是否覆盖指定问题、是否给出来源、是否区分事实与推断
```

### 代码生成助手

```text
需求：
输出代码必须通过测试。

实现：
- agent 生成代码
- grader tool 调用 run_test_suite
- rubric 要求测试通过、签名正确、边界条件覆盖
- needs_revision 时把测试失败信息反馈给 agent
- stream.custom 展示每轮 grading verdict
```

### 数据分析助手

```text
需求：
用户需要看到分析进度，最终报告要包含固定结构。

实现：
- 工具内部用 get_stream_writer 输出 progress
- stream_mode="custom" 或 stream.custom 展示阶段进度
- rubric 检查报告包含结论、证据、限制、下一步建议
```

### 客服/运营工作流

```text
需求：
AI 回答必须遵守格式和合规要求。

实现：
- streaming 展示生成过程
- rubric 检查语气、禁用承诺、必要免责声明、下一步动作
- failed criteria 直接作为内部质量提示
```

---

## 最佳实践

### Streaming

1. 新项目优先使用 event streaming。
2. 用户可见 UI 优先使用 `stream.subagents`，少暴露 graph 内部节点名。
3. 调试 graph 执行时使用 `subgraphs=True` 和 namespace。
4. 同时展示 coordinator 和 subagent 输出时，使用并发消费或 `interleave()`。
5. 对 token、tool call、custom update 分区渲染，不要混在同一个文本流里。
6. 对长工具输出使用 delta 流式展示，完整结果放入可展开详情。
7. 给 subagent UI 使用稳定 key，例如 subagent path 或 tool call id。

### Rubrics

1. Rubric 要写成可判断的 checklist。
2. 每条 criterion 尽量只表达一个要求。
3. 可测试的要求交给 grader tools，而不是让 grader 主观判断。
4. `max_iterations` 不宜过大，避免成本失控；复杂任务可设置 3-5。
5. 用 `on_evaluation` 记录每轮结果，便于复盘和构造评估数据。
6. 对生产任务区分 `failed` 与 `grader_error`，前者多是 rubric 问题，后者多是基础设施问题。
7. 对用户可见结果，明确处理 `max_iterations_reached`，不要假装已完全通过。

### 组合使用

1. `RubricMiddleware` 负责质量闭环，streaming 负责解释闭环。
2. 把 rubric verdict 展示成质量状态，而不是普通聊天内容。
3. 当 `needs_revision` 出现时，UI 可以显示“正在根据评分反馈修订”。
4. 对多 subagent 工作流，可把 grader 视为最终 reviewer。
5. 使用 LangSmith tracing 记录完整运行链路，streaming 只承载实时体验。

---

## 故障排查

| 问题 | 可能原因 | 处理方式 |
|------|----------|----------|
| 看不到 subagent 事件 | `agent.stream()` 未设置 `subgraphs=True` | 打开 `subgraphs=True`，或改用 `stream_events().subagents` |
| UI 分不清 main 与 subagent | 未使用 namespace 或 subagent handle | event streaming 用 `subagent.name`，chunk 模式用 `chunk["ns"]` |
| token 顺序看起来乱 | coordinator 与 subagent 并发输出 | 按 source 分区展示，或消费 raw events 还原到达顺序 |
| 工具调用参数不完整 | tool call args 是分块流式输出 | 累积 `tool_call_chunks` 后再解析完整参数 |
| custom update 不出现 | 工具中未使用 `get_stream_writer()` 或 stream mode 不对 | 工具内部写 writer，调用时设置 `stream_mode="custom"` |
| rubric 没有运行 | invocation state 没传 `rubric` | 在 `agent.invoke()` 或 `stream_events()` 输入中加入 `rubric` |
| 一直 `needs_revision` | rubric 过严、反馈不够具体、agent 无法满足条件 | 拆分 criterion，提供 grader tools，降低或澄清要求 |
| 出现 `max_iterations_reached` | 达到迭代上限 | 检查最后一次 `criteria`，决定人工接管或提高上限 |
| 出现 `failed` | rubric 格式有问题或无法基于 transcript 评估 | 重写 rubric，让每条标准可判断 |
| 出现 `grader_error` | grader provider、凭证、结构化输出或网络异常 | 检查模型配置、API key、超时和日志 |
| callback 抛异常但流程继续 | `on_evaluation` 异常被记录并抑制 | 不要用 callback 控制流程，把它当观测钩子 |

---

## 快速参考

### API 选择

| 目标 | API |
|------|-----|
| 新项目实时 UI | `agent.stream_events(..., version="v3")` |
| 观察 delegated subagents | `stream.subagents` |
| 观察 coordinator 消息 | `stream.messages` |
| 观察 subagent 消息 | `subagent.messages` |
| 观察工具调用 | `stream.tool_calls` / `subagent.tool_calls` |
| 已有 LangGraph chunk 流 | `agent.stream(..., version="v2")` |
| 开启 subagent/subgraph chunk | `subgraphs=True` |
| 多 stream mode | `stream_mode=["updates", "messages", "custom"]` |
| Rubric 自检修订 | `RubricMiddleware` |
| Rubric 实时事件 | `stream_events(..., transformers=[CustomTransformer])` + `stream.custom` |

### Event streaming projections

| Projection | 说明 |
|------------|------|
| `messages` | 文本/消息输出 |
| `tool_calls` | 工具调用 |
| `subagents` | delegated task 对应的 subagent handles |
| `values` | 状态值 |
| `output` | 最终输出 |
| `custom` | 自定义事件 |

### Streaming chunk fields

| 字段 | 说明 |
|------|------|
| `chunk["type"]` | 当前 chunk 类型，如 `updates`、`messages`、`custom` |
| `chunk["ns"]` | namespace，空为 main agent，包含 `tools:` 通常代表 subagent |
| `chunk["data"]` | 当前 chunk payload |

### RubricMiddleware fields

| 参数 | 作用 |
|------|------|
| `model` | grader 模型 |
| `system_prompt` | grader 系统提示词 |
| `tools` | grader 可用工具 |
| `max_iterations` | 最大修订/评分迭代次数 |
| `on_evaluation` | 每轮评分后的 callback |

### Rubric verdicts

| Verdict | 含义 |
|---------|------|
| `satisfied` | 已满足 rubric |
| `needs_revision` | 需要修订，并会继续循环 |
| `max_iterations_reached` | 达到上限后停止 |
| `failed` | rubric 无法评估或格式不合适 |
| `grader_error` | grader 执行异常 |

---

## 资料来源

- Deep Agents Event streaming 官方文档：<https://docs.langchain.com/oss/python/deepagents/event-streaming>
- Deep Agents Streaming 官方文档：<https://docs.langchain.com/oss/python/deepagents/streaming>
- Deep Agents Grading rubrics 官方文档：<https://docs.langchain.com/oss/python/deepagents/rubric>
- LangChain Event Streaming 相关概念：<https://docs.langchain.com/oss/python/langchain/event-streaming>
- LangGraph Streaming 相关概念：<https://docs.langchain.com/oss/python/langgraph/streaming>
