# LangGraph Fault Tolerance 详细指南

> 基于 LangGraph 官方 Fault tolerance 文档整理。本文聚焦 LangGraph 如何为节点配置 retries、timeouts、error handlers、graph defaults、Functional API 容错参数，以及如何用 graceful shutdown 在 superstep 边界安全停止并恢复运行。

## 目录

1. [整体理解](#整体理解)
2. [Fault Tolerance 是什么](#fault-tolerance-是什么)
3. [失败处理顺序](#失败处理顺序)
4. [Retries](#retries)
5. [RetryPolicy 默认行为](#retrypolicy-默认行为)
6. [RetryPolicy 参数](#retrypolicy-参数)
7. [Custom Retry Logic](#custom-retry-logic)
8. [Inspect Retry State](#inspect-retry-state)
9. [Timeouts](#timeouts)
10. [Run Timeout](#run-timeout)
11. [Idle Timeout](#idle-timeout)
12. [Progress Signals 与 Heartbeat](#progress-signals-与-heartbeat)
13. [NodeTimeoutError](#nodetimeouterror)
14. [Dynamic Timeouts with Send](#dynamic-timeouts-with-send)
15. [Error Handling](#error-handling)
16. [NodeError](#nodeerror)
17. [Route with Command](#route-with-command)
18. [Resume-safe Failures](#resume-safe-failures)
19. [Interrupt 与 Subgraph Failure](#interrupt-与-subgraph-failure)
20. [Graph Defaults](#graph-defaults)
21. [Precedence 与 Applicability](#precedence-与-applicability)
22. [Functional API](#functional-api)
23. [Graceful Shutdown](#graceful-shutdown)
24. [Limitations](#limitations)
25. [最佳实践](#最佳实践)
26. [故障排查](#故障排查)
27. [快速参考](#快速参考)
28. [资料来源](#资料来源)

---

## 整体理解

LangGraph Fault tolerance 解决的是：

```text
节点慢、失败、超时、外部服务异常、进程要停机时，
如何让 graph 尽量可恢复、可补偿、可继续。
```

LangGraph 提供三类可组合机制：

| 机制 | 作用 |
|------|------|
| Retries | 节点失败后按异常类型和 backoff 自动重试 |
| Timeouts | 限制单次节点 attempt 的运行时间或空闲时间 |
| Error handling | 重试耗尽后执行恢复/补偿逻辑 |

以及一个运行级机制：

| 机制 | 作用 |
|------|------|
| Graceful shutdown | 在 superstep 边界安全停止并保存可恢复 checkpoint |

核心流程：

```text
Attempt starts
  -> run node
      -> success: graph 继续
      -> exception / NodeTimeoutError:
           retry_policy 判断是否重试
             -> yes: 重新执行 node attempt
             -> no / exhausted:
                  error_handler 是否存在
                    -> yes: 运行 handler，更新 state 或 Command goto
                    -> no: exception bubble up
```

---

## Fault Tolerance 是什么

Fault tolerance 指 LangGraph 对节点失败的系统性处理能力。

典型失败来源：

| 失败来源 | 示例 |
|----------|------|
| 慢外部 API | HTTP 请求卡住 |
| 临时网络错误 | connection reset、timeout |
| rate limit | provider 返回 429 |
| LLM/tool 异常 | provider 5xx、工具内部错误 |
| 业务不可恢复错误 | 参数错误、权限错误 |
| 节点代码异常 | 未处理 exception |
| 进程关闭 | SIGTERM、容器回收 |

设计目标：

1. 对短暂故障自动重试。
2. 对卡住节点及时超时。
3. 对最终失败走补偿分支，而不是整图崩溃。
4. 对停机请求在安全边界保存 checkpoint。
5. 与 checkpointer 配合，恢复后不重复已成功 superstep。

注意：

```text
Per-node timeouts 和 node-level error handlers 需要 langgraph>=1.2。
```

---

## 失败处理顺序

官方明确了组合顺序：

```text
timeout / exception
  -> retry policy
  -> error handler
  -> bubble up
```

具体：

| 阶段 | 说明 |
|------|------|
| Node attempt | 节点开始执行 |
| Timeout | 如果超时，抛出 `NodeTimeoutError` |
| Exception | 节点任意异常都进入 retry 判断 |
| Retry | `retry_policy` 决定是否还有 attempt |
| Error handler | 重试耗尽后执行恢复函数 |
| Bubble up | 没 handler 或 handler 也失败时异常向外抛 |

关键点：

```text
error_handler 不是每次失败都执行；
它只在 retry policy 耗尽后执行。
```

如果没有 retry policy：

```text
节点第一次失败后直接进入 error_handler；
如果没有 handler，则异常抛出。
```

---

## Retries

给节点配置 retry：

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "call_api",
    call_api,
    retry_policy=RetryPolicy(max_attempts=3),
)
```

适合 retry 的错误：

| 错误 | 原因 |
|------|------|
| 网络抖动 | 再试可能成功 |
| 外部服务 5xx | 服务端临时异常 |
| provider timeout | 可能是临时拥塞 |
| `NodeTimeoutError` | 默认可重试 |
| 临时 rate limit | 可结合 backoff |

不适合 retry 的错误：

| 错误 | 原因 |
|------|------|
| 参数类型错误 | 重试不会改变输入 |
| 语法错误 | 代码问题 |
| 权限错误 | 需要配置或人工处理 |
| 业务规则失败 | 应走业务分支或 handler |
| 不幂等副作用 | 可能重复扣款/发邮件 |

---

## RetryPolicy 默认行为

默认 `retry_on` 使用 `default_retry_on`。

默认会重试大多数异常，但不会重试以下异常及其子类：

| 默认不重试 |
|------------|
| `ValueError` |
| `TypeError` |
| `ArithmeticError` |
| `ImportError` |
| `LookupError` |
| `NameError` |
| `SyntaxError` |
| `RuntimeError` |
| `ReferenceError` |
| `StopIteration` |
| `StopAsyncIteration` |
| `OSError` |

对于常见 HTTP 库：

```text
requests / httpx 等通常只对 5xx status code 重试。
```

`NodeTimeoutError`：

```text
默认 retryable。
```

---

## RetryPolicy 参数

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_attempts` | `int` | `3` | 最大尝试次数，包含第一次 |
| `initial_interval` | `float` | `0.5` | 第一次 retry 前等待秒数 |
| `backoff_factor` | `float` | `2.0` | 每次 retry 后等待时间乘数 |
| `max_interval` | `float` | `128.0` | retry 间隔最大秒数 |
| `jitter` | `bool` | `True` | 是否添加随机抖动 |
| `retry_on` | exception type / sequence / callable | `default_retry_on` | 判断哪些异常可重试 |

Backoff 示例：

```text
initial_interval=0.5
backoff_factor=2

retry waits:
0.5s -> 1s -> 2s -> 4s ...
```

`jitter=True` 的价值：

```text
避免大量节点或请求在同一时间重试，减少 thundering herd。
```

---

## Custom Retry Logic

可以传 exception type、exception tuple/list，或 callable。

扩展默认逻辑：

```python
from langgraph.types import RetryPolicy, default_retry_on

def custom_retry_on(exc: BaseException) -> bool:
    if isinstance(exc, MyCustomError):
        return False
    return default_retry_on(exc)

builder.add_node(
    "call_api",
    call_api,
    retry_policy=RetryPolicy(max_attempts=3, retry_on=custom_retry_on),
)
```

常见写法：

```python
RetryPolicy(
    max_attempts=5,
    initial_interval=1.0,
    backoff_factor=2.0,
    retry_on=(TimeoutError, ConnectionError),
)
```

建议：

| 场景 | retry_on |
|------|----------|
| 外部 API | 网络错误、5xx、timeout |
| LLM provider | provider timeout、rate limit、5xx |
| 业务校验失败 | 不重试 |
| 权限/认证错误 | 不重试 |

---

## Inspect Retry State

节点内部可以通过 `runtime.execution_info` 查看当前 attempt。

示例：第一次用主 API，重试时切换 fallback API。

```python
from langgraph.runtime import Runtime

def my_node(state: State, runtime: Runtime) -> State:
    if runtime.execution_info.node_attempt > 1:
        return {"result": call_fallback_api()}
    return {"result": call_primary_api()}
```

`execution_info` 字段：

| Attribute | Type | Description |
|-----------|------|-------------|
| `node_attempt` | `int` | 当前 attempt，1 表示第一次 |
| `node_first_attempt_time` | `float \| None` | 第一次 attempt 的 Unix timestamp |
| `thread_id` | `str \| None` | 当前 thread ID，无 checkpointer 时可能为 None |
| `run_id` | `str \| None` | 当前 run ID |
| `checkpoint_id` | `str` | 当前 checkpoint ID |
| `task_id` | `str` | 当前 task ID |

即使没有 retry policy，`execution_info` 也可用，`node_attempt` 默认为 `1`。

---

## Timeouts

`timeout=` 限制单次 node attempt 的运行时间。

可传：

| 类型 | 示例 |
|------|------|
| 数字秒数 | `timeout=60` |
| `timedelta` | `timeout=timedelta(minutes=2)` |
| `TimeoutPolicy` | `TimeoutPolicy(run_timeout=120, idle_timeout=30)` |

示例：

```python
from datetime import timedelta
from langgraph.types import TimeoutPolicy

builder.add_node("call_model", call_model, timeout=60)
builder.add_node("call_model", call_model, timeout=timedelta(minutes=2))

builder.add_node(
    "call_model",
    call_model,
    timeout=TimeoutPolicy(run_timeout=120, idle_timeout=30),
)
```

重要限制：

```text
Node timeouts only apply to async nodes.
sync nodes with timeout are rejected at compile time.
```

如果是阻塞 I/O：

```python
import asyncio

async def node(state):
    result = await asyncio.to_thread(blocking_call)
    return {"result": result}
```

---

## Run Timeout

`run_timeout` 是硬性 wall-clock cap。

特点：

| 特点 | 说明 |
|------|------|
| 不会刷新 | 无论节点是否有进展 |
| 限制单次 attempt | retry 后重新计时 |
| 超时抛 `NodeTimeoutError` | 交给 retry policy 判断 |

示例：

```python
from langgraph.types import TimeoutPolicy

builder.add_node(
    "call_model",
    call_model,
    timeout=TimeoutPolicy(run_timeout=120),
)
```

适合：

| 场景 | 说明 |
|------|------|
| 必须限制总耗时 | LLM/API 单次调用不可超过 N 秒 |
| 服务 SLA | 节点不能无限占用资源 |
| 防止死循环 | async 节点长时间不返回 |

---

## Idle Timeout

`idle_timeout` 是“无进展超时”。

特点：

| 特点 | 说明 |
|------|------|
| 进展会刷新计时 | 有 progress signal 时重置 |
| 适合长任务 | 只要持续有进展就不超时 |
| 可和 run_timeout 同时设置 | 谁先触发就取消 attempt |

示例：

```python
builder.add_node(
    "call_model",
    call_model,
    timeout=TimeoutPolicy(idle_timeout=30),
)
```

适合：

| 场景 | 说明 |
|------|------|
| 流式 LLM | 只要持续输出 token 就不 idle |
| 批处理 | 每批处理后 heartbeat |
| 长查询 | 有进度事件则继续 |
| 下载/上传 | 有数据块就刷新 |

---

## Progress Signals 与 Heartbeat

默认 `refresh_on="auto"` 时，以下信号会刷新 idle clock：

| Progress signal | 说明 |
|-----------------|------|
| `CONFIG_KEY_SEND` state writes | 节点写 state |
| async stream chunks | 节点 yield 流式 chunk |
| child-task scheduling | 调度子任务 |
| runtime stream-writer calls | runtime writer 输出 |
| LangChain callback events | LLM token、tool call、chain start/end 等 |

### Heartbeat mode

如果只想让显式 heartbeat 刷新 idle clock：

```python
builder.add_node(
    "call_model",
    call_model,
    timeout=TimeoutPolicy(idle_timeout=30, refresh_on="heartbeat"),
)
```

适合：

```text
子调用非常 chatty，但你不想让它们的普通 callback 重置 idle 判断。
```

### Manual heartbeats

长任务没有自然 progress signal 时，主动调用：

```python
from langgraph.runtime import Runtime

async def long_running_node(state: State, runtime: Runtime) -> State:
    for batch in fetch_batches():
        process(batch)
        runtime.heartbeat()
    return {"result": "done"}
```

`runtime.heartbeat()`：

```text
在没有 idle timeout 的场景中是 no-op；
可以无条件调用。
```

---

## NodeTimeoutError

超时时 LangGraph 抛出 `NodeTimeoutError`。

字段：

| Attribute | Type | Description |
|-----------|------|-------------|
| `node` | `str` | 超时节点名 |
| `elapsed` | `float` | 超时前经过秒数 |
| `kind` | `"idle"` or `"run"` | 触发的是 idle 还是 run timeout |
| `idle_timeout` | `float \| None` | 配置的 idle timeout |
| `run_timeout` | `float \| None` | 配置的 run timeout |

行为：

1. 抛出 `NodeTimeoutError`。
2. 清除该失败 attempt 的 writes。
3. 交给 retry policy 判断是否重试。
4. retry 后 timeout clock 重新开始。

组合示例：

```python
from langgraph.types import RetryPolicy, TimeoutPolicy

builder.add_node(
    "call_model",
    call_model,
    timeout=TimeoutPolicy(idle_timeout=30),
    retry_policy=RetryPolicy(max_attempts=3),
)
```

---

## Dynamic Timeouts with Send

使用 `Send` 动态分发节点时，可以给每次 push 单独设置 timeout。

```python
from langgraph.types import Send, TimeoutPolicy

def fan_out(state: OverallState):
    return [
        Send("process_item", {"item": item}, timeout=TimeoutPolicy(idle_timeout=15))
        for item in state["items"]
    ]
```

规则：

| 情况 | 使用的 timeout |
|------|----------------|
| `Send(..., timeout=...)` | 使用 Send 上的 timeout |
| Send 没有 timeout | 使用目标节点 `add_node(..., timeout=...)` |

适合：

| 场景 | 说明 |
|------|------|
| map-reduce | 每个 item 有不同超时预算 |
| 优先级任务 | 高优先任务给更严格 timeout |
| 大小不同的 item | 根据 item 复杂度动态设置 |

---

## Error Handling

Error handler 在节点失败且所有 retries 耗尽后执行。

配置：

```python
from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy

def payment_error_handler(state: State, error: NodeError) -> Command:
    return Command(
        update={"status": f"compensated: {error.error}"},
        goto="finalize",
    )

builder.add_node(
    "charge_payment",
    charge_payment,
    retry_policy=RetryPolicy(max_attempts=3, retry_on=ConnectionError),
    error_handler=payment_error_handler,
)
```

适合：

| 场景 | 说明 |
|------|------|
| Saga compensation | 支付失败后释放库存 |
| fallback path | 主流程失败后进入备用节点 |
| graceful degradation | 写入降级状态并继续 |
| alerting | 记录错误后路由到人工处理 |

不适合：

```text
把所有错误都吞掉。
未知错误如果不能安全恢复，应让它 bubble up。
```

---

## NodeError

Error handler 可通过类型注解接收 `NodeError`：

```python
from langgraph.errors import NodeError

def my_handler(state: State, error: NodeError) -> Command:
    print(f"Node {error.node} failed with: {error.error}")
    return Command(update={"status": "recovered"}, goto="next_step")
```

字段：

| Attribute | Type | Description |
|-----------|------|-------------|
| `node` | `str` | 失败节点名 |
| `error` | `BaseException` | 节点抛出的原始异常 |

`NodeError` 是 frozen dataclass。

handler 签名是灵活的：

```python
def handler(state): ...
def handler(state, runtime): ...
def handler(state, error: NodeError): ...
def handler(state, error: NodeError, config: RunnableConfig): ...
```

---

## Route with Command

Error handler 可以返回 `Command` 同时更新 state 并指定下一跳。

示例：

```python
def payment_error_handler(state: State, error: NodeError) -> Command:
    return Command(
        update={"status": f"compensated_after_{error.node}: {error.error}"},
        goto="finalize",
    )
```

这实现了 Saga / compensation pattern：

```text
reserve_inventory
  -> charge_payment
       -> success: continue
       -> retry exhausted:
            payment_error_handler
              -> update status
              -> goto finalize
```

关键点：

```text
retry policy 决定“什么时候重试”；
error_handler 决定“最终失败后怎么补偿”。
二者解耦。
```

---

## Resume-safe Failures

官方强调：

```text
Failure provenance is checkpointed.
```

也就是说：

```text
如果节点失败后、handler 完成前，graph 被 interrupt 或进程崩溃；
恢复时 handler 会看到同一个 NodeError context。
```

这对补偿流程很重要：

| 场景 | 价值 |
|------|------|
| 支付失败后补偿 | 恢复后仍知道失败节点和错误 |
| 人工处理错误 | 可以展示原始失败原因 |
| 崩溃恢复 | 不丢失失败上下文 |

---

## Interrupt 与 Subgraph Failure

### `interrupt()` 行为

`interrupt()` 不会进入 error handler。

原因：

```text
interrupt 使用 GraphBubbleUp 机制暂停 graph；
它绕过 retry policy 和 error handler。
```

所以：

| 事件 | 是否触发 retry/error_handler |
|------|-----------------------------|
| 普通 exception | 是 |
| `NodeTimeoutError` | 是 |
| `interrupt()` | 否 |

### Subgraph failures

如果某个节点包装了 subgraph，而 subgraph 抛出未处理异常：

```text
异常会冒泡到 parent node。
如果 parent node 有 error_handler，则 handler 收到该异常。
```

---

## Graph Defaults

如果多个节点共享同样的容错配置，可以用 `set_node_defaults()`。

```python
from langgraph.errors import NodeError
from langgraph.types import RetryPolicy, TimeoutPolicy

def default_error_handler(state: State, error: NodeError) -> State:
    return {"status": f"handled: {error.error}"}

graph = (
    StateGraph(State)
    .set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=3),
        error_handler=default_error_handler,
        timeout=TimeoutPolicy(run_timeout=30),
    )
    .add_node("step_a", step_a)
    .add_node("step_b", step_b)
    .add_edge(START, "step_a")
    .compile()
)
```

可配置：

| 默认项 | 说明 |
|--------|------|
| `retry_policy` | 默认 retry 策略 |
| `error_handler` | 默认错误恢复函数 |
| `timeout` | 默认超时策略 |
| `cache_policy` | 默认缓存策略 |

---

## Precedence 与 Applicability

### Precedence

节点级配置覆盖 graph defaults：

```python
graph = (
    StateGraph(State)
    .set_node_defaults(error_handler=default_error_handler)
    .add_node("step_a", step_a)
    .add_node("step_b", step_b, error_handler=custom_error_handler)
    .compile()
)
```

结果：

| 节点 | 使用 handler |
|------|--------------|
| `step_a` | `default_error_handler` |
| `step_b` | `custom_error_handler` |

Defaults 在 `compile()` 时解析，所以 `set_node_defaults()` 可以在 `add_node()` 前或后调用。

### Applicability matrix

| `set_node_defaults` parameter | Regular nodes | Error-handler nodes | 原因 |
|-------------------------------|---------------|---------------------|------|
| `retry_policy` | 适用 | 适用 | handler 的临时失败也可重试 |
| `timeout` | 适用 | 适用 | handler 卡住也要取消 |
| `error_handler` | 适用 | 不适用 | handler 不能 catch 自己 |
| `cache_policy` | 适用 | 不适用 | 缓存 handler 结果不安全 |

### Scope

```text
Parent graph 的 defaults 不会被 subgraphs 继承。
每个 graph 维护自己的 defaults。
```

---

## Functional API

Functional API 的 `@task` 和 `@entrypoint` 也支持 `timeout` 与 `retry_policy`。

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy, TimeoutPolicy

@task(
    timeout=TimeoutPolicy(idle_timeout=30),
    retry_policy=RetryPolicy(max_attempts=3),
)
async def call_api(url: str) -> str:
    response = await fetch(url)
    return response.text

@entrypoint(timeout=60)
async def my_workflow(inputs: dict) -> str:
    result = await call_api("https://api.example.com/data")
    return result
```

行为与 `add_node` 一致：

| 情况 | 行为 |
|------|------|
| timeout | 抛 `NodeTimeoutError` |
| timed-out attempt | buffered writes 被清除 |
| retry policy matches | 重试 |
| retries exhausted | 异常继续传播或进入 handler |

---

## Graceful Shutdown

Graceful shutdown 用于合作式停机：

```text
让正在运行的 graph 在当前 superstep 完成后停止，
保存可恢复 checkpoint，
之后可用同一 config 恢复。
```

适合：

| 场景 | 说明 |
|------|------|
| SIGTERM | 容器/进程即将被回收 |
| supervisor reclaim | 外部调度器回收资源 |
| 滚动部署 | 不中断已完成 superstep |
| 长任务迁移 | 保存 checkpoint 后稍后继续 |

使用：

```python
from langgraph.runtime import RunControl
from langgraph.errors import GraphDrained

control = RunControl()

# In a signal handler or supervisor:
# control.request_drain("sigterm")

try:
    result = graph.invoke(inputs, config, control=control)
except GraphDrained as e:
    print(f"Drained: {e.reason}")
```

### Semantics

| Scenario | Behavior |
|----------|----------|
| Node mid-execution | 当前节点继续完成，下一 superstep 才 drain |
| Node 正在 retry | retry loop 直到成功或耗尽后 drain |
| 同 tick graph 自然结束 | 正常返回，可看 `control.drain_requested` |
| 还有更多 supersteps | 抛 `GraphDrained(reason)`，checkpoint 已保存 |
| Subgraph request drain | `GraphDrained` 冒泡到 parent，在 parent superstep 边界停止 |

### Resume after drain

使用同一 `thread_id`：

```python
result = graph.invoke(None, config)
```

### Node 内读取 drain 状态

```python
from langgraph.runtime import Runtime

async def my_node(state: State, runtime: Runtime) -> State:
    if runtime.drain_requested:
        return {"status": "skipped", "reason": runtime.drain_reason}
    return {"status": await do_work()}
```

### SIGTERM hook pattern

```python
import signal
from langgraph.runtime import RunControl
from langgraph.errors import GraphDrained

control = RunControl()
signal.signal(signal.SIGTERM, lambda *_: control.request_drain("sigterm"))

try:
    result = graph.invoke(inputs, config, control=control)
except GraphDrained as e:
    log.info("graph drained: %s", e.reason)
```

注意：

```text
request_drain() 不会取消正在运行的 asyncio tasks，也不会 kill threads。
如果需要硬上限，应结合 graceful timeout 和 task cancellation。
```

---

## Limitations

| 限制 | 说明 |
|------|------|
| Timeouts are async-only | sync nodes 配 timeout 会在 compile 时报错 |
| One handler per node | 每个节点最多一个 `error_handler` |
| Handler failures bubble up | handler 自己抛错时，异常会向外传播 |
| Defaults not inherited by subgraphs | parent graph 的 `set_node_defaults` 不影响 subgraph |

---

## 最佳实践

1. 对所有外部 API 节点配置 retry policy。
2. 对可能卡住的 async 节点配置 timeout。
3. 区分 `run_timeout` 和 `idle_timeout`：前者限制总时长，后者限制无进展时间。
4. 长任务没有自然进度时，用 `runtime.heartbeat()`。
5. Retry 只用于 transient failures，不要重试确定性业务错误。
6. 对有副作用节点，先设计幂等性，再配置 retry。
7. 用 error handler 实现补偿分支，不要简单吞异常。
8. 默认策略用 `set_node_defaults()`，关键节点单独覆盖。
9. HITL interrupt 不会走 error handler，应单独设计。
10. 生产停机用 `RunControl.request_drain()`，不要直接杀进程。
11. 对需要恢复的 graph 配 checkpointer，否则 drain/resume 价值有限。
12. 对 subgraph 单独设置 defaults，不要假设继承 parent。

---

## 故障排查

| 问题 | 可能原因 | 处理方式 |
|------|----------|----------|
| timeout 配了但 compile 报错 | 节点是 sync function | 改 async，或用 `asyncio.to_thread` 包 blocking I/O |
| 节点超时后没有重试 | retry policy 不匹配或 max_attempts 到了 | 检查 `retry_on` 和 `max_attempts` |
| error handler 没执行 | retry 还没耗尽，或是 `interrupt()` | 等 retry 耗尽；interrupt 需单独处理 |
| handler 自己失败导致 graph 崩 | handler exception 会 bubble up | 给 handler 做简单可靠逻辑，必要时给 defaults timeout/retry |
| retry 导致重复扣款/发邮件 | 副作用节点不幂等 | 设计 idempotency key，或限制 retry |
| idle timeout 被频繁刷新 | `refresh_on="auto"` 被 callback 重置 | 改 `refresh_on="heartbeat"` |
| drain 后无法恢复 | 没有 checkpoint/thread_id | 配 checkpointer，用同一 config 调 `invoke(None, config)` |
| subgraph 没有默认 retry/timeout | defaults 不继承 | 在 subgraph 自己调用 `set_node_defaults()` |
| Functional API timeout 行为不符合预期 | 任务不是 async 或误以为 entrypoint 覆盖 task | 分别给 `@task` / `@entrypoint` 配置 |

---

## 快速参考

### 失败处理顺序

```text
node attempt
  -> exception / NodeTimeoutError
  -> retry_policy
  -> error_handler
  -> bubble up
```

### RetryPolicy

| 参数 | 默认 | 说明 |
|------|------|------|
| `max_attempts` | `3` | 总尝试次数 |
| `initial_interval` | `0.5` | 首次重试等待 |
| `backoff_factor` | `2.0` | backoff 倍数 |
| `max_interval` | `128.0` | 最大等待 |
| `jitter` | `True` | 随机抖动 |
| `retry_on` | `default_retry_on` | 判断可重试异常 |

### TimeoutPolicy

| 参数 | 说明 |
|------|------|
| `run_timeout` | 单次 attempt 总时长上限 |
| `idle_timeout` | 无进展时长上限 |
| `refresh_on="auto"` | callback/stream/write 等自动刷新 idle |
| `refresh_on="heartbeat"` | 仅 `runtime.heartbeat()` 刷新 idle |

### Error Handler

```python
def handler(state: State, error: NodeError) -> Command:
    return Command(update={"status": "recovered"}, goto="next_step")
```

### Graph Defaults

```python
graph = (
    StateGraph(State)
    .set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=3),
        timeout=TimeoutPolicy(run_timeout=30),
        error_handler=default_handler,
    )
)
```

### Graceful Shutdown

```python
control = RunControl()
control.request_drain("sigterm")

try:
    graph.invoke(inputs, config, control=control)
except GraphDrained:
    graph.invoke(None, config)
```

---

## 资料来源

- LangGraph Fault tolerance 官方文档：<https://docs.langchain.com/oss/python/langgraph/fault-tolerance>
