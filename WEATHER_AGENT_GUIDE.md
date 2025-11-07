# 气象 Agent 代码详解与优化建议

## 1. 代码结构速览
- **系统提示词 (`SYSTEM_PROMPT`)**：为模型设定“气象专家且说话充满双关”的人设，并列出可用工具及调用策略。
- **上下文模式 (`Context` dataclass)**：自定义运行期上下文，当前只包含 `user_id`，方便工具函数读取用户信息。
- **工具 (`@tool`)**：
  - `get_weather_for_location`：根据城市名称返回天气信息（示例中固定返回晴天文案）。
  - `get_user_location`：从 `ToolRuntime` 的上下文读 `user_id`，返回预设位置。
- **模型 (`init_chat_model`)**：初始化 `claude-sonnet-4-5-20250929`，温度设为 0，确保确定性输出。
- **响应格式 (`ResponseFormat` dataclass)**：约束 Agent 的结构化回复字段，包括必填的 `punny_response` 以及可选的 `weather_conditions`。
- **记忆 (`InMemorySaver`)**：LangGraph 的内存检查点，用于保存对话状态。
- **Agent (`create_agent`)**：将模型、系统提示、工具、上下文 Schema、响应格式及记忆整合到一个可多轮调用的 Agent。
- **调用 (`agent.invoke`)**：使用 `configurable.thread_id` 区分会话，并通过 `Context(user_id="1")` 传入用户信息，演示连续两轮对话。

## 2. 运行流程拆解
1. **定义规则**：`SYSTEM_PROMPT` 指示 Agent 如何回答、可用工具以及使用顺序（先确定位置再查天气）。
2. **准备上下文**：`Context` dataclass 规定运行态会携带 `user_id`，LangGraph 在执行工具时会注入该上下文。
3. **注册工具**：通过 `@tool` 装饰器让 LangChain 知晓函数签名、描述与类型提示，执行时可自动注入 `ToolRuntime`。
4. **建模**：`init_chat_model` 返回可流式推理的聊天模型实例。
5. **定义输出 Schema**：`ResponseFormat` 让 Agent 在 `structured_response` 字段内返回命名字段，便于下游消费。
6. **挂载内存**：`InMemorySaver` 记录每次节点执行产生的 state，实现多轮续聊。
7. **创建 Agent**：`create_agent` 将上述组件拼装成 LangGraph 图，并默认包含工具路由与结构化输出节点。
8. **发起请求**：向 `agent.invoke` 传入消息列表、`config`（携带 `thread_id`）以及 `Context`。LangGraph 读取消息 → 判断是否缺失位置信息 → 调用 `get_user_location` → 再调用 `get_weather_for_location` → 依据响应格式生成最终回答。
9. **二次对话**：同一 `thread_id` 复用内存，不再重复询问位置，直接给出感谢回复。

## 3. 关键组件详解
### 3.1 工具设计
- **签名**：`get_weather_for_location(city: str)` 接受自然语言城市名；`get_user_location(runtime: ToolRuntime[Context])` 则依赖 LangGraph 自动注入的 `runtime`。
- **上下文注入**：`ToolRuntime[Context]` 允许在工具内部访问 `runtime.context`，从而读取 `user_id`、会话元数据等。
- **容错**：示例中为了简洁未实现异常处理，真实场景需捕获 HTTP/网络错误，并返回结构化告警信息。

### 3.2 模型与提示
- 将“工具使用策略”直接写入系统提示，可确保模型在需要位置信息时触发 `get_user_location`。
- 温度设为 0 让双关输出稳定；若想更活泼可调高温度或加入 `top_p` 控制。

### 3.3 响应格式
- `ResponseFormat` 让 Agent 在 `response['structured_response']` 中返回 dataclass，可通过 `response['output']` 查看原始文本。
- 当 `weather_conditions` 不适用时返回 `None`，避免空字符串占位。

### 3.4 记忆与线程
- `configurable.thread_id` 是 LangGraph 的记忆隔离键；同一线程自动延续上下文，不同值即可实现多用户并发。
- `InMemorySaver` 适合 demo 或单进程部署，生产环境可换成数据库/Redis 等持久化实现。

## 4. 可行的优化方向
1. **真实天气数据**
   - 将 `get_weather_for_location` 接入真实天气 API（如 OpenWeather、和风天气），并根据用户时区/坐标返回更具体的天气、温度与提醒。
   - 对 API 调用结果做缓存（按城市+时间片）以减少请求量。
2. **位置推断增强**
   - 从用户最近一次对话或 profile 获取城市，必要时再调用 `get_user_location`；或允许用户主动覆盖位置。
   - 增加地理解析工具，将自然语言描述（如 “in the bay area”）转换为标准城市。
3. **配置解耦**
   - 将模型名称、温度、API Key、默认城市等放入 `.env`/配置文件，便于不同环境切换。
   - 在 `Context` 中加入 `preferred_unit`、`language` 等字段，支持个性化响应。
4. **可观测性与日志**
   - 在工具内部增加结构化日志，记录请求参数、耗时、调用结果或错误。
   - 将 `thread_id`、`user_id` 与日志 trace 关联，方便追踪多轮对话问题。
5. **错误与降级策略**
   - 捕获天气 API 超时或异常时，返回“临时无法获取实时天气”的安全提示，并允许 fallback 到历史数据。
   - 根据工具失败次数决定是否继续调用，避免无限循环。
6. **输出质量控制**
   - 在最终响应前增加一个“验证节点”，检查 `punny_response` 是否包含双关元素、是否提及天气信息。
   - 可添加 `safety` 或 `style` 过滤器，确保回答符合品牌语气。
7. **测试与自动化**
   - 利用 LangChain 的 `ToolInvocation` mock，编写单元测试验证：当用户未提供位置时会调用 `get_user_location`；已有位置时只调用一次天气工具。
   - 对 `ResponseFormat` 序列化/反序列化做测试，保证字段变更后兼容。

## 5. 示例扩展思路
- **多城市比较**：添加 `compare_weather(cities: list[str])` 工具，帮助用户比较差异。
- **订阅提醒**：结合调度器（如 APScheduler）按 `Context` 中的 `user_id` 推送每日天气。
- **多语言支持**: 在 `Context` 或 `messages` metadata 中携带语言偏好，提示词里声明“根据用户语言回答”。
- **安全策略**：为工具增加权限控制，例如只有在 `Context` 中标记 `allow_location_lookup=True` 时才调用 `get_user_location`。

## 6. 调试与部署建议
- 使用 `agent.stream(...)` 观察 LangGraph 节点执行顺序，有助于诊断工具调用逻辑。
- 在本地开发阶段，可将模型换成开源或较小的 LLM 以节省成本，再切换到 Claude 进行最终验证。
- 若部署到无状态环境，记得将 `InMemorySaver` 替换为集中式存储，否则多实例间无法共享记忆。

通过以上拆解与优化方向，可以把示例中的“玩具天气 Agent”逐步升级为可在真实场景中运行、可观测、可维护的智能助手。
