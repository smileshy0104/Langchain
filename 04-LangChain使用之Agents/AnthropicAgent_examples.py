#!/usr/bin/env python3
"""
Anthropic 工具调用 Agent 示例（LangChain 0.3 语法）。

演示如何使用 Anthropic Claude 与 LangChain 的 create_tool_calling_agent
构建一个可以调用 Python 工具的对话式 Agent，并兼容类似
`agent.invoke({"messages": [...]})` 的调用方式。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, Tool, tool

# 从项目根目录加载 .env
dotenv.load_dotenv(dotenv_path="../.env")


def _require_env_var(name: str) -> str:
    """确保必需的环境变量存在。"""
    value = os.getenv(name)
    if not value or value.startswith("your-"):
        raise EnvironmentError(
            f"未检测到有效的 {name}，请在项目根目录的 .env 中配置后重试。"
        )
    return value


def _normalize_tools(
    tools: Sequence[Callable[..., Any] | BaseTool],
) -> list[BaseTool]:
    """将函数或 BaseTool 统一转换为工具实例。"""
    normalized: list[BaseTool] = []
    for item in tools:
        if isinstance(item, BaseTool):
            normalized.append(item)
        elif callable(item):
            normalized.append(Tool.from_function(item))
        else:
            raise TypeError(
                "tools 参数中的元素既不是可调用对象也不是 BaseTool 实例。"
            )
    return normalized


def _as_messages(payloads: Iterable[Mapping[str, Any]]) -> list[BaseMessage]:
    """将 {role, content} 形式的消息转换为 LangChain 所需的消息对象。"""
    role_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    messages: list[BaseMessage] = []
    for payload in payloads:
        role = payload.get("role")
        content = payload.get("content", "")
        if role not in role_map:
            raise ValueError(f"不支持的消息角色: {role!r}")
        messages.append(role_map[role](content=content))
    return messages


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气预报（示例函数）。"""
    return f"It's always sunny in {city}!"


def create_agent(
    *,
    model: str,
    tools: Sequence[Callable[..., Any] | BaseTool],
    system_prompt: str,
    temperature: float = 0.2,
) -> "RunnableAgent":
    """
    创建一个可调用 Python 工具的 Anthropic Agent。

    返回的对象提供 `.invoke({"messages": [...]})` 接口，以兼容
    LangChain v1 示例中的调用方式。
    """

    api_key = _require_env_var("ANTHROPIC_API_KEY")
    normalized_tools = _normalize_tools(tools)

    llm = ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    agent_runnable = create_tool_calling_agent(
        llm=llm,
        tools=normalized_tools,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=agent_runnable,
        tools=normalized_tools,
        verbose=True,
    )

    return RunnableAgent(executor=executor)


@dataclass
class RunnableAgent:
    """为 AgentExecutor 提供更易用的 `.invoke()` 封装。"""

    executor: AgentExecutor

    def invoke(self, inputs: MutableMapping[str, Any]) -> Mapping[str, Any]:
        if "messages" not in inputs:
            raise KeyError("调用参数需要包含 'messages' 键。")

        raw_messages = inputs["messages"]
        if isinstance(raw_messages, Iterable) and not isinstance(raw_messages, list):
            raw_messages = list(raw_messages)

        if isinstance(raw_messages, list) and raw_messages and isinstance(
            raw_messages[0], Mapping
        ):
            # 允许传入 [{"role": "...", "content": "..."}] 格式的消息
            inputs = dict(inputs)
            inputs["messages"] = _as_messages(raw_messages)  # type: ignore[assignment]

        return self.executor.invoke(inputs)


def run_demo() -> None:
    """运行与用户示例等效的演示。"""
    agent = create_agent(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
        temperature=0.0,
    )

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "what is the weather in sf"},
            ]
        }
    )

    print("\n📤 Agent 输出:")
    print(result.get("output"))


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
