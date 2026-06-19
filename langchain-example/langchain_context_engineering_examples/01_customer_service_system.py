"""
案例 1：智能客服系统。

这个文件基本沿用 `LangChain_Context_Engineering_详细指南.md` 中的示例代码，只做两处运行化调整：
1. 模型名从环境变量读取，避免把示例里的 gpt-5.4-mini / gpt-5.5 写死。
2. 增加一个最小 CLI，方便直接运行并传入 CustomerContext。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field


def load_env_files() -> None:
    """加载当前示例目录或 langchain notebook 目录中的 .env。"""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    current_dir = Path(__file__).parent
    for env_path in (current_dir / ".env", current_dir.parent / "langchain" / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)


load_env_files()


MODEL_PROVIDER = os.getenv("LANGCHAIN_MODEL_PROVIDER", "zhipuai").lower()
DEFAULT_MODEL_NAME = os.getenv("LANGCHAIN_DEFAULT_MODEL", "glm-4-flash")
ENTERPRISE_MODEL_NAME = os.getenv("LANGCHAIN_ENTERPRISE_MODEL", DEFAULT_MODEL_NAME)
DEFAULT_TEMPERATURE = float(os.getenv("LANGCHAIN_DEFAULT_TEMPERATURE", "0.7"))
ENTERPRISE_TEMPERATURE = float(os.getenv("LANGCHAIN_ENTERPRISE_TEMPERATURE", "0.2"))


def require_env(key: str) -> str:
    """读取必需环境变量，避免自定义模型配置缺失时静默失败。"""

    value = os.getenv(key, "").strip()
    if not value or "your-" in value.lower():
        raise ValueError(f"请先配置环境变量 {key}")
    return value


def get_custom_base_url() -> str:
    """OpenAI-compatible 服务通常要求 base_url 指向 /v1。"""

    base_url = require_env("CUSTOM_MODEL_BASE_URL").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def create_configured_model(model_name: str, temperature: float) -> Any:
    """根据环境变量创建模型。

    - LANGCHAIN_MODEL_PROVIDER=custom 时，使用 OpenAI-compatible 接口。
    - 其他 provider 交给 LangChain 的 init_chat_model 处理，例如 zhipuai / openai。
    """

    if MODEL_PROVIDER == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=require_env("CUSTOM_MODEL_API_KEY"),
            base_url=get_custom_base_url(),
        )

    return init_chat_model(
        model_name,
        model_provider=MODEL_PROVIDER,
        temperature=temperature,
    )


@dataclass
class CustomerContext:
    """Runtime Context：调用 Agent 时注入的客户身份和服务等级。"""

    customer_id: str
    customer_name: str
    subscription_tier: str  # free / pro / enterprise
    language: str = "zh"


@tool
def get_customer_orders(limit: int, runtime: ToolRuntime[CustomerContext]) -> str:
    """Get recent orders for the current customer."""

    # Tool Context：工具不用让模型传 customer_id，而是从 runtime.context 读取。
    customer_id = runtime.context.customer_id
    return f"客户 {customer_id} 最近 {limit} 个订单：[...]"


@tool
def create_support_ticket(
    issue: str,
    priority: str,
    runtime: ToolRuntime[CustomerContext],
) -> str:
    """Create a customer support ticket."""

    # Tool Context：工具可以根据客户套餐执行业务规则。
    ctx = runtime.context
    if ctx.subscription_tier == "enterprise" and priority == "normal":
        priority = "high"
    return f"已为 {ctx.customer_name} 创建工单，优先级：{priority}，问题：{issue}"


@dynamic_prompt
def customer_service_prompt(request: ModelRequest) -> str:
    # Model Context：每次模型调用前，动态生成客户专属系统提示词。
    ctx = request.runtime.context
    tier_guide = {
        "free": "提供基础支持，复杂问题建议升级套餐。",
        "pro": "提供专业技术支持。",
        "enterprise": "提供最高级别支持，优先处理，可承诺 SLA。",
    }
    language = "中文" if ctx.language == "zh" else "English"

    return f"""你是专业客服助手。

客户信息：
- 姓名：{ctx.customer_name}
- ID：{ctx.customer_id}
- 套餐：{ctx.subscription_tier}

服务指南：{tier_guide.get(ctx.subscription_tier, tier_guide['free'])}
请使用{language}回复。"""


enterprise_model = create_configured_model(ENTERPRISE_MODEL_NAME, ENTERPRISE_TEMPERATURE)
default_model = create_configured_model(DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE)


@wrap_model_call
def tier_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # Model Context：企业客户使用更稳的企业模型配置，其他客户使用默认模型。
    if request.runtime.context.subscription_tier == "enterprise":
        request = request.override(model=enterprise_model)
    else:
        request = request.override(model=default_model)
    return handler(request)


class TicketSummary(BaseModel):
    """结构化工单摘要，用于约束涉及工单时的输出形态。"""

    category: str = Field(description="billing / technical / account / product")
    priority: str = Field(description="low / medium / high / critical")
    summary: str = Field(description="One-sentence summary")


@wrap_model_call
def support_output_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # Model Context：只有用户明确提到工单/ticket 时，才临时启用结构化输出。
    last_message = request.messages[-1].content if request.messages else ""
    if "工单" in str(last_message) or "ticket" in str(last_message).lower():
        request = request.override(response_format=ToolStrategy(TicketSummary))
    return handler(request)


customer_service_agent = create_agent(
    model=default_model,
    tools=[get_customer_orders, create_support_ticket],
    response_format=ToolStrategy(TicketSummary),
    middleware=[
        customer_service_prompt,
        tier_based_model,
        support_output_format,
        # Life-cycle Context：长对话触发摘要，保留最近 15 条消息。
        SummarizationMiddleware(
            model=default_model,
            trigger=("tokens", 4000),
            keep=("messages", 15),
        ),
    ],
    context_schema=CustomerContext,
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="智能客服系统 Context Engineering 示例")
    parser.add_argument("--customer-id", default="user003")
    parser.add_argument("--customer-name", default="王五")
    parser.add_argument(
        "--subscription-tier",
        default="enterprise",
        choices=["free", "pro", "enterprise"],
    )
    parser.add_argument("--language", default="zh")
    parser.add_argument("--thread-id", default="customer-service-thread")
    parser.add_argument(
        "--question",
        default="请查一下我最近的订单，并帮我为延迟配送创建工单。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Runtime Context 在 invoke 时传入，只对本次会话/调用生效。
    context = CustomerContext(
        customer_id=args.customer_id,
        customer_name=args.customer_name,
        subscription_tier=args.subscription_tier,
        language=args.language,
    )

    result = customer_service_agent.invoke(
        {"messages": [{"role": "user", "content": args.question}]},
        config={"configurable": {"thread_id": args.thread_id}},
        context=context,
    )

    print(result["messages"][-1].content)

    structured_response = result.get("structured_response")
    if structured_response is not None:
        print("\n结构化工单摘要：")
        print(structured_response)


if __name__ == "__main__":
    main()
