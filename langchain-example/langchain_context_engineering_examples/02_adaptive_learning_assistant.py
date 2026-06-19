"""
案例 2：自适应学习助手。

这个文件沿用 `LangChain_Context_Engineering_详细指南.md` 中的案例 2：
1. 通过 LearnerContext 注入学习者画像。
2. 工具通过 ToolRuntime 读写 Store，实现跨会话学习进度。
3. dynamic_prompt 根据专业水平、学习风格和语言动态调整教学方式。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


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
DEFAULT_TEMPERATURE = float(os.getenv("LANGCHAIN_DEFAULT_TEMPERATURE", "0.7"))


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
    """根据环境变量创建模型。"""

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
class LearnerContext:
    """Runtime Context：调用 Agent 时注入学习者画像。"""

    learner_id: str
    learner_name: str
    expertise_level: Literal["beginner", "intermediate", "expert"]
    learning_style: Literal["visual", "auditory", "reading", "kinesthetic"]
    preferred_language: str = "zh"


@tool
def save_learning_progress(
    topic: str,
    mastery_level: float,
    runtime: ToolRuntime[LearnerContext],
) -> str:
    """Save learning progress to long-term Store."""

    # Tool Context：工具从 runtime.context 获取 learner_id，并把进度写入 Store。
    learner_id = runtime.context.learner_id
    runtime.store.put(
        ("learning_progress", learner_id),
        topic,
        {"mastery": mastery_level},
    )
    return f"已记录 {topic} 的学习进度：{mastery_level * 100:.0f}%"


@tool
def get_learning_history(topic: str, runtime: ToolRuntime[LearnerContext]) -> str:
    """Get previous learning progress from Store."""

    # Tool Context：读取同一 learner_id 下的长期学习记录。
    learner_id = runtime.context.learner_id
    progress = runtime.store.get(("learning_progress", learner_id), topic)

    if progress:
        return f"{topic} 历史掌握度：{progress.value['mastery'] * 100:.0f}%"
    return f"没有 {topic} 的学习记录"


@dynamic_prompt
def adaptive_learning_prompt(request: ModelRequest) -> str:
    # Model Context：根据学习者画像动态调整教学策略。
    ctx = request.runtime.context

    level_guide = {
        "beginner": "使用简单语言，避免术语，多举例说明。",
        "intermediate": "可以使用专业术语，但需要适当解释。",
        "expert": "可以进行深入技术讨论。",
    }
    style_guide = {
        "visual": "多使用图表、流程图、示意图描述。",
        "auditory": "使用对话式讲解，强调重点。",
        "reading": "提供详细文字说明和参考资料。",
        "kinesthetic": "设计动手练习和实践任务。",
    }

    return f"""你是 {ctx.learner_name} 的个人学习助手。

专业水平：{level_guide[ctx.expertise_level]}
学习风格：{style_guide[ctx.learning_style]}
回复语言：{ctx.preferred_language}

如果用户询问已学内容，优先调用学习历史工具。
如果用户完成练习或自评掌握度，请调用 save_learning_progress 保存进度。"""


model = create_configured_model(DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE)


adaptive_learning_agent = create_agent(
    model=model,
    tools=[save_learning_progress, get_learning_history],
    middleware=[adaptive_learning_prompt],
    # Context Schema：指定运行时 Context 的数据结构。
    context_schema=LearnerContext,
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自适应学习助手 Context Engineering 示例")
    parser.add_argument("--learner-id", default="learner001")
    parser.add_argument("--learner-name", default="小林")
    parser.add_argument(
        "--expertise-level",
        default="beginner",
        choices=["beginner", "intermediate", "expert"],
    )
    parser.add_argument(
        "--learning-style",
        default="visual",
        choices=["visual", "auditory", "reading", "kinesthetic"],
    )
    parser.add_argument("--preferred-language", default="zh")
    parser.add_argument("--thread-id", default="adaptive-learning-thread")
    parser.add_argument(
        "--question",
        default="请用适合我的方式讲解 LangChain 的 Tool Context，并给我一个小练习。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Runtime Context 在 invoke 时传入，只对本次会话/调用生效。
    context = LearnerContext(
        learner_id=args.learner_id,
        learner_name=args.learner_name,
        expertise_level=args.expertise_level,
        learning_style=args.learning_style,
        preferred_language=args.preferred_language,
    )

    result = adaptive_learning_agent.invoke(
        {"messages": [{"role": "user", "content": args.question}]},
        config={"configurable": {"thread_id": args.thread_id}},
        context=context,
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
