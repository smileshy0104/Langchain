#!/usr/bin/env python3
"""
通用工具模块 - LLM 初始化和辅助函数

提供统一的 LLM 初始化接口，支持:
- 智谱AI GLM-4（推荐，中文优化）
- OpenAI GPT 系列
- 其他 OpenAI 兼容 API
"""

import os
from typing import Literal
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def get_llm(
    provider: Literal["zhipuai", "openai"] = "zhipuai",
    model: str | None = None,
    temperature: float = 0.7,
    streaming: bool = False
):
    """
    获取 LLM 实例

    Args:
        provider: LLM 提供商
            - "zhipuai": 智谱AI GLM（推荐，中文优化）
            - "openai": OpenAI 或兼容 API
        model: 模型名称
            - zhipuai: "glm-4" (默认), "glm-4-flash", "glm-4-plus"
            - openai: "gpt-4", "gpt-3.5-turbo" 等
        temperature: 温度参数（0.0-1.0）
            - 0.0: 确定性输出，适合事实性任务
            - 0.7: 平衡创造性和准确性（默认）
            - 1.0: 最大创造性，适合创意写作
        streaming: 是否启用流式输出

    Returns:
        LangChain LLM 实例

    Raises:
        ValueError: 如果缺少必要的 API 密钥

    Examples:
        >>> # 使用智谱AI GLM-4
        >>> llm = get_llm(provider="zhipuai")

        >>> # 使用 OpenAI GPT-4
        >>> llm = get_llm(provider="openai", model="gpt-4")

        >>> # 启用流式输出
        >>> llm = get_llm(streaming=True)
    """

    if provider == "zhipuai":
        return _get_zhipuai_llm(model, temperature, streaming)
    elif provider == "openai":
        return _get_openai_llm(model, temperature, streaming)
    else:
        raise ValueError(f"不支持的 provider: {provider}")


def _get_zhipuai_llm(model: str | None, temperature: float, streaming: bool):
    """获取智谱AI LLM 实例"""
    from langchain_community.chat_models import ChatZhipuAI

    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        raise ValueError(
            "未检测到有效的 ZHIPUAI_API_KEY。\n"
            "请在 .env 文件中设置，或访问 https://open.bigmodel.cn/ 获取密钥。"
        )

    return ChatZhipuAI(
        model=model or "glm-4.6",
        api_key=api_key,
        temperature=temperature,
        streaming=streaming
    )


def _get_openai_llm(model: str | None, temperature: float, streaming: bool):
    """获取 OpenAI LLM 实例"""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("LLM_API_KEY")
    if not api_key or api_key.startswith("your-"):
        raise ValueError(
            "未检测到有效的 LLM_API_KEY。\n"
            "请在 .env 文件中设置 OpenAI API 密钥。"
        )

    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL_ID", "gpt-4"),
        openai_api_key=api_key,
        openai_api_base=os.getenv("LLM_BASE_URL"),
        temperature=temperature,
        streaming=streaming
    )


def require_env_var(name: str) -> str:
    """
    确保必需的环境变量存在

    Args:
        name: 环境变量名称

    Returns:
        环境变量值

    Raises:
        EnvironmentError: 如果环境变量不存在或无效
    """
    value = os.getenv(name)
    if not value or value.startswith("your-"):
        raise EnvironmentError(
            f"未检测到有效的 {name}。\n"
            f"请在 .env 文件中配置后重试。"
        )
    return value


# 测试代码
if __name__ == "__main__":
    print("🧪 测试 LLM 初始化\n")

    # 测试智谱AI
    try:
        print("1️⃣ 测试智谱AI GLM-4...")
        llm = get_llm(provider="zhipuai", model="glm-4.6", temperature=0.3)
        print(f"   ✅ 成功: {llm.__class__.__name__}")
        print(f"   模型: {llm.model_name}")  # ChatZhipuAI 使用 model_name 属性
        print(f"   温度: {llm.temperature}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")

    print()

    # 测试 OpenAI
    try:
        print("2️⃣ 测试 OpenAI API...")
        llm = get_llm(provider="openai", model="gpt-4", temperature=0.5)
        print(f"   ✅ 成功: {llm.__class__.__name__}")
        print(f"   模型: {llm.model_name}")
        print(f"   温度: {llm.temperature}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")

    print("\n✨ 测试完成！")
