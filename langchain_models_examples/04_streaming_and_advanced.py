"""
LangChain Models - 流式处理和高级用法示例
演示Token流、异步流、链式调用、批处理等
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
import asyncio
import os

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. Token 流式处理 ====================

def token_streaming_example():
    """Token 流式处理示例"""
    print("=" * 50)
    print("Token 流式处理示例")
    print("=" * 50)

    model = ChatZhipuAI(
        model="glm-4.6",
        streaming=True
    )

    print("\n生成中: ", end="", flush=True)
    for chunk in model.stream([HumanMessage(content="写一首关于春天的短诗")]):
        print(chunk.content, end="", flush=True)

    print("\n")


# ==================== 2. 异步流式处理 ====================

async def async_streaming_example():
    """异步流式处理示例"""
    print("\n" + "=" * 50)
    print("异步流式处理示例")
    print("=" * 50)

    model = ChatZhipuAI(
        model="glm-4.6",
        streaming=True
    )

    print("\n异步生成中: ", end="", flush=True)
    async for chunk in model.astream([
        HumanMessage(content="解释什么是人工智能")
    ]):
        print(chunk.content, end="", flush=True)

    print("\n")


# ==================== 3. 流式工具调用 ====================

@tool
def get_weather(location: str) -> str:
    """获取天气信息"""
    return f"{location}: 晴朗，22°C"


def streaming_tool_calls():
    """流式工具调用示例"""
    print("\n" + "=" * 50)
    print("流式工具调用示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")
    model_with_tools = model.bind_tools([get_weather])

    print("\n流式接收工具调用...")
    gathered = None
    for chunk in model_with_tools.stream("北京的天气怎么样?"):
        gathered = chunk if gathered is None else gathered + chunk

        # 显示工具调用构建过程
        if chunk.tool_call_chunks:
            for tool_chunk in chunk.tool_call_chunks:
                if name := tool_chunk.get("name"):
                    print(f"\n工具: {name}")
                if args := tool_chunk.get("args"):
                    print(".", end="", flush=True)

    # 显示完整工具调用
    if gathered and gathered.tool_calls:
        print("\n\n完整工具调用:")
        for tool_call in gathered.tool_calls:
            print(f"  工具: {tool_call['name']}")
            print(f"  参数: {tool_call['args']}")


# ==================== 4. 链式调用 (Chains) ====================

def chain_example():
    """链式调用示例"""
    print("\n" + "=" * 50)
    print("链式调用示例")
    print("=" * 50)

    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}"),
        ("human", "{input}")
    ])

    # 创建链
    model = ChatZhipuAI(model="glm-4.6")
    chain = prompt | model | StrOutputParser()

    # 调用链
    response = chain.invoke({
        "role": "诗人",
        "input": "写一首关于月亮的诗"
    })

    print(f"\n{response}")


# ==================== 5. 批处理优化 ====================

async def batch_processing_example():
    """批处理示例"""
    print("\n" + "=" * 50)
    print("批处理示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")

    # 准备多个请求
    questions = [
        "什么是AI?",
        "什么是机器学习?",
        "什么是深度学习?"
    ]

    print("\n并行处理多个问题...")

    # 并行处理
    tasks = [
        model.ainvoke([HumanMessage(content=q)])
        for q in questions
    ]

    responses = await asyncio.gather(*tasks)

    for i, (q, r) in enumerate(zip(questions, responses), 1):
        print(f"\n问题 {i}: {q}")
        print(f"回答: {r.content[:100]}...")


# ==================== 6. Fallback 机制 ====================

def fallback_example():
    """Fallback 机制示例"""
    print("\n" + "=" * 50)
    print("Fallback 机制示例")
    print("=" * 50)

    # 主模型
    primary_model = ChatZhipuAI(model="glm-4.6")

    # 备用模型 (假设使用更快的模型作为备用)
    fallback_model = ChatZhipuAI(model="glm-4-flash")

    # 创建带 fallback 的模型
    model_with_fallback = primary_model.with_fallbacks([fallback_model])

    print("\n使用带 Fallback 的模型")
    print("主模型: glm-4.6")
    print("备用模型: glm-4-flash")

    # 如果主模型失败，会自动使用备用模型
    response = model_with_fallback.invoke([HumanMessage(content="你好")])
    print(f"\n回答: {response.content}")


# ==================== 7. 重试配置 ====================

def retry_configuration():
    """重试配置示例"""
    print("\n" + "=" * 50)
    print("重试配置示例")
    print("=" * 50)

    model = ChatZhipuAI(
        model="glm-4.6",
        max_retries=5,  # 最多重试 5 次
        timeout=120,  # 超时时间 120 秒
    )

    print("\n配置:")
    print("  最大重试次数: 5")
    print("  超时时间: 120秒")

    try:
        response = model.invoke([
            HumanMessage(content="介绍一下 Python 语言")
        ])
        print(f"\n成功: {response.content[:150]}...")
    except Exception as e:
        print(f"\n失败: {str(e)}")


# ==================== 8. Token 使用统计 ====================

def token_usage_example():
    """Token 使用统计示例"""
    print("\n" + "=" * 50)
    print("Token 使用统计示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")

    response = model.invoke([
        HumanMessage(content="解释机器学习的基本概念")
    ])

    print(f"\n回答: {response.content[:150]}...")

    # 获取 token 使用信息
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        print(f"\nToken 使用:")
        print(f"  输入: {usage.get('input_tokens', 0)} tokens")
        print(f"  输出: {usage.get('output_tokens', 0)} tokens")
        print(f"  总计: {usage.get('total_tokens', 0)} tokens")


# ==================== 9. 监控和日志 ====================

import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def monitored_call():
    """带监控的模型调用"""
    print("\n" + "=" * 50)
    print("监控和日志示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")

    start_time = time.time()

    try:
        response = model.invoke([
            HumanMessage(content="什么是深度学习?")
        ])

        duration = time.time() - start_time

        logger.info(f"模型调用成功 - 耗时: {duration:.2f}s")
        print(f"\n✓ 调用成功")
        print(f"  耗时: {duration:.2f}秒")
        print(f"  回答: {response.content[:100]}...")

        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            logger.info(f"Token使用 - 输入: {usage.get('input_tokens')}, 输出: {usage.get('output_tokens')}")

    except Exception as e:
        logger.error(f"模型调用失败: {e}")
        raise


# ==================== 10. 流式处理中的 Token 统计 ====================

def streaming_with_stats():
    """流式处理中的统计"""
    print("\n" + "=" * 50)
    print("流式处理 Token 统计")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")

    total_tokens = 0
    content = ""

    print("\n生成中: ", end="", flush=True)
    for chunk in model.stream([HumanMessage(content="介绍人工智能的应用")]):
        content += chunk.content
        print(chunk.content, end="", flush=True)

        # 某些 chunk 包含 usage 信息
        if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
            total_tokens = chunk.usage_metadata.get('total_tokens', 0)

    print(f"\n\n总 tokens: {total_tokens}")


# ==================== 主函数 ====================

async def run_async_examples():
    """运行异步示例"""
    await async_streaming_example()
    await batch_processing_example()


def main():
    """主函数"""
    try:
        # 同步示例
        token_streaming_example()
        streaming_tool_calls()
        chain_example()
        fallback_example()
        retry_configuration()
        token_usage_example()
        monitored_call()
        streaming_with_stats()

        # 异步示例
        print("\n" + "=" * 50)
        print("运行异步示例...")
        print("=" * 50)
        asyncio.run(run_async_examples())

        print("\n" + "=" * 50)
        print("所有示例完成!")
        print("=" * 50)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
