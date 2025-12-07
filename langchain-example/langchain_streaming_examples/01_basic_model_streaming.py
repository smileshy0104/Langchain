"""
示例1：Model 中使用 Streaming - 基础用法
演示如何在 LangChain Model 中使用流式传输
"""

import os
import time
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessageChunk

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your_api_key")


# ==================== 示例 1.1: 基础流式输出 ====================

def example_01_basic_streaming():
    """示例 1.1: 基础 Token 流式输出"""
    print("\n" + "=" * 60)
    print("示例 1.1: 基础 Token 流式输出")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.7)

    print("\n👤 用户: 用一句话介绍人工智能")
    print("\n🤖 AI 流式响应:")
    print("   ", end="", flush=True)

    # 流式输出，逐个 token 显示
    for chunk in model.stream("用一句话介绍人工智能"):
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", flush=True)
            time.sleep(0.02)  # 模拟打字效果

    print("\n")


# ==================== 示例 1.2: 累积消息块 ====================

def example_02_accumulate_chunks():
    """示例 1.2: 累积消息块"""
    print("\n" + "=" * 60)
    print("示例 1.2: 累积消息块")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.7)

    print("\n👤 用户: 列举三种编程语言")
    print("\n📊 累积过程:")

    full_message = None
    chunk_count = 0

    for chunk in model.stream("列举三种编程语言"):
        chunk_count += 1

        # 累积消息块
        if full_message is None:
            full_message = chunk
        else:
            full_message = full_message + chunk

        # 每5个块显示一次当前累积结果
        if chunk_count % 5 == 0 and hasattr(full_message, 'content'):
            print(f"   块 #{chunk_count}: {full_message.content[:50]}...")

    print(f"\n✅ 完整消息 (共 {chunk_count} 个块):")
    print(f"   类型: {type(full_message)}")
    print(f"   内容: {full_message.content}")


# ==================== 示例 1.3: 流式输出带元数据 ====================

def example_03_streaming_with_metadata():
    """示例 1.3: 流式输出带元数据"""
    print("\n" + "=" * 60)
    print("示例 1.3: 流式输出带元数据")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.7)

    print("\n👤 用户: 解释什么是机器学习")
    print("\n📋 流式块详情:")

    total_content = ""
    chunk_sizes = []

    for i, chunk in enumerate(model.stream("解释什么是机器学习"), 1):
        if hasattr(chunk, 'content') and chunk.content:
            content = chunk.content
            total_content += content
            chunk_sizes.append(len(content))

            # 显示每个块的详细信息
            if i <= 5 or i % 10 == 0:  # 只显示前5个和之后每10个
                print(f"   块 #{i}:")
                print(f"      内容: '{content}'")
                print(f"      长度: {len(content)} 字符")
                if hasattr(chunk, 'response_metadata'):
                    print(f"      元数据: {chunk.response_metadata}")

    print(f"\n📊 统计:")
    print(f"   总块数: {len(chunk_sizes)}")
    print(f"   总字符: {len(total_content)}")
    print(f"   平均块大小: {sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0:.2f} 字符")


# ==================== 示例 1.4: 实时打字效果 ====================

def example_04_typing_effect():
    """示例 1.4: 模拟实时打字效果"""
    print("\n" + "=" * 60)
    print("示例 1.4: 实时打字效果")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.7)

    print("\n👤 用户: 写一首关于春天的短诗")
    print("\n🖊️  AI 正在创作...")
    print("\n" + "-" * 60)

    # 模拟打字机效果
    for chunk in model.stream("写一首关于春天的短诗（四句）"):
        if hasattr(chunk, 'content') and chunk.content:
            for char in chunk.content:
                print(char, end="", flush=True)
                time.sleep(0.05)  # 打字延迟

    print("\n" + "-" * 60)


# ==================== 示例 1.5: 流式输出与非流式对比 ====================

def example_05_streaming_vs_non_streaming():
    """示例 1.5: 流式 vs 非流式性能对比"""
    print("\n" + "=" * 60)
    print("示例 1.5: 流式 vs 非流式性能对比")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.7)
    prompt = "用100字介绍深度学习"

    # 非流式
    print("\n⏱️  非流式调用:")
    print("   等待完整响应...")
    start_time = time.time()
    response = model.invoke(prompt)
    end_time = time.time()
    print(f"   ✅ 完成！总耗时: {end_time - start_time:.2f}秒")
    print(f"   响应: {response.content[:50]}...")

    print("\n" + "-" * 60)

    # 流式
    print("\n⚡ 流式调用:")
    print("   ", end="", flush=True)
    start_time = time.time()
    first_chunk_time = None

    for i, chunk in enumerate(model.stream(prompt)):
        if hasattr(chunk, 'content') and chunk.content:
            if first_chunk_time is None:
                first_chunk_time = time.time()
                print(f"\n   ⚡ 首个块到达: {first_chunk_time - start_time:.2f}秒")
                print("   ", end="", flush=True)

            print(chunk.content, end="", flush=True)

    end_time = time.time()
    print(f"\n   ✅ 完成！总耗时: {end_time - start_time:.2f}秒")

    if first_chunk_time:
        print(f"\n💡 感知性能提升: {(end_time - start_time) - (first_chunk_time - start_time):.2f}秒")


# ==================== 示例 1.6: 处理流式中断 ====================

def example_06_handle_streaming_interruption():
    """示例 1.6: 处理流式传输中断"""
    print("\n" + "=" * 60)
    print("示例 1.6: 处理流式传输中断")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.7)

    print("\n👤 用户: 详细介绍云计算的发展历史")
    print("\n🛑 模拟：在接收到50个字符后中断")
    print("\n🤖 AI 响应:")
    print("   ", end="", flush=True)

    total_chars = 0
    max_chars = 50

    try:
        for chunk in model.stream("详细介绍云计算的发展历史"):
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content

                # 检查是否超过限制
                if total_chars + len(content) > max_chars:
                    # 只打印部分内容
                    remaining = max_chars - total_chars
                    print(content[:remaining], end="", flush=True)
                    print("\n\n   ⚠️  已达到字符限制，中断流式传输")
                    break

                print(content, end="", flush=True)
                total_chars += len(content)

    except KeyboardInterrupt:
        print("\n\n   ⚠️  用户中断")

    print(f"\n   📊 接收字符数: {total_chars}")


# ==================== 主函数 ====================

def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("LangChain Streaming - Model 基础用法")
    print("=" * 60)

    examples = [
        # ("基础 Token 流式输出", example_01_basic_streaming),
        # ("累积消息块", example_02_accumulate_chunks),
        # ("流式输出带元数据", example_03_streaming_with_metadata),
        # ("实时打字效果", example_04_typing_effect),
        # ("流式 vs 非流式对比", example_05_streaming_vs_non_streaming),
        ("处理流式中断", example_06_handle_streaming_interruption),
    ]

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"运行示例 {i}/{len(examples)}: {name}")
        print(f"{'='*60}")
        try:
            func()
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已终止")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
