import asyncio
import os
import time
import dotenv
from typing import List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatZhipuAI

dotenv.load_dotenv(dotenv_path="../../.env")

# 检查API密钥
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key or api_key == "your-zhipu-api-key-here":
    print("❌ 错误：请在.env文件中设置您的ZHIPUAI_API_KEY")
    exit(1)

# 初始化GLM大模型
chat_model = ChatZhipuAI(
    model="glm-4.6",
    temperature=0.7,
    api_key=api_key
)

def create_test_messages(topic: str) -> List:
    """创建测试消息"""
    return [
        SystemMessage(content="你是一位专业的科技知识专家，请用简洁明了的语言解释技术概念。"),
        HumanMessage(content=f"请帮我介绍一下什么是{topic}")
    ]

def sync_test(topic: str = "机器学习") -> Tuple[str, float]:
    """同步调用测试"""
    print(f"🔄 开始同步调用测试：{topic}")
    messages = create_test_messages(topic)

    start_time = time.time()
    response = chat_model.invoke(messages)  # 同步调用
    duration = time.time() - start_time

    print(f"✅ 同步调用完成，耗时：{duration:.2f}秒")
    return response.content, duration

async def async_test(topic: str = "机器学习") -> Tuple[str, float]:
    """异步调用测试"""
    print(f"⚡ 开始异步调用测试：{topic}")
    messages = create_test_messages(topic)

    start_time = time.time()
    response = await chat_model.ainvoke(messages)  # 异步调用
    duration = time.time() - start_time

    print(f"✅ 异步调用完成，耗时：{duration:.2f}秒")
    return response.content, duration

async def run_concurrent_tests(topics: List[str]) -> List[Tuple[str, float]]:
    """并发异步调用测试"""
    print(f"🚀 开始并发异步调用测试，共{len(topics)}个任务")

    start_time = time.time()

    # 创建多个异步任务
    tasks = [async_test(topic) for topic in topics]

    # 并发执行所有任务
    results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    print(f"✅ 并发调用完成，总耗时：{total_time:.2f}秒")
    print(f"📊 平均每个调用耗时：{total_time / len(topics):.2f}秒")

    return results

def run_sequential_sync_tests(topics: List[str]) -> List[Tuple[str, float]]:
    """顺序同步调用测试"""
    print(f"📝 开始顺序同步调用测试，共{len(topics)}个任务")

    start_time = time.time()
    results = []

    for topic in topics:
        content, duration = sync_test(topic)
        results.append((content, duration))

    total_time = time.time() - start_time
    print(f"✅ 顺序调用完成，总耗时：{total_time:.2f}秒")
    print(f"📊 平均每个调用耗时：{total_time / len(topics):.2f}秒")

    return results

def print_summary(sync_time: float, async_time: float, concurrent_time: float, num_tasks: int):
    """打印性能对比总结"""
    print("\n" + "="*60)
    print("📈 性能对比总结")
    print("="*60)
    print(f"任务数量: {num_tasks}")
    print(f"顺序同步调用: {sync_time:.2f}秒")
    print(f"单个异步调用: {async_time:.2f}秒")
    print(f"并发异步调用: {concurrent_time:.2f}秒")

    if concurrent_time > 0:
        speedup = sync_time / concurrent_time
        print(f"🚀 并发加速比: {speedup:.2f}x")

        if speedup > 1.5:
            print("💡 异步并发显著提升了性能！")
        elif speedup > 1.1:
            print("💡 异步并发提升了性能。")
        else:
            print("💡 在此场景下异步并发提升有限。")

async def main():
    """主测试函数"""
    print("🎯 GLM-4.6 同步 vs 异步调用性能测试")
    print("="*60)

    # 测试主题
    test_topics = ["机器学习", "深度学习", "自然语言处理"]

    # 1. 单个调用对比
    print("\n🔍 单个调用对比测试")
    print("-" * 40)

    sync_content, sync_duration = sync_test("机器学习")
    print(f"📝 同步响应内容: {sync_content[:100]}...\n")

    async_content, async_duration = await async_test("机器学习")
    print(f"📝 异步响应内容: {async_content[:100]}...\n")

    # 2. 多任务性能对比
    print("\n🏃‍♂️ 多任务性能对比测试")
    print("-" * 40)

    # 顺序同步调用
    sync_results = run_sequential_sync_tests(test_topics)
    sync_total_time = sum(duration for _, duration in sync_results)

    # 并发异步调用
    async_results = await run_concurrent_tests(test_topics)
    async_total_time = sum(duration for _, duration in async_results)

    # 3. 打印总结
    print_summary(sync_total_time, async_duration, async_total_time, len(test_topics))

    # 4. 显示部分结果
    print("\n📋 部分回答展示:")
    print("-" * 40)
    for i, (content, _) in enumerate(async_results[:2], 1):
        print(f"问题 {i}: {test_topics[i-1]}")
        print(f"回答: {content[:150]}...")
        print()

if __name__ == "__main__":
    # 运行完整的异步测试
    asyncio.run(main())