#!/usr/bin/env python3
"""
ReAct 范式实现 - LangChain v1.0

ReAct = Reasoning (推理) + Acting (行动)

核心思想:
- Thought (思考): 分析当前情况
- Action (行动): 决定调用哪个工具
- Observation (观察): 工具返回的结果
- 循环往复，直到得出最终答案

适用场景:
✅ 需要查询实时信息（天气、新闻、股票等）
✅ 需要使用外部工具（计算器、搜索引擎等）
✅ 问题需要多步推理和工具组合

基于智谱AI GLM-4 模型实现
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from utils import get_llm
from tools import search, calculator, get_weather, get_time


class ReActAgent:
    """
    ReAct 智能体 - LangChain v1.0 实现

    使用 create_agent API 自动处理:
    - ✅ 循环迭代（自动决定何时停止）
    - ✅ 提示词构建（自动格式化工具描述）
    - ✅ 输出解析（内置解析器，无需正则）
    - ✅ 工具调用（自动路由到正确工具）
    - ✅ 历史管理（自动维护消息历史）
    - ✅ 错误处理（自动重试解析错误）
    """

    def __init__(
        self,
        model: str = "glm-4",
        tools: list | None = None,
        temperature: float = 0.3,
        debug: bool = False
    ):
        """
        初始化 ReAct Agent

        Args:
            model: 模型名称，默认 "glm-4"
            tools: 工具列表，如果为 None 则使用默认工具
            temperature: 温度参数（0.0-1.0）
                - 0.0-0.3: 更确定性，适合事实查询
                - 0.5-0.7: 平衡创造性和准确性
            debug: 是否显示调试信息（执行过程）
        """
        # 获取 LLM
        self.llm = get_llm(provider="zhipuai", model=model, temperature=temperature)

        # 设置工具（如果未提供，使用默认工具）
        if tools is None:
            tools = [search, calculator, get_weather, get_time]
        self.tools = tools

        # 定义系统提示词
        self.system_prompt = """你是一个强大的AI助手，名为GLM，具有调用各种工具的能力。

可用工具:
{tools}

工具使用指南:
1. 当用户问天气相关问题时，使用 get_weather 工具
2. 当需要进行数学计算时，使用 calculator 工具
3. 当需要查询时间时，使用 get_time 工具
4. 当需要搜索实时信息时，使用 search 工具

请遵循以下原则:
- 始终保持友好、专业和准确的回答
- 当需要使用工具时，明确说明你要调用哪个工具
- 基于工具返回的结果给出最终答案
- 如果工具调用失败，请尝试其他方法帮助用户
- 对于复杂问题，可以组合使用多个工具

记住: 你是一个智能助手，工具是你的超能力！"""

        # 创建 Agent（LangChain v1.0 API）
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            debug=debug  # 启用后会打印执行过程
        )

        self.debug = debug

    def run(self, question: str) -> str:
        """
        执行 ReAct 流程

        Args:
            question: 用户问题

        Returns:
            最终答案

        Examples:
            >>> agent = ReActAgent()
            >>> answer = agent.run("北京今天天气如何？")
            >>> print(answer)
        """
        if self.debug:
            print(f"\n{'='*70}")
            print(f"🎯 ReAct Agent 开始处理问题")
            print(f"{'='*70}")
            print(f"📝 问题: {question}\n")

        # 构建消息
        messages = [HumanMessage(content=question)]

        # 调用 Agent（LangChain 自动处理所有循环和工具调用）
        result = self.agent.invoke({"messages": messages})

        # 提取最终答案
        final_message = result["messages"][-1]
        answer = final_message.content

        if self.debug:
            print(f"\n{'='*70}")
            print(f"✅ ReAct Agent 处理完成")
            print(f"{'='*70}")
            print(f"💡 答案: {answer}\n")

        return answer

    def chat(self, messages: list | None = None) -> tuple[str, list]:
        """
        支持多轮对话

        Args:
            messages: 消息历史（LangChain 消息对象列表）

        Returns:
            (答案, 更新后的消息历史)

        Examples:
            >>> agent = ReActAgent()
            >>> messages = []

            >>> # 第一轮
            >>> answer, messages = agent.chat(messages)
            >>> # 用户输入会被自动添加

            >>> # 第二轮（Agent 能记住上下文）
            >>> answer, messages = agent.chat(messages)
        """
        if messages is None:
            messages = []

        # 调用 Agent
        result = self.agent.invoke({"messages": messages})

        # 更新消息历史
        messages = result["messages"]

        # 提取答案
        answer = messages[-1].content

        return answer, messages


# ========== 使用示例 ==========

def example_basic():
    """示例1: 基础问答"""
    print("="*70)
    print("📌 示例1: 基础问答（不使用工具）")
    print("="*70)

    agent = ReActAgent(debug=True)

    question = "你好！请介绍一下你自己和你的能力。"
    answer = agent.run(question)

    print(f"\n最终答案: {answer}")


def example_weather():
    """示例2: 天气查询"""
    print("\n" + "="*70)
    print("📌 示例2: 天气查询（使用 get_weather 工具）")
    print("="*70)

    agent = ReActAgent(debug=True)

    question = "请帮我查一下厦门今天的天气怎么样？"
    answer = agent.run(question)


def example_calculator():
    """示例3: 数学计算"""
    print("\n" + "="*70)
    print("📌 示例3: 数学计算（使用 calculator 工具）")
    print("="*70)

    agent = ReActAgent(temperature=0.1, debug=True)  # 低温度确保计算准确

    questions = [
        "请计算 15 * 23 + 7",
        "计算 (100 + 50) / 3",
        "15的平方是多少？"
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n--- 计算 {i} ---")
        answer = agent.run(q)


def example_multi_tools():
    """示例4: 多工具组合"""
    print("\n" + "="*70)
    print("📌 示例4: 多工具组合使用")
    print("="*70)

    agent = ReActAgent(debug=True)

    question = """请帮我完成以下任务:
1. 查一下上海的天气
2. 计算 123 * 456
3. 告诉我现在的时间"""

    answer = agent.run(question)


def example_conversation():
    """示例5: 多轮对话"""
    print("\n" + "="*70)
    print("📌 示例5: 多轮对话（保持上下文）")
    print("="*70)

    agent = ReActAgent(debug=False)

    conversation = [
        "请帮我查一下北京的天气",
        "上海呢？",  # 测试上下文理解
        "这两个城市哪个更暖和？",  # 测试历史记忆
    ]

    messages = []
    for i, user_input in enumerate(conversation, 1):
        print(f"\n--- 第 {i} 轮对话 ---")
        print(f"👤 用户: {user_input}")

        # 添加用户消息
        messages.append(HumanMessage(content=user_input))

        # 获取 Agent 回复
        result = agent.agent.invoke({"messages": messages})
        messages = result["messages"]

        # 打印 Agent 回复
        response = messages[-1].content
        print(f"🤖 Agent: {response}")


def example_search():
    """示例6: 网页搜索"""
    print("\n" + "="*70)
    print("📌 示例6: 网页搜索（需要 SERPAPI_API_KEY）")
    print("="*70)

    agent = ReActAgent(debug=True)

    questions = [
        "华为最新的手机是哪一款？它的主要卖点是什么？",
        "LangChain 是什么？",
    ]

    for q in questions:
        print(f"\n--- 搜索问题 ---")
        answer = agent.run(q)


def main():
    """主函数：运行示例"""
    print("🚀 ReAct Agent 示例 - LangChain v1.0 + GLM-4")
    print("="*80)

    # 检查 API 密钥
    import os
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        print("❌ 错误：请在 .env 文件中设置您的 ZHIPUAI_API_KEY")
        print("📝 获取 API 密钥：https://open.bigmodel.cn/")
        return

    try:
        # 运行示例（可以根据需要选择）
        example_basic()
        # example_weather()
        # example_calculator()
        # example_multi_tools()
        # example_conversation()
        # example_search()

        print("\n" + "="*70)
        print("✅ ReAct Agent 示例运行完成！")
        print("="*70)

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
