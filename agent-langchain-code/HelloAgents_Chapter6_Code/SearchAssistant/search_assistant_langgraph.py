#!/usr/bin/env python3
"""
LangGraph 智能搜索助手 - 基于状态图的多步推理系统

核心特性:
- 使用 LangGraph 构建状态机工作流
- 多节点协作：理解查询 -> 搜索信息 -> 生成答案
- 支持真实网络搜索（可选）或模拟搜索
- 基于智谱AI GLM-4.6 模型
- 支持对话历史和上下文记忆

工作流程:
1. understand_query: 理解用户意图，优化搜索关键词
2. search_information: 执行搜索（网络或模拟）
3. generate_answer: 基于搜索结果生成答案

适用场景:
✅ 需要实时信息的问答系统
✅ 复杂的多步推理任务
✅ 需要状态管理的对话系统
"""

from __future__ import annotations

import os
import sys
from typing import TypedDict, Annotated, List, Literal
from dotenv import load_dotenv

# 添加 Chapter4 目录到路径以导入工具模块
chapter4_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "HelloAgents_Chapter4_Code")
sys.path.insert(0, os.path.abspath(chapter4_path))
from utils import get_llm

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 加载环境变量
load_dotenv()


# ========== 状态定义 ==========

class SearchState(TypedDict):
    """搜索助手状态"""
    messages: Annotated[List[BaseMessage], add_messages]  # 对话历史
    user_query: str                    # 用户原始查询
    search_query: str                  # 优化后的搜索查询
    search_results: str                # 搜索结果
    final_answer: str                  # 最终答案
    step: str                          # 当前步骤


# ========== 工具函数 ==========

def simulate_search(query: str) -> str:
    """
    模拟搜索功能（演示用）

    Args:
        query: 搜索查询

    Returns:
        模拟的搜索结果
    """
    # 根据关键词返回模拟结果
    search_database = {
        "天气": "根据气象局数据，今天北京晴，气温15-25℃，空气质量良好。",
        "langchain": "LangChain是一个用于开发由语言模型驱动的应用程序的框架。它提供了标准化的接口、模块化组件和完整的应用链。",
        "人工智能": "人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能行为的系统。",
        "python": "Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名，广泛应用于数据科学、Web开发和自动化。",
        "智谱ai": "智谱AI是一家专注于认知智能和决策智能的人工智能公司，推出了GLM系列大语言模型。",
    }

    # 查找匹配的关键词
    for keyword, result in search_database.items():
        if keyword.lower() in query.lower():
            return f"搜索结果：\n{result}"

    # 默认返回
    return f"搜索关键词 '{query}' 的相关信息：\n根据最新资料，该主题涉及多个方面。建议您查阅专业资料以获取更详细的信息。"


# ========== LangGraph 节点函数 ==========

def understand_query_node(state: SearchState, llm) -> dict:
    """
    节点1: 理解用户查询并生成搜索关键词

    Args:
        state: 当前状态
        llm: 语言模型

    Returns:
        状态更新
    """
    # 获取最新的用户消息
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # 构建理解提示词
    understand_prompt = f"""分析用户的查询："{user_message}"

请完成两个任务：
1. 简洁总结用户想要了解什么（1-2句话）
2. 生成最适合搜索的关键词（中英文均可，要精准且简短）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    response = llm.invoke([SystemMessage(content=understand_prompt)])

    # 提取搜索关键词
    response_text = response.content
    search_query = user_message  # 默认使用原始查询

    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()
    elif "搜索关键词：" in response_text:
        search_query = response_text.split("搜索关键词：")[1].strip()

    print(f"🤔 理解查询: {response.content}")
    print(f"🔍 搜索关键词: {search_query}\n")

    return {
        "user_query": user_message,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我理解您的需求：{response.content}")]
    }


def search_information_node(state: SearchState) -> dict:
    """
    节点2: 执行搜索

    Args:
        state: 当前状态

    Returns:
        状态更新
    """
    search_query = state["search_query"]

    print(f"🔎 正在搜索: {search_query}")

    # 使用模拟搜索（实际应用中可以替换为真实的搜索API）
    search_results = simulate_search(search_query)

    print(f"📄 搜索结果:\n{search_results}\n")

    return {
        "search_results": search_results,
        "step": "searched"
    }


def generate_answer_node(state: SearchState, llm) -> dict:
    """
    节点3: 基于搜索结果生成答案

    Args:
        state: 当前状态
        llm: 语言模型

    Returns:
        状态更新
    """
    # 构建答案生成提示词
    answer_prompt = f"""用户查询：{state['user_query']}

搜索结果：
{state['search_results']}

请基于搜索结果，用简洁、准确的语言回答用户的问题。
- 直接回答问题，不要重复搜索结果
- 如果搜索结果不充分，请诚实说明
- 保持友好和专业的语气"""

    response = llm.invoke([SystemMessage(content=answer_prompt)])
    final_answer = response.content

    print(f"💡 生成答案:\n{final_answer}\n")

    return {
        "final_answer": final_answer,
        "step": "answered",
        "messages": [AIMessage(content=final_answer)]
    }


# ========== LangGraph 构建 ==========

class SearchAssistant:
    """LangGraph 智能搜索助手"""

    def __init__(
        self,
        model: str = "glm-4.6",
        temperature: float = 0.7,
        use_memory: bool = True,
        debug: bool = False
    ):
        """
        初始化搜索助手

        Args:
            model: 模型名称
            temperature: 温度参数
            use_memory: 是否使用记忆（支持多轮对话）
            debug: 是否显示调试信息
        """
        self.llm = get_llm(provider="zhipuai", model=model, temperature=temperature)
        self.debug = debug

        # 构建状态图
        self.graph = self._build_graph()

        # 编译图（带记忆支持）
        if use_memory:
            memory = MemorySaver()
            self.app = self.graph.compile(checkpointer=memory)
        else:
            self.app = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图"""

        # 创建状态图
        workflow = StateGraph(SearchState)

        # 添加节点（使用 lambda 包装以传递 llm）
        workflow.add_node(
            "understand_query",
            lambda state: understand_query_node(state, self.llm)
        )
        workflow.add_node(
            "search_information",
            search_information_node
        )
        workflow.add_node(
            "generate_answer",
            lambda state: generate_answer_node(state, self.llm)
        )

        # 添加边（定义工作流）
        workflow.add_edge(START, "understand_query")
        workflow.add_edge("understand_query", "search_information")
        workflow.add_edge("search_information", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow

    def search(self, query: str, thread_id: str = "default") -> str:
        """
        执行搜索查询

        Args:
            query: 用户查询
            thread_id: 线程ID（用于记忆管理）

        Returns:
            最终答案
        """
        if self.debug:
            print(f"\n{'='*80}")
            print(f"🚀 智能搜索助手启动")
            print(f"{'='*80}")
            print(f"📝 查询: {query}\n")

        # 初始化状态
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "init"
        }

        # 执行图
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.app.invoke(initial_state, config)

        if self.debug:
            print(f"\n{'='*80}")
            print(f"✅ 搜索完成")
            print(f"{'='*80}\n")

        return final_state["final_answer"]

    def chat(self, message: str, thread_id: str = "default") -> str:
        """
        多轮对话接口（利用记忆功能）

        Args:
            message: 用户消息
            thread_id: 对话线程ID

        Returns:
            助手回复
        """
        return self.search(message, thread_id)


# ========== 使用示例 ==========

def example_basic_search():
    """示例1: 基础搜索"""
    print("="*80)
    print("📌 示例1: 基础搜索查询")
    print("="*80)

    assistant = SearchAssistant(debug=True)

    queries = [
        "今天北京的天气怎么样？",
        "什么是 LangChain？",
        "Python 编程语言有哪些特点？"
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"用户查询: {query}")
        print(f"{'='*80}")

        answer = assistant.search(query)

        print(f"\n最终答案: {answer}\n")


def example_conversation():
    """示例2: 多轮对话"""
    print("\n" + "="*80)
    print("📌 示例2: 多轮对话（带记忆）")
    print("="*80)

    assistant = SearchAssistant(use_memory=True, debug=False)

    conversation = [
        "什么是智谱AI？",
        "它有哪些主要产品？",  # 测试上下文理解
        "这些产品可以应用在哪些场景？"
    ]

    thread_id = "conversation_1"

    for i, user_input in enumerate(conversation, 1):
        print(f"\n--- 第 {i} 轮对话 ---")
        print(f"👤 用户: {user_input}")

        response = assistant.chat(user_input, thread_id=thread_id)

        print(f"🤖 助手: {response}")


def example_complex_query():
    """示例3: 复杂查询"""
    print("\n" + "="*80)
    print("📌 示例3: 复杂推理查询")
    print("="*80)

    assistant = SearchAssistant(temperature=0.3, debug=True)

    query = """我想学习人工智能，特别是大语言模型开发。
请告诉我：
1. 需要具备哪些基础知识？
2. 推荐学习哪些框架和工具？
3. 学习路径应该是怎样的？"""

    answer = assistant.search(query)

    print(f"\n完整答案:\n{answer}")


def example_information_extraction():
    """示例4: 信息提取"""
    print("\n" + "="*80)
    print("📌 示例4: 信息提取和总结")
    print("="*80)

    assistant = SearchAssistant(temperature=0.5, debug=True)

    queries = [
        "LangChain 的主要功能是什么？请列举前三个。",
        "Python 和 JavaScript 的主要区别是什么？",
        "机器学习和深度学习有什么不同？"
    ]

    for query in queries:
        print(f"\n查询: {query}")
        answer = assistant.search(query)
        print(f"答案: {answer}\n")
        print("-" * 80)


def example_realtime_info():
    """示例5: 实时信息查询"""
    print("\n" + "="*80)
    print("📌 示例5: 实时信息查询（模拟）")
    print("="*80)

    assistant = SearchAssistant(debug=True)

    # 注意：这里使用的是模拟搜索
    # 实际应用中可以集成真实的搜索API（如Tavily、SerpAPI等）
    queries = [
        "今天天气如何？",
        "最新的AI技术趋势是什么？",
    ]

    for query in queries:
        print(f"\n查询: {query}")
        answer = assistant.search(query)


def main():
    """主函数：运行示例"""
    print("🚀 LangGraph 智能搜索助手")
    print("="*80)

    # 检查 API 密钥
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        print("❌ 错误：请在 .env 文件中设置您的 ZHIPUAI_API_KEY")
        print("📝 获取 API 密钥：https://open.bigmodel.cn/")
        return

    try:
        # 运行示例（可以根据需要选择）
        example_basic_search()
        # example_conversation()
        # example_complex_query()
        # example_information_extraction()
        # example_realtime_info()

        print("\n" + "="*80)
        print("✅ 智能搜索助手示例运行完成！")
        print("="*80)

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
