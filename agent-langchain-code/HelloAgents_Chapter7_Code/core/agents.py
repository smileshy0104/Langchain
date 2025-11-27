"""
Agent 基类
定义所有 Agent 的统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class BaseAgent(ABC):
    """
    Agent 抽象基类
    所有具体 Agent 必须继承此类并实现 run 方法
    """

    def __init__(
        self,
        name: str,
        llm: ChatZhipuAI,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 Agent

        Args:
            name: Agent 名称
            llm: LangChain LLM 实例
            tools: 可用工具列表
            system_prompt: 系统提示词
            **kwargs: 其他配置参数
        """
        self.name = name
        self.llm = llm
        self.tools = tools or []
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.config = kwargs
        self._history: List[Dict[str, str]] = []

    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词（子类可重写）"""
        return "你是一个有用的 AI 助手。"

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """
        执行 Agent（抽象方法，子类必须实现）

        Args:
            input_text: 用户输入
            **kwargs: 其他参数

        Returns:
            Agent 的响应
        """
        pass

    def add_message(self, role: str, content: str):
        """
        添加消息到历史记录

        Args:
            role: 角色（user/assistant/system）
            content: 消息内容
        """
        self._history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """
        获取对话历史

        Returns:
            历史消息列表
        """
        return self._history.copy()

    def clear_history(self):
        """清空对话历史"""
        self._history.clear()
        print(f"✅ {self.name} 的对话历史已清空")

    def get_tools_description(self) -> str:
        """
        获取工具描述

        Returns:
            工具描述字符串
        """
        if not self.tools:
            return "暂无可用工具"

        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")

        return "\n".join(descriptions)

    def _build_messages(
        self,
        user_input: str,
        include_system: bool = True,
        include_history: bool = True
    ) -> List[Dict[str, str]]:
        """
        构建消息列表

        Args:
            user_input: 用户输入
            include_system: 是否包含系统消息
            include_history: 是否包含历史消息

        Returns:
            消息列表
        """
        messages = []

        # 添加系统消息
        if include_system and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 添加历史消息
        if include_history:
            messages.extend(self._history)

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

    def _save_interaction(self, user_input: str, assistant_response: str):
        """
        保存交互到历史记录

        Args:
            user_input: 用户输入
            assistant_response: Agent 响应
        """
        self.add_message("user", user_input)
        self.add_message("assistant", assistant_response)

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(name={self.name}, tools={len(self.tools)})"

    def __repr__(self) -> str:
        """详细表示"""
        return self.__str__()


class ChatAgent(BaseAgent):
    """
    简单的对话 Agent 实现
    可以作为其他 Agent 的基类或直接使用
    """

    def run(self, input_text: str, **kwargs) -> str:
        """
        执行简单对话

        Args:
            input_text: 用户输入
            **kwargs: 其他参数

        Returns:
            Agent 响应
        """
        print(f"🤖 {self.name} 正在思考...")

        # 构建消息
        messages = self._build_messages(input_text)

        # 调用 LLM
        try:
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # 保存交互
            self._save_interaction(input_text, response_text)

            return response_text

        except Exception as e:
            error_msg = f"执行失败: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg


class ToolAgent(BaseAgent):
    """
    支持工具调用的 Agent 基类
    提供工具调用的通用逻辑
    """

    def __init__(
        self,
        name: str,
        llm: ChatZhipuAI,
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        初始化支持工具的 Agent

        Args:
            name: Agent 名称
            llm: LLM 实例
            tools: 工具列表
            system_prompt: 系统提示词
            **kwargs: 其他参数
        """
        super().__init__(name, llm, tools, system_prompt, **kwargs)
        self.max_iterations = kwargs.get('max_iterations', 5)

    def _get_enhanced_system_prompt(self) -> str:
        """
        获取增强的系统提示词（包含工具信息）

        Returns:
            增强的系统提示词
        """
        base_prompt = self.system_prompt

        if not self.tools:
            return base_prompt

        tools_info = "\n\n## 可用工具\n"
        tools_info += self.get_tools_description()
        tools_info += "\n\n请在需要时使用这些工具来帮助回答问题。"

        return base_prompt + tools_info

    def _parse_tool_calls(self, text: str) -> List[Dict[str, str]]:
        """
        从文本中解析工具调用
        子类应该重写此方法以实现特定的解析逻辑

        Args:
            text: 要解析的文本

        Returns:
            工具调用列表
        """
        return []

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        执行单个工具

        Args:
            tool_name: 工具名称
            tool_input: 工具输入

        Returns:
            工具执行结果
        """
        # 查找工具
        tool = None
        for t in self.tools:
            if t.name == tool_name:
                tool = t
                break

        if not tool:
            return f"❌ 错误: 未找到工具 '{tool_name}'"

        try:
            print(f"🔧 执行工具: {tool_name}({tool_input})")
            result = tool.invoke(tool_input)
            print(f"✅ 工具结果: {result}")
            return str(result)
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
