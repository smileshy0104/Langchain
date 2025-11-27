"""
SimpleAgent - 基础对话 Agent
使用 LangChain 实现简单的对话功能，可选支持工具调用
"""

import re
from typing import List, Optional, Dict, Any
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import BaseTool
from core.agents import BaseAgent


class SimpleAgent(BaseAgent):
    """
    简单对话 Agent
    支持基础对话和可选的工具调用功能
    """

    def __init__(
        self,
        name: str,
        llm: ChatZhipuAI,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        enable_tool_calling: bool = True,
        max_tool_iterations: int = 3,
        **kwargs
    ):
        """
        初始化 SimpleAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            tools: 工具列表
            system_prompt: 系统提示词
            enable_tool_calling: 是否启用工具调用
            max_tool_iterations: 最大工具调用迭代次数
            **kwargs: 其他参数
        """
        super().__init__(name, llm, tools, system_prompt, **kwargs)
        self.enable_tool_calling = enable_tool_calling and tools is not None
        self.max_tool_iterations = max_tool_iterations

        if self.enable_tool_calling:
            print(f"✅ {name} 初始化完成，工具调用已启用")
        else:
            print(f"✅ {name} 初始化完成，工具调用已禁用")

    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return "你是一个友好且乐于助人的 AI 助手。"

    def _get_enhanced_system_prompt(self) -> str:
        """
        获取增强的系统提示词（包含工具信息）

        Returns:
            增强的系统提示词
        """
        base_prompt = self.system_prompt

        if not self.enable_tool_calling or not self.tools:
            return base_prompt

        # 添加工具信息
        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题:\n\n"
        tools_section += self.get_tools_description()

        tools_section += "\n\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式:\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n\n"
        tools_section += "例如:\n"
        tools_section += "- `[TOOL_CALL:calculator:2 + 3]`\n"
        tools_section += "- `[TOOL_CALL:search:Python编程]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"

        return base_prompt + tools_section

    def run(self, input_text: str, **kwargs) -> str:
        """
        执行 Agent

        Args:
            input_text: 用户输入
            **kwargs: 其他参数

        Returns:
            Agent 响应
        """
        print(f"\n🤖 {self.name} 正在处理: {input_text}")

        # 构建消息列表
        messages = []

        # 添加系统消息
        system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # 添加历史消息
        messages.extend(self._history)

        # 添加当前用户消息
        messages.append({"role": "user", "content": input_text})

        # 如果没有启用工具调用，使用简单对话逻辑
        if not self.enable_tool_calling:
            return self._simple_chat(messages, input_text, **kwargs)

        # 支持多轮工具调用的逻辑
        return self._run_with_tools(messages, input_text, **kwargs)

    def _simple_chat(
        self,
        messages: List[Dict[str, str]],
        input_text: str,
        **kwargs
    ) -> str:
        """
        简单对话模式（无工具调用）

        Args:
            messages: 消息列表
            input_text: 用户输入
            **kwargs: 其他参数

        Returns:
            响应文本
        """
        try:
            response = self.llm.invoke(messages, **kwargs)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # 保存交互
            self._save_interaction(input_text, response_text)

            print(f"✅ {self.name} 响应完成")
            return response_text

        except Exception as e:
            error_msg = f"执行失败: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def _run_with_tools(
        self,
        messages: List[Dict[str, str]],
        input_text: str,
        **kwargs
    ) -> str:
        """
        支持工具调用的运行逻辑

        Args:
            messages: 消息列表
            input_text: 用户输入
            **kwargs: 其他参数

        Returns:
            最终响应
        """
        current_iteration = 0
        final_response = ""

        while current_iteration < self.max_tool_iterations:
            # 调用 LLM
            try:
                response = self.llm.invoke(messages, **kwargs)
                response_text = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                return f"LLM 调用失败: {str(e)}"

            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response_text)

            if tool_calls:
                print(f"🔧 检测到 {len(tool_calls)} 个工具调用")

                # 执行所有工具调用并收集结果
                tool_results = []
                clean_response = response_text

                for call in tool_calls:
                    result = self._execute_tool_call(
                        call['tool_name'],
                        call['parameters']
                    )
                    tool_results.append(result)

                    # 从响应中移除工具调用标记
                    clean_response = clean_response.replace(call['original'], "")

                # 添加 assistant 的响应（移除了工具调用标记）
                if clean_response.strip():
                    messages.append({"role": "assistant", "content": clean_response.strip()})

                # 添加工具结果作为新的用户消息
                tool_results_text = "\n\n".join(tool_results)
                messages.append({
                    "role": "user",
                    "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"
                })

                current_iteration += 1
                continue

            # 没有工具调用，这是最终回答
            final_response = response_text
            break

        # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= self.max_tool_iterations and not final_response:
            try:
                response = self.llm.invoke(messages, **kwargs)
                final_response = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                final_response = f"达到最大迭代次数，且最终调用失败: {str(e)}"

        # 保存到历史记录
        self._save_interaction(input_text, final_response)

        print(f"✅ {self.name} 响应完成")
        return final_response

    def _parse_tool_calls(self, text: str) -> List[Dict[str, str]]:
        """
        解析文本中的工具调用

        Args:
            text: 要解析的文本

        Returns:
            工具调用列表，每个元素包含 tool_name, parameters, original
        """
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, text)

        tool_calls = []
        for tool_name, parameters in matches:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })

        return tool_calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            parameters: 工具参数

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
            print(f"  🔧 执行工具: {tool_name}")
            print(f"  📥 输入: {parameters}")

            # 调用工具
            result = tool.invoke(parameters)

            print(f"  ✅ 输出: {result}")
            return f"🔧 工具 {tool_name} 执行结果:\n{result}"

        except Exception as e:
            error_msg = f"❌ 工具调用失败: {str(e)}"
            print(f"  {error_msg}")
            return error_msg

    def add_tool(self, tool: BaseTool):
        """
        动态添加工具

        Args:
            tool: 工具实例
        """
        if tool not in self.tools:
            self.tools.append(tool)
            self.enable_tool_calling = True
            print(f"🔧 工具 '{tool.name}' 已添加到 {self.name}")

    def remove_tool(self, tool_name: str) -> bool:
        """
        移除工具

        Args:
            tool_name: 工具名称

        Returns:
            是否成功移除
        """
        for i, tool in enumerate(self.tools):
            if tool.name == tool_name:
                self.tools.pop(i)
                print(f"🔧 工具 '{tool_name}' 已从 {self.name} 移除")
                if not self.tools:
                    self.enable_tool_calling = False
                return True
        return False

    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return self.enable_tool_calling and len(self.tools) > 0

    def list_tools(self) -> List[str]:
        """列出所有可用工具"""
        return [tool.name for tool in self.tools]
