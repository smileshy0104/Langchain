"""
ReActAgent - 推理与行动结合的 Agent
使用 LangChain 实现 ReAct (Reasoning and Acting) 范式
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import BaseTool
from core.agents import BaseAgent


# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""


class ReActAgent(BaseAgent):
    """
    ReAct Agent
    实现推理(Reasoning)和行动(Acting)循环
    """

    def __init__(
        self,
        name: str,
        llm: ChatZhipuAI,
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 ReActAgent

        Args:
            name: Agent 名称
            llm: LLM 实例
            tools: 工具列表
            system_prompt: 系统提示词（可选）
            max_steps: 最大执行步数
            custom_prompt: 自定义提示词模板
            **kwargs: 其他参数
        """
        super().__init__(name, llm, tools, system_prompt, **kwargs)
        self.max_steps = max_steps
        self.prompt_template = custom_prompt or REACT_PROMPT_TEMPLATE
        self.current_history: List[str] = []

        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

    def run(self, input_text: str, **kwargs) -> str:
        """
        执行 ReAct Agent

        Args:
            input_text: 用户问题
            **kwargs: 其他参数

        Returns:
            最终答案
        """
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        print("="  * 60)

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n📍 第 {current_step} 步")
            print("-" * 60)

            # 1. 构建提示词
            prompt = self._build_prompt(input_text)

            # 2. 调用 LLM
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.llm.invoke(messages, **kwargs)
                response_text = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                error_msg = f"LLM 调用失败: {str(e)}"
                print(f"❌ {error_msg}")
                return error_msg

            print(f"\n💭 Agent 输出:\n{response_text}")

            # 3. 解析输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"\n🤔 Thought: {thought}")
                self.current_history.append(f"Thought: {thought}")

            if not action:
                print("⚠️  警告: 未检测到 Action，继续下一步")
                continue

            print(f"⚡ Action: {action}")

            # 4. 检查完成条件
            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"\n✅ 任务完成！")
                print("=" * 60)

                # 保存交互
                self._save_interaction(input_text, final_answer)
                return final_answer

            # 5. 执行工具调用
            tool_name, tool_input = self._parse_action(action)

            if tool_name:
                observation = self._execute_tool(tool_name, tool_input)
                print(f"📊 Observation: {observation}")

                self.current_history.append(f"Action: {action}")
                self.current_history.append(f"Observation: {observation}")
            else:
                print("⚠️  警告: 无法解析 Action，继续下一步")

        # 达到最大步数
        final_answer = "抱歉，我无法在限定步数内完成这个任务。请尝试简化问题或提供更多信息。"
        print(f"\n⚠️  达到最大步数限制")
        print("=" * 60)

        self._save_interaction(input_text, final_answer)
        return final_answer

    def _build_prompt(self, question: str) -> str:
        """
        构建 ReAct 提示词

        Args:
            question: 用户问题

        Returns:
            完整提示词
        """
        tools_desc = self.get_tools_description()
        history_str = "\n".join(self.current_history) if self.current_history else "（暂无历史）"

        return self.prompt_template.format(
            tools=tools_desc,
            question=question,
            history=history_str
        )

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析 LLM 输出的 Thought 和 Action

        Args:
            text: LLM 输出文本

        Returns:
            (thought, action) 元组
        """
        thought = None
        action = None

        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 提取 Action
        action_match = re.search(r'Action:\s*(.+?)(?=\n\n|$)', text, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()

        return thought, action

    def _parse_action(self, action: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析 Action 为工具名和输入

        格式: tool_name[tool_input]

        Args:
            action: Action 字符串

        Returns:
            (tool_name, tool_input) 元组
        """
        # 匹配格式: tool_name[input]
        match = re.match(r'(\w+)\[(.+)\]', action)

        if match:
            tool_name = match.group(1).strip()
            tool_input = match.group(2).strip()
            return tool_name, tool_input

        return None, None

    def _parse_action_input(self, action: str) -> str:
        """
        解析 Finish 动作的最终答案

        Args:
            action: Action 字符串

        Returns:
            最终答案
        """
        match = re.search(r'Finish\[(.+)\]', action, re.DOTALL)
        if match:
            return match.group(1).strip()
        return action

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        执行工具调用

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
            print(f"  🔧 执行工具: {tool_name}")
            print(f"  📥 输入: {tool_input}")

            result = tool.invoke(tool_input)

            print(f"  ✅ 输出: {result}")
            return str(result)

        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            print(f"  ❌ {error_msg}")
            return error_msg
