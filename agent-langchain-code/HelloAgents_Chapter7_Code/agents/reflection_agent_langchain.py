"""
ReflectionAgent - 自我反思 Agent
使用 LangChain 实现自我反思 (Reflection) 范式
"""

from typing import List, Optional, Dict, Any
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from core.agents import BaseAgent

class ReflectionAgent(BaseAgent):
    """
    Reflection Agent
    通过自我反思和改进来提高输出质量
    """

    def __init__(
        self,
        name: str,
        llm: ChatZhipuAI,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        max_reflections: int = 3,
        **kwargs
    ):
        super().__init__(name, llm, tools, system_prompt, **kwargs)
        self.max_reflections = max_reflections
        print(f"✅ {name} 初始化完成，最大反思次数: {max_reflections}")

    def run(self, input_text: str, **kwargs) -> str:
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")
        print("=" * 60)

        # 1. 初始生成
        print("\n📝 第 1 步: 生成初始回答")
        initial_response = self._generate_initial_response(input_text)
        print(f"  初始回答长度: {len(initial_response)} 字符")
        print(f"  内容摘要: {initial_response[:50]}...")
        
        current_response = initial_response
        self._save_interaction(input_text, current_response)

        # 2. 反思循环
        for i in range(self.max_reflections):
            print(f"\n🤔 第 {i+2} 步: 反思与改进 (轮次 {i+1}/{self.max_reflections})")
            
            # 反思
            critique = self._reflect(input_text, current_response)
            print(f"  💡 反思意见: {critique}")
            
            # 简单的终止条件判断
            if "无需改进" in critique or "完美" in critique or "很好" in critique and len(critique) < 20:
                 print("  ✨ 反思认为回答已足够好，结束循环")
                 break

            # 改进
            improved_response = self._refine(input_text, current_response, critique)
            print(f"  ✅ 改进后回答长度: {len(improved_response)} 字符")
            
            current_response = improved_response
            self._save_interaction(f"Refinement {i+1}", current_response)

        print("\n✅ 任务完成！")
        print("=" * 60)
        return current_response

    def _generate_initial_response(self, task: str) -> str:
        """生成初始回答"""
        messages = [
            SystemMessage(content=self.system_prompt or "你是一个乐于助人的助手。"),
            HumanMessage(content=task)
        ]
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)

    def _reflect(self, task: str, current_response: str) -> str:
        """反思当前回答"""
        reflect_prompt = f"""
任务: {task}

当前回答:
{current_response}

请仔细阅读上述任务和回答。
1. 评估回答是否完全解决了任务。
2. 指出回答中的错误、遗漏或不清晰的地方。
3. 给出具体的改进建议。
4. 如果回答已经很完美，请直接输出"无需改进"。
"""
        messages = [
            SystemMessage(content="你是一个严厉的批评家，负责评估AI助手的回答质量。"),
            HumanMessage(content=reflect_prompt)
        ]
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)

    def _refine(self, task: str, current_response: str, critique: str) -> str:
        """根据反思改进回答"""
        refine_prompt = f"""
任务: {task}

当前回答:
{current_response}

反馈意见:
{critique}

请根据上述反馈意见，重写并改进回答。确保解决了所有指出的问题。
只输出改进后的回答内容，不要包含其他解释。
"""
        messages = [
            SystemMessage(content="你是一个专业的编辑，负责根据反馈改进文章。"),
            HumanMessage(content=refine_prompt)
        ]
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)