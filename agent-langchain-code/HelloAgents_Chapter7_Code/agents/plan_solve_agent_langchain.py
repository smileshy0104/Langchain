"""
PlanAndSolveAgent - 计划与执行 Agent
使用 LangChain 实现 Plan-and-Solve 范式
"""

import re
import json
from typing import List, Optional, Dict, Any
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from core.agents import BaseAgent

class PlanAndSolveAgent(BaseAgent):
    """
    Plan-and-Solve Agent
    先制定计划，然后分步骤执行
    """

    def __init__(
        self,
        name: str,
        llm: ChatZhipuAI,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name, llm, tools, system_prompt, **kwargs)
        print(f"✅ {name} 初始化完成")

    def run(self, input_text: str, **kwargs) -> str:
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")
        print("=" * 60)

        # 1. 制定计划
        print("\n📝 第 1 步: 制定计划")
        plan = self._create_plan(input_text)
        print(f"  计划步骤: {len(plan)} 步")
        for i, step in enumerate(plan):
            print(f"  {i+1}. {step}")
        
        self._save_interaction(f"Plan", json.dumps(plan, ensure_ascii=False))

        # 2. 执行计划
        print("\n⚙️ 第 2 步: 执行计划")
        step_results = []
        
        for i, step in enumerate(plan):
            print(f"\n📍 执行步骤 {i+1}/{len(plan)}: {step}")
            
            # 执行当前步骤
            result = self._execute_step(step, step_results)
            print(f"  ✅ 结果: {result[:100]}..." if len(result) > 100 else f"  ✅ 结果: {result}")
            
            step_results.append(f"步骤 {i+1}: {step}\n结果: {result}")

        # 3. 汇总结果
        print("\n📊 第 3 步: 汇总最终答案")
        final_answer = self._generate_final_answer(input_text, step_results)
        
        print(f"\n✅ 任务完成！")
        print("=" * 60)
        
        self._save_interaction(input_text, final_answer)
        return final_answer

    def _create_plan(self, task: str) -> List[str]:
        """制定计划"""
        plan_prompt = f"""
任务: {task}

请将上述任务分解为一系列清晰、简单的步骤。
返回一个 JSON 字符串列表，例如:
["步骤1", "步骤2", "步骤3"]

注意:
1. 步骤之间要有逻辑顺序。
2. 如果任务很简单，可以直接返回一个步骤。
3. 不需要解释，只返回 JSON 列表。
"""
        messages = [
            SystemMessage(content="你是一个专业的项目经理，擅长分解任务。"),
            HumanMessage(content=plan_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 尝试解析 JSON
            # 提取可能被 ```json ... ``` 包裹的内容
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
                
            return json.loads(content)
        except Exception as e:
            print(f"⚠️ 计划解析失败: {e}")
            # 降级策略：按行分割
            return [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('[')]

    def _execute_step(self, step: str, previous_results: List[str]) -> str:
        """执行单个步骤"""
        context = "\n\n".join(previous_results) if previous_results else "无"
        
        execute_prompt = f"""
当前步骤: {step}

之前的执行结果:
{context}

请执行当前步骤。利用之前的执行结果（如果有的话）。
如果需要计算或查询信息，请直接给出结果。
"""
        messages = [
            SystemMessage(content="你是一个高效的执行者。"),
            HumanMessage(content=execute_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)

    def _generate_final_answer(self, task: str, step_results: List[str]) -> str:
        """生成最终答案"""
        context = "\n\n".join(step_results)
        
        summary_prompt = f"""
原始任务: {task}

执行过程:
{context}

请根据上述执行过程，给出最终的完整答案。
"""
        messages = [
            SystemMessage(content="你是一个善于总结的助手。"),
            HumanMessage(content=summary_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)