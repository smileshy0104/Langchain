#!/usr/bin/env python3
"""
Plan-and-Solve 范式实现 - LangChain v1.0

Plan-and-Solve = 规划 (Plan) + 执行 (Solve)

核心思想:
1. Planner (规划器): 将复杂问题分解为简单步骤
2. Executor (执行器): 按照计划逐步执行每个步骤
3. 先制定完整计划，再逐步执行

适用场景:
✅ 复杂的多步骤问题
✅ 可以提前规划的任务
✅ 需要结构化解决方案

特点:
- 📋 清晰的执行计划
- 🎯 结构化解决问题
- ⏱️ 可预测的执行流程

基于智谱AI GLM-4 模型实现，使用 LCEL 链
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import List
from utils import get_llm


# 定义计划的输出结构
class Plan(BaseModel):
    """计划输出结构"""
    steps: List[str] = Field(description="步骤列表，每个步骤是一个字符串")


class Planner:
    """
    规划器 - 将复杂问题分解为简单步骤

    使用 LangChain v1.0 LCEL (LangChain Expression Language) 链:
    prompt | llm | parser
    """

    def __init__(self, llm, debug: bool = False):
        """
        初始化规划器

        Args:
            llm: LangChain LLM 实例
            debug: 是否显示调试信息
        """
        self.llm = llm
        self.debug = debug

        # 定义输出解析器
        self.parser = JsonOutputParser(pydantic_object=Plan)

        # 定义提示词模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。

请确保计划中的每个步骤都是:
1. 独立的、可执行的子任务
2. 按照逻辑顺序排列
3. 描述清晰、具体

{format_instructions}

输出严格的 JSON 格式，不要包含额外的解释。"""),
            ("human", "问题: {question}")
        ])

        # 创建 LCEL 链（自动串联所有步骤）
        self.chain = (
            self.prompt.partial(
                format_instructions=self.parser.get_format_instructions()
            )
            | self.llm
            | self.parser
        )

    def plan(self, question: str) -> List[str]:
        """
        生成执行计划

        Args:
            question: 用户问题

        Returns:
            步骤列表

        Examples:
            >>> planner = Planner(llm)
            >>> steps = planner.plan("计算一个数学应用题")
            >>> print(steps)
            ['步骤1', '步骤2', '步骤3']
        """
        if self.debug:
            print(f"\n{'='*70}")
            print(f"📋 正在生成计划...")
            print(f"{'='*70}")

        try:
            # 调用 LCEL 链（自动处理提示词、LLM调用、解析）
            result = self.chain.invoke({"question": question})
            steps = result.get("steps", [])

            if self.debug:
                print(f"✅ 计划已生成:")
                for i, step in enumerate(steps, 1):
                    print(f"   {i}. {step}")

            return steps

        except Exception as e:
            print(f"❌ 生成计划时出错: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return []


class Executor:
    """
    执行器 - 按照计划逐步执行

    使用 LCEL 链自动处理每个步骤
    """

    def __init__(self, llm, debug: bool = False):
        """
        初始化执行器

        Args:
            llm: LangChain LLM 实例
            debug: 是否显示调试信息
        """
        self.llm = llm
        self.debug = debug

        # 定义提示词模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。

你将收到:
- 原始问题
- 完整的计划
- 到目前为止已经完成的步骤和结果
- 当前要执行的步骤

请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:""")
        ])

        # 创建 LCEL 链
        self.chain = self.prompt | self.llm | StrOutputParser()

    def execute(self, question: str, plan: List[str]) -> str:
        """
        执行计划

        Args:
            question: 原始问题
            plan: 步骤列表

        Returns:
            最终答案

        Examples:
            >>> executor = Executor(llm)
            >>> answer = executor.execute(question, ['步骤1', '步骤2'])
        """
        if self.debug:
            print(f"\n{'='*70}")
            print(f"⚙️ 正在执行计划...")
            print(f"{'='*70}")

        history = ""
        final_answer = ""

        for i, step in enumerate(plan, 1):
            if self.debug:
                print(f"\n▶️ 正在执行步骤 {i}/{len(plan)}: {step}")

            # 使用 LCEL 链执行当前步骤
            response = self.chain.invoke({
                "question": question,
                "plan": "\n".join([f"{j+1}. {s}" for j, s in enumerate(plan)]),
                "history": history if history else "无",
                "current_step": step
            })

            # 更新历史
            history += f"步骤 {i}: {step}\n结果: {response}\n\n"
            final_answer = response

            if self.debug:
                print(f"✅ 步骤 {i} 已完成")
                print(f"   结果: {response}")

        return final_answer


class PlanAndSolveAgent:
    """
    Plan-and-Solve 智能体 - LangChain v1.0 实现

    组合 Planner 和 Executor，先规划后执行
    """

    def __init__(
        self,
        model: str = "glm-4",
        temperature: float = 0.3,
        debug: bool = False
    ):
        """
        初始化 Plan-and-Solve Agent

        Args:
            model: 模型名称，默认 "glm-4"
            temperature: 温度参数
                - 0.0-0.3: 更确定性，适合逻辑推理
                - 0.5-0.7: 平衡创造性和准确性
            debug: 是否显示调试信息
        """
        # 获取 LLM
        self.llm = get_llm(provider="zhipuai", model=model, temperature=temperature)
        self.debug = debug

        # 创建规划器和执行器
        self.planner = Planner(self.llm, debug=debug)
        self.executor = Executor(self.llm, debug=debug)

    def run(self, question: str) -> str:
        """
        执行 Plan-and-Solve 流程

        Args:
            question: 用户问题

        Returns:
            最终答案

        Examples:
            >>> agent = PlanAndSolveAgent()
            >>> answer = agent.run("一个数学应用题")
            >>> print(answer)
        """
        if self.debug:
            print(f"\n{'='*70}")
            print(f"🎯 Plan-and-Solve Agent 开始处理问题")
            print(f"{'='*70}")
            print(f"📝 问题: {question}")

        # === 阶段1: 规划 ===
        plan = self.planner.plan(question)

        if not plan:
            error_msg = "无法生成有效的行动计划"
            if self.debug:
                print(f"\n❌ {error_msg}")
            return error_msg

        # === 阶段2: 执行 ===
        final_answer = self.executor.execute(question, plan)

        if self.debug:
            print(f"\n{'='*70}")
            print(f"✅ Plan-and-Solve Agent 处理完成")
            print(f"{'='*70}")
            print(f"💡 最终答案: {final_answer}\n")

        return final_answer


# ========== 使用示例 ==========

def example_basic():
    """示例1: 基础数学应用题"""
    print("="*70)
    print("📌 示例1: 数学应用题")
    print("="*70)

    agent = PlanAndSolveAgent(debug=True)

    question = """一个水果店周一卖出了15个苹果。
周二卖出的苹果数量是周一的两倍。
周三卖出的数量比周二少了5个。
请问这三天总共卖出了多少个苹果？"""

    answer = agent.run(question)

    print(f"\n📊 最终答案: {answer}")


def example_complex_math():
    """示例2: 复杂数学问题"""
    print("\n" + "="*70)
    print("📌 示例2: 复杂数学问题")
    print("="*70)

    agent = PlanAndSolveAgent(temperature=0.1, debug=True)

    question = """小明有100元，买了3本书，每本25元。
然后他用剩下的钱买了2支笔，每支笔的价格是书价格的1/5。
请问:
1. 小明买书花了多少钱？
2. 小明买笔花了多少钱？
3. 小明还剩多少钱？"""

    answer = agent.run(question)


def example_logic_problem():
    """示例3: 逻辑推理问题"""
    print("\n" + "="*70)
    print("📌 示例3: 逻辑推理问题")
    print("="*70)

    agent = PlanAndSolveAgent(temperature=0.5, debug=True)

    question = """A、B、C三个人参加比赛，排名分别是前三名。
已知:
1. A 不是第一名
2. C 不是第三名
3. B 不是第二名
请推理出他们的排名。"""

    answer = agent.run(question)


def example_word_problem():
    """示例4: 应用题"""
    print("\n" + "="*70)
    print("📌 示例4: 实际应用题")
    print("="*70)

    agent = PlanAndSolveAgent(debug=True)

    question = """一辆汽车从A地到B地，全程240公里。
汽车以每小时60公里的速度行驶了2小时后，
由于堵车，速度降到了每小时40公里。
如果总共用了5小时到达，请问:
1. 汽车正常速度行驶了多少公里？
2. 堵车路段有多少公里？"""

    answer = agent.run(question)


def example_planning_task():
    """示例5: 任务规划"""
    print("\n" + "="*70)
    print("📌 示例5: 任务规划问题")
    print("="*70)

    agent = PlanAndSolveAgent(temperature=0.7, debug=True)

    question = """帮我规划一个周末学习计划:
- 周六上午学习 Python
- 周六下午学习 LangChain
- 周日上午复习前一天内容
- 周日下午做一个小项目
每个时段2小时，请给出详细的学习建议。"""

    answer = agent.run(question)


def main():
    """主函数：运行示例"""
    print("🚀 Plan-and-Solve Agent 示例 - LangChain v1.0 + GLM-4")
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
        # example_complex_math()
        # example_logic_problem()
        # example_word_problem()
        # example_planning_task()

        print("\n" + "="*70)
        print("✅ Plan-and-Solve Agent 示例运行完成！")
        print("="*70)

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
