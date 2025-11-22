#!/usr/bin/env python3
"""
Reflection 范式实现 - LangChain v1.0

Reflection = 生成 (Generate) + 评审 (Reflect) + 优化 (Refine)

核心思想:
1. 初始生成: 快速生成初始方案
2. 自我反思: 评审方案质量，找出问题
3. 迭代优化: 根据反思结果改进方案
4. 循环往复: 直到达到质量标准或最大迭代次数

适用场景:
✅ 需要高质量输出（代码、文章、方案等）
✅ 可以通过迭代改进的任务
✅ 有明确质量标准的场景

特点:
- 🎨 追求完美质量
- 🔄 迭代优化
- 📈 持续改进

基于智谱AI GLM-4 模型实现，使用 LCEL 链
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils import get_llm


class ReflectionAgent:
    """
    Reflection 智能体 - LangChain v1.0 实现

    使用三个 LCEL 链:
    1. initial_chain: 初始生成
    2. reflect_chain: 自我反思
    3. refine_chain: 优化改进
    """

    def __init__(
        self,
        model: str = "glm-4",
        temperature: float = 0.2,
        max_iterations: int = 3,
        debug: bool = False
    ):
        """
        初始化 Reflection Agent

        Args:
            model: 模型名称，默认 "glm-4"
            temperature: 温度参数
                - 0.0-0.3: 更确定性，适合代码生成
                - 0.5-0.7: 平衡创造性和准确性，适合文章写作
            max_iterations: 最大迭代次数
            debug: 是否显示调试信息
        """
        # 获取 LLM
        self.llm = get_llm(provider="zhipuai", model=model, temperature=temperature)
        self.max_iterations = max_iterations
        self.debug = debug

        # === 1. 初始执行链 ===
        self.initial_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。

要求:
- 包含完整的函数签名
- 包含详细的文档字符串（docstring）
- 遵循 PEP 8 编码规范
- 代码简洁、可读

直接输出代码，不要包含任何额外的解释。"""),
            ("human", "任务: {task}")
        ])
        self.initial_chain = self.initial_prompt | self.llm | StrOutputParser()

        # === 2. 反思链 ===
        self.reflect_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。

你的任务是审查以下Python代码，并专注于找出其在**算法效率**上的主要瓶颈。

评审要点:
1. 分析时间复杂度和空间复杂度
2. 识别性能瓶颈
3. 提出算法级别的优化建议
4. 如果代码在算法层面已经达到最优，才能回答"无需改进"

请直接输出你的反馈，不要包含任何额外的解释。"""),
            ("human", """# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种**算法上更优**的解决方案来显著提升性能。""")
        ])
        self.reflect_chain = self.reflect_prompt | self.llm | StrOutputParser()

        # === 3. 优化链 ===
        self.refine_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位资深的Python程序员。你正在根据代码评审专家的反馈来优化你的代码。

要求:
- 根据反馈实施具体的优化
- 保持代码的完整性和可读性
- 遵循 PEP 8 编码规范
- 包含完整的函数签名和文档字符串

直接输出优化后的代码，不要包含任何额外的解释。"""),
            ("human", """# 原始任务:
{task}

# 上一轮尝试的代码:
```python
{last_code}
```

# 评审员的反馈:
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。""")
        ])
        self.refine_chain = self.refine_prompt | self.llm | StrOutputParser()

    def run(self, task: str) -> str:
        """
        执行 Reflection 流程

        Args:
            task: 任务描述

        Returns:
            最终生成的代码

        Examples:
            >>> agent = ReflectionAgent()
            >>> code = agent.run("编写一个函数，找出1到n之间所有的素数")
            >>> print(code)
        """
        if self.debug:
            print(f"\n{'='*70}")
            print(f"🎯 Reflection Agent 开始处理任务")
            print(f"{'='*70}")
            print(f"📝 任务: {task}\n")

        # === 1. 初始执行 ===
        if self.debug:
            print("▶️ [阶段1] 初始生成...")

        code = self.initial_chain.invoke({"task": task})

        if self.debug:
            print("✅ 初始代码已生成:")
            print(f"\n```python\n{code}\n```\n")

        # === 2. 迭代循环：反思与优化 ===
        for i in range(self.max_iterations):
            if self.debug:
                print(f"{'='*70}")
                print(f"🔄 [迭代 {i+1}/{self.max_iterations}]")
                print(f"{'='*70}\n")

            # a. 反思
            if self.debug:
                print("🤔 正在进行反思...")

            feedback = self.reflect_chain.invoke({
                "task": task,
                "code": code
            })

            if self.debug:
                print(f"✅ 反馈已生成:")
                print(f"\n{feedback}\n")

            # b. 检查终止条件
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                if self.debug:
                    print("✨ 代码已达最优，停止迭代\n")
                break

            # c. 优化
            if self.debug:
                print("⚡ 正在进行优化...")

            code = self.refine_chain.invoke({
                "task": task,
                "last_code": code,
                "feedback": feedback
            })

            if self.debug:
                print("✅ 优化后的代码:")
                print(f"\n```python\n{code}\n```\n")

        # === 3. 返回最终代码 ===
        if self.debug:
            print(f"{'='*70}")
            print(f"✅ Reflection Agent 处理完成")
            print(f"{'='*70}\n")
            print(f"💡 最终代码:\n```python\n{code}\n```\n")

        return code


# ========== 使用示例 ==========

def example_prime_numbers():
    """示例1: 素数查找函数"""
    print("="*70)
    print("📌 示例1: 编写素数查找函数")
    print("="*70)

    agent = ReflectionAgent(
        temperature=0.2,
        max_iterations=2,
        debug=True
    )

    task = "编写一个Python函数，找出1到n之间所有的素数（prime numbers）。"
    final_code = agent.run(task)

    print("\n" + "="*70)
    print("📊 最终生成的代码:")
    print("="*70)
    print(f"\n```python\n{final_code}\n```")


def example_fibonacci():
    """示例2: 斐波那契数列"""
    print("\n" + "="*70)
    print("📌 示例2: 编写斐波那契数列函数")
    print("="*70)

    agent = ReflectionAgent(
        temperature=0.2,
        max_iterations=2,
        debug=True
    )

    task = "编写一个Python函数，计算第n个斐波那契数。要求高效实现。"
    final_code = agent.run(task)


def example_sorting():
    """示例3: 排序算法"""
    print("\n" + "="*70)
    print("📌 示例3: 编写快速排序函数")
    print("="*70)

    agent = ReflectionAgent(
        temperature=0.2,
        max_iterations=2,
        debug=True
    )

    task = "编写一个Python函数，实现快速排序算法（QuickSort）。"
    final_code = agent.run(task)


def example_data_structure():
    """示例4: 数据结构实现"""
    print("\n" + "="*70)
    print("📌 示例4: 实现LRU缓存")
    print("="*70)

    agent = ReflectionAgent(
        temperature=0.2,
        max_iterations=3,
        debug=True
    )

    task = """设计并实现一个LRU (Least Recently Used) 缓存类。
要求:
1. 支持 get(key) 和 put(key, value) 操作
2. 时间复杂度 O(1)
3. 使用Python实现"""

    final_code = agent.run(task)


def example_algorithm():
    """示例5: 算法问题"""
    print("\n" + "="*70)
    print("📌 示例5: 两数之和问题")
    print("="*70)

    agent = ReflectionAgent(
        temperature=0.1,
        max_iterations=2,
        debug=True
    )

    task = """编写一个Python函数，给定一个整数数组 nums 和一个目标值 target，
找出数组中和为目标值的两个数的索引。
要求: 时间复杂度尽可能低。"""

    final_code = agent.run(task)


def example_text_generation():
    """示例6: 文本生成（非代码）"""
    print("\n" + "="*70)
    print("📌 示例6: 文章写作（测试非代码场景）")
    print("="*70)

    agent = ReflectionAgent(
        temperature=0.7,  # 提高温度以增加创造性
        max_iterations=2,
        debug=True
    )

    # 修改提示词模板以适应文本生成
    agent.initial_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的技术文章作者。请根据要求撰写文章，要求逻辑清晰、内容准确。"),
        ("human", "主题: {task}")
    ])

    agent.reflect_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位严格的文章编辑。评审文章的逻辑性、准确性和可读性。"),
        ("human", "主题: {task}\n\n文章:\n{code}\n\n请提供改进建议。")
    ])

    agent.refine_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业作者。根据编辑反馈优化文章。"),
        ("human", "主题: {task}\n\n原文:\n{last_code}\n\n反馈:\n{feedback}\n\n请输出优化后的文章。")
    ])

    task = "写一篇500字的文章，介绍什么是LangChain以及它的主要特性。"
    final_article = agent.run(task)


def main():
    """主函数：运行示例"""
    print("🚀 Reflection Agent 示例 - LangChain v1.0 + GLM-4")
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
        example_prime_numbers()
        # example_fibonacci()
        # example_sorting()
        # example_data_structure()
        # example_algorithm()
        # example_text_generation()

        print("\n" + "="*70)
        print("✅ Reflection Agent 示例运行完成！")
        print("="*70)

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
