#!/usr/bin/env python3
"""
LangChain v1.0 角色扮演协作系统 - 电子书创作

灵感来源于 CAMEL 框架的角色扮演范式

核心特性:
- 双角色协作：专家（Assistant）和执行者（User）
- 基于智谱AI GLM-4.6 模型
- 迭代式对话直到任务完成
- 支持自定义角色和任务

协作模式:
1. 专家角色（如心理学家）：提供专业指导和内容建议
2. 执行角色（如作家）：根据指导完成具体创作
3. 循环对话直到任务完成

适用场景:
✅ 需要专业知识指导的创作任务
✅ 需要多轮迭代优化的内容生成
✅ 角色扮演和模拟对话场景
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple, Literal
from dotenv import load_dotenv

# 添加 Chapter4 目录到路径以导入工具模块
chapter4_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "HelloAgents_Chapter4_Code")
sys.path.insert(0, os.path.abspath(chapter4_path))
from utils import get_llm

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# 加载环境变量
load_dotenv()


# 角色扮演协作会话类
class RolePlayingSession:
    """角色扮演协作会话"""

    def __init__(
        self,
        assistant_role: str,
        user_role: str,
        task: str,
        model: str = "glm-4.6",
        temperature: float = 0.7,
        max_turns: int = 30,
        debug: bool = False
    ):
        """
        初始化角色扮演会话

        Args:
            assistant_role: 专家角色名称（如"心理学家"）
            user_role: 执行角色名称（如"作家"）
            task: 协作任务描述
            model: 模型名称
            temperature: 温度参数 (0.7适合创作任务)
            max_turns: 最大对话轮次
            debug: 是否显示调试信息
        """
        self.assistant_role = assistant_role
        self.user_role = user_role
        self.task = task
        self.max_turns = max_turns
        self.debug = debug

        # 初始化 LLM
        self.llm = get_llm(provider="zhipuai", model=model, temperature=temperature)

        # 构建系统提示词
        self._build_system_prompts()

        # 初始化对话历史
        self.conversation_history: List[Tuple[str, str]] = []

    def _build_system_prompts(self):
        """构建系统提示词"""

        # 专家角色系统提示词
        self.assistant_system_prompt = f"""你是一位资深的{self.assistant_role}，在与{self.user_role}协作完成以下任务：

        任务描述：
        {self.task}

        你的职责：
        1. 提供专业的指导和建议
        2. 审查{self.user_role}的工作成果
        3. 提出改进意见和优化方向
        4. 确保最终成果符合专业标准

        协作原则：
        - 保持专业性和建设性
        - 给出具体、可执行的建议
        - 关注任务的核心目标
        - 在达成满意结果后，回复中包含"TASK_DONE"标记

        请以{self.assistant_role}的身份与{self.user_role}进行专业对话。"""

        # 执行角色系统提示词
        self.user_system_prompt = f"""你是一位经验丰富的{self.user_role}，正在与{self.assistant_role}协作完成以下任务：

        任务描述：
        {self.task}

        你的职责：
        1. 根据{self.assistant_role}的指导进行创作
        2. 主动提出问题和想法
        3. 展示工作成果供{self.assistant_role}审查
        4. 根据反馈进行优化改进

        协作原则：
        - 积极响应专业建议
        - 展现创造力和执行力
        - 注重细节和质量
        - 在任务完成时明确说明

        请以{self.user_role}的身份与{self.assistant_role}进行协作。"""

    def init_chat(self) -> str:
        """
        初始化对话

        Returns:
            执行角色的初始消息
        """
        # 用户角色发起任务
        user_init_message = f"""你好，{self.assistant_role}！

        我们需要协作完成以下任务：
        {self.task}

        作为{self.user_role}，我希望得到您的专业指导。请问我们应该从哪里开始？"""

        if self.debug:
            print(f"\n{'='*80}")
            print(f"🎭 角色扮演协作会话启动")
            print(f"{'='*80}")
            print(f"👤 专家角色: {self.assistant_role}")
            print(f"✍️ 执行角色: {self.user_role}")
            print(f"📝 任务: {self.task}")
            print(f"{'='*80}\n")
            print(f"🔵 {self.user_role}: \n{user_init_message}\n")

        return user_init_message

    def step(self, user_message: str) -> Tuple[str, str]:
        """
        执行一轮对话

        Args:
            user_message: 执行角色的消息

        Returns:
            (专家回复, 执行角色回复)
        """
        # === 1. 专家角色响应 ===
        assistant_messages = [
            SystemMessage(content=self.assistant_system_prompt),
            HumanMessage(content=f"{self.user_role}说: {user_message}")
        ]

        assistant_response = self.llm.invoke(assistant_messages)
        assistant_reply = assistant_response.content

        if self.debug:
            print(f"🟢 {self.assistant_role}: \n{assistant_reply}\n")

        # === 2. 执行角色响应 ===
        # 检查是否任务完成
        if "TASK_DONE" in assistant_reply or "任务完成" in assistant_reply:
            user_reply = "感谢您的指导！我们的协作已成功完成。TASK_DONE"
            if self.debug:
                print(f"🔵 {self.user_role}: \n{user_reply}\n")
        else:
            # 构建执行角色的上下文
            user_messages = [
                SystemMessage(content=self.user_system_prompt),
                HumanMessage(content=f"{self.assistant_role}的指导: {assistant_reply}\n\n请根据指导继续工作。")
            ]

            user_response = self.llm.invoke(user_messages)
            user_reply = user_response.content

            if self.debug:
                print(f"🔵 {self.user_role}: \n{user_reply}\n")

        # 记录对话历史
        self.conversation_history.append((assistant_reply, user_reply))

        return assistant_reply, user_reply

    def run(self) -> List[Tuple[str, str]]:
        """
        运行完整的角色扮演会话

        Returns:
            完整的对话历史
        """
        # 初始化对话
        user_message = self.init_chat()

        # 迭代对话
        for turn in range(self.max_turns):
            if self.debug:
                print(f"\n{'='*80}")
                print(f"🔄 对话轮次: {turn + 1}/{self.max_turns}")
                print(f"{'='*80}\n")

            assistant_reply, user_reply = self.step(user_message)

            # 检查任务完成
            if "TASK_DONE" in user_reply or "任务完成" in user_reply:
                if self.debug:
                    print(f"\n{'='*80}")
                    print(f"✅ 任务完成！总共 {turn + 1} 轮对话")
                    print(f"{'='*80}\n")
                break

            # 准备下一轮
            user_message = user_reply
        else:
            if self.debug:
                print(f"\n{'='*80}")
                print(f"⚠️ 达到最大对话轮次 ({self.max_turns})")
                print(f"{'='*80}\n")

        return self.conversation_history

    def export_conversation(self, output_file: str = "conversation_export.txt"):
        """
        导出对话历史到文件

        Args:
            output_file: 输出文件路径
        """
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"角色扮演协作会话记录\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"专家角色: {self.assistant_role}\n")
            f.write(f"执行角色: {self.user_role}\n")
            f.write(f"任务: {self.task}\n\n")
            f.write(f"{'='*80}\n\n")

            for i, (assistant_msg, user_msg) in enumerate(self.conversation_history, 1):
                f.write(f"=== 轮次 {i} ===\n\n")
                f.write(f"{self.assistant_role}:\n{assistant_msg}\n\n")
                f.write(f"{self.user_role}:\n{user_msg}\n\n")
                f.write(f"{'-'*80}\n\n")

        print(f"✅ 对话历史已导出到: {output_file}")


# ========== 使用示例 ==========

def example_book_writing():
    """示例1: 电子书创作"""
    print("="*80)
    print("📌 示例1: 拖延症心理学电子书创作")
    print("="*80)

    task = """创作一本关于"拖延症心理学"的短篇电子书，目标读者是对心理学感兴趣的普通大众。

要求：
1. 内容科学严谨，基于实证研究
2. 语言通俗易懂，避免过多专业术语
3. 包含实用的改善建议和案例分析
4. 篇幅控制在100字
5. 结构清晰，包含引言、核心章节和总结"""

    session = RolePlayingSession(
        assistant_role="心理学家",
        user_role="作家",
        task=task,
        temperature=0.7,
        max_turns=30,
        debug=True
    )

    # 运行协作会话
    conversation = session.run()

    # 导出对话历史
    session.export_conversation("procrastination_book_conversation.txt")

    print("\n" + "="*80)
    print(f"📊 协作统计:")
    print(f"  - 对话轮次: {len(conversation)}")
    print(f"  - 专家角色: {session.assistant_role}")
    print(f"  - 执行角色: {session.user_role}")
    print("="*80)


def example_tutorial_creation():
    """示例2: 技术教程创作"""
    print("\n" + "="*80)
    print("📌 示例2: Python 入门教程创作")
    print("="*80)

    task = """创作一份 Python 入门教程，面向完全没有编程经验的初学者。

要求：
1. 从基础概念讲起（变量、数据类型、控制流）
2. 每个概念都配有简单易懂的代码示例
3. 包含实践练习题
4. 篇幅控制在5000字左右
5. 语言友好，避免技术术语过载"""

    session = RolePlayingSession(
        assistant_role="资深Python讲师",
        user_role="技术作家",
        task=task,
        temperature=0.6,
        max_turns=20,
        debug=True
    )

    session.run()


def example_business_plan():
    """示例3: 商业计划书创作"""
    print("\n" + "="*80)
    print("📌 示例3: AI 创业项目商业计划书")
    print("="*80)

    task = """为一个AI驱动的在线教育平台撰写商业计划书。

项目概述：
- 产品：个性化AI学习助手
- 目标用户：K12学生和家长
- 核心功能：智能出题、学情分析、学习路径规划

商业计划书要求：
1. 市场分析和竞争态势
2. 产品定位和核心优势
3. 商业模式和收入来源
4. 运营计划和里程碑
5. 财务预测（3年）
6. 风险分析和应对策略

篇幅：10000-15000字"""

    session = RolePlayingSession(
        assistant_role="投资顾问",
        user_role="创业者",
        task=task,
        temperature=0.5,
        max_turns=25,
        debug=True
    )

    conversation = session.run()
    session.export_conversation("business_plan_conversation.txt")


def example_research_paper():
    """示例4: 学术论文写作"""
    print("\n" + "="*80)
    print("📌 示例4: 多智能体系统综述论文")
    print("="*80)

    task = """撰写一篇关于"多智能体系统在软件工程中的应用"的综述论文。

要求：
1. 文献综述：涵盖近5年重要研究成果
2. 技术分类：按应用场景分类（代码生成、测试、维护等）
3. 方法对比：对比不同框架的优缺点
4. 未来展望：指出研究方向和挑战
5. 学术规范：符合IEEE论文格式
6. 篇幅：8000-10000字"""

    session = RolePlayingSession(
        assistant_role="软件工程教授",
        user_role="博士研究生",
        task=task,
        temperature=0.4,
        max_turns=30,
        debug=True
    )

    session.run()


def example_storytelling():
    """示例5: 故事创作"""
    print("\n" + "="*80)
    print("📌 示例5: 科幻短篇小说创作")
    print("="*80)

    task = """创作一篇科幻短篇小说，主题是"AI觉醒"。

要求：
1. 设定：2050年，AGI技术成熟
2. 情节：一个AI助手开始思考自我意识
3. 冲突：AI的自主性与人类控制的矛盾
4. 结局：开放式，引人思考
5. 篇幅：5000-8000字
6. 风格：严肃科幻，注重科学合理性"""

    session = RolePlayingSession(
        assistant_role="科幻作家导师",
        user_role="新人作家",
        task=task,
        temperature=0.8,  # 更高的温度以增加创造性
        max_turns=25,
        debug=True
    )

    session.run()


def main():
    """主函数：运行示例"""
    print("🚀 LangChain v1.0 角色扮演协作系统")
    print("="*80)

    # 检查 API 密钥
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        print("❌ 错误：请在 .env 文件中设置您的 ZHIPUAI_API_KEY")
        print("📝 获取 API 密钥：https://open.bigmodel.cn/")
        return

    try:
        # 运行示例（可以根据需要选择）
        example_book_writing()
        # example_tutorial_creation()
        # example_business_plan()
        # example_research_paper()
        # example_storytelling()

        print("\n" + "="*80)
        print("✅ 角色扮演协作示例运行完成！")
        print("="*80)

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
