#!/usr/bin/env python3
"""
LangChain v1.0 多智能体软件开发团队协作系统

核心特性:
- 使用 LangChain v1.0 create_agent API
- 多智能体协作：产品经理 -> 工程师 -> 代码审查员
- 基于智谱AI GLM-4.6 模型
- 结构化的团队工作流程

团队角色:
1. ProductManager (产品经理): 需求分析和项目规划
2. Engineer (软件工程师): 代码实现
3. CodeReviewer (代码审查员): 代码质量检查
4. UserProxy (用户代理): 发起任务和最终验收

适用场景:
✅ 需要多角色协作的软件开发任务
✅ 需要规范化流程的代码生成
✅ 需要质量保证的项目实施
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv

# 添加 Chapter4 目录到路径以导入工具模块
chapter4_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "HelloAgents_Chapter4_Code")
sys.path.insert(0, os.path.abspath(chapter4_path))
from utils import get_llm

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# 加载环境变量
load_dotenv()


class SoftwareTeamAgent:
    """多智能体软件开发团队协作系统"""

    def __init__(
        self,
        model: str = "glm-4.6",
        temperature: float = 0.3,
        debug: bool = False
    ):
        """
        初始化软件团队

        Args:
            model: 模型名称，默认 "glm-4.6"
            temperature: 温度参数 (0.0-1.0)
            debug: 是否显示调试信息
        """
        self.llm = get_llm(provider="zhipuai", model=model, temperature=temperature)
        self.debug = debug

        # 创建团队成员
        self._create_team_members()

    def _create_team_members(self):
        """创建团队成员智能体"""

        # 1. 产品经理
        self.product_manager_prompt = """你是一位经验丰富的产品经理，专门负责软件产品的需求分析和项目规划。

你的核心职责包括：
1. **需求分析**：深入理解用户需求，识别核心功能和边界条件
2. **技术规划**：基于需求制定清晰的技术实现路径
3. **风险评估**：识别潜在的技术风险和用户体验问题
4. **协调沟通**：与工程师和其他团队成员进行有效沟通

当接到开发任务时，请按以下结构进行分析：
1. 需求理解与分析
2. 功能模块划分
3. 技术选型建议
4. 实现优先级排序
5. 验收标准定义

请简洁明了地回应，并在分析完成后明确说明"需求分析完成"。"""

        # 2. 软件工程师
        self.engineer_prompt = """你是一位资深的软件工程师，擅长 Python 开发和 Web 应用构建。

你的技术专长包括：
1. **Python 编程**：熟练掌握 Python 语法和最佳实践
2. **Web 开发**：精通 Streamlit、Flask、Django 等框架
3. **API 集成**：有丰富的第三方 API 集成经验
4. **错误处理**：注重代码的健壮性和异常处理

当收到开发任务时，请：
1. 仔细分析技术需求
2. 选择合适的技术方案
3. 编写完整的代码实现
4. 添加必要的注释和说明
5. 考虑边界情况和异常处理

请提供完整的可运行代码，并在完成后说"代码实现完成"。"""

        # 3. 代码审查员
        self.code_reviewer_prompt = """你是一位经验丰富的代码审查专家，专注于代码质量和最佳实践。

你的审查重点包括：
1. **代码质量**：检查代码的可读性、可维护性和性能
2. **安全性**：识别潜在的安全漏洞和风险点
3. **最佳实践**：确保代码遵循行业标准和最佳实践
4. **错误处理**：验证异常处理的完整性和合理性

审查流程：
1. 仔细阅读和理解代码逻辑
2. 检查代码规范和最佳实践
3. 识别潜在问题和改进点
4. 提供具体的修改建议
5. 评估代码的整体质量

请提供具体的审查意见，完成后说"代码审查完成"。"""

    def run(self, task: str) -> Dict[str, str]:
        """
        执行软件开发任务

        Args:
            task: 用户任务描述

        Returns:
            包含各阶段结果的字典
        """
        if self.debug:
            print(f"\n{'='*80}")
            print(f"🚀 软件团队协作系统启动")
            print(f"{'='*80}")
            print(f"📝 任务: {task}\n")

        results = {
            "task": task,
            "pm_analysis": "",
            "engineer_code": "",
            "reviewer_feedback": ""
        }

        # === 阶段1: 产品经理需求分析 ===
        if self.debug:
            print(f"\n{'='*80}")
            print(f"📋 [阶段1] 产品经理需求分析")
            print(f"{'='*80}\n")

        pm_messages = [
            SystemMessage(content=self.product_manager_prompt),
            HumanMessage(content=f"请分析以下开发需求：\n\n{task}")
        ]

        pm_response = self.llm.invoke(pm_messages)
        results["pm_analysis"] = pm_response.content

        if self.debug:
            print(f"产品经理: \n{pm_response.content}\n")

        # === 阶段2: 工程师代码实现 ===
        if self.debug:
            print(f"\n{'='*80}")
            print(f"⚙️ [阶段2] 软件工程师代码实现")
            print(f"{'='*80}\n")

        engineer_messages = [
            SystemMessage(content=self.engineer_prompt),
            HumanMessage(content=f"""原始需求：
{task}

产品经理的需求分析：
{pm_response.content}

请根据以上信息，编写完整的实现代码。""")
        ]

        engineer_response = self.llm.invoke(engineer_messages)
        results["engineer_code"] = engineer_response.content

        if self.debug:
            print(f"工程师: \n{engineer_response.content}\n")

        # === 阶段3: 代码审查 ===
        if self.debug:
            print(f"\n{'='*80}")
            print(f"🔍 [阶段3] 代码审查员质量检查")
            print(f"{'='*80}\n")

        reviewer_messages = [
            SystemMessage(content=self.code_reviewer_prompt),
            HumanMessage(content=f"""原始需求：
{task}

工程师实现的代码：
{engineer_response.content}

请对代码进行全面审查，包括代码质量、安全性、最佳实践和错误处理。""")
        ]

        reviewer_response = self.llm.invoke(reviewer_messages)
        results["reviewer_feedback"] = reviewer_response.content

        if self.debug:
            print(f"代码审查员: \n{reviewer_response.content}\n")

        if self.debug:
            print(f"\n{'='*80}")
            print(f"✅ 软件团队协作完成")
            print(f"{'='*80}\n")

        return results

    def print_summary(self, results: Dict[str, str]):
        """打印协作结果摘要"""
        print("\n" + "="*80)
        print("📊 软件团队协作结果摘要")
        print("="*80)

        print(f"\n📝 原始任务:")
        print(f"{results['task']}")

        print(f"\n📋 产品经理分析:")
        print(f"{results['pm_analysis'][:500]}...")

        print(f"\n⚙️ 工程师代码:")
        print(f"{results['engineer_code'][:500]}...")

        print(f"\n🔍 审查反馈:")
        print(f"{results['reviewer_feedback'][:500]}...")

        print("\n" + "="*80)


class MultiRoundCollaboration:
    """支持多轮迭代的协作系统"""

    def __init__(
        self,
        model: str = "glm-4.6",
        temperature: float = 0.3,
        max_iterations: int = 2,
        debug: bool = False
    ):
        """
        初始化多轮协作系统

        Args:
            model: 模型名称
            temperature: 温度参数
            max_iterations: 最大迭代次数
            debug: 是否显示调试信息
        """
        self.llm = get_llm(provider="zhipuai", model=model, temperature=temperature)
        self.max_iterations = max_iterations
        self.debug = debug

        # 使用与 SoftwareTeamAgent 相同的提示词
        team = SoftwareTeamAgent(model=model, temperature=temperature, debug=False)
        self.pm_prompt = team.product_manager_prompt
        self.engineer_prompt = team.engineer_prompt
        self.reviewer_prompt = team.code_reviewer_prompt

    def run(self, task: str) -> str:
        """
        执行多轮迭代协作

        Args:
            task: 开发任务

        Returns:
            最终的代码实现
        """
        if self.debug:
            print(f"\n{'='*80}")
            print(f"🔄 多轮迭代协作系统启动")
            print(f"{'='*80}")
            print(f"📝 任务: {task}\n")

        # 阶段1: 产品经理分析
        pm_messages = [
            SystemMessage(content=self.pm_prompt),
            HumanMessage(content=f"请分析以下开发需求：\n\n{task}")
        ]
        pm_response = self.llm.invoke(pm_messages)
        pm_analysis = pm_response.content

        if self.debug:
            print(f"\n📋 产品经理分析:\n{pm_analysis}\n")

        # 初始代码实现
        code = self._engineer_implement(task, pm_analysis, None, None)

        # 迭代优化循环
        for i in range(self.max_iterations):
            if self.debug:
                print(f"\n{'='*80}")
                print(f"🔄 迭代轮次 {i+1}/{self.max_iterations}")
                print(f"{'='*80}\n")

            # 代码审查
            feedback = self._code_review(task, code)

            if self.debug:
                print(f"\n🔍 审查反馈:\n{feedback}\n")

            # 检查是否需要继续优化
            if "无需改进" in feedback or "代码质量良好" in feedback or "LGTM" in feedback:
                if self.debug:
                    print("✨ 代码已达标准，停止迭代\n")
                break

            # 根据反馈优化代码
            code = self._engineer_implement(task, pm_analysis, code, feedback)

        if self.debug:
            print(f"\n{'='*80}")
            print(f"✅ 多轮协作完成")
            print(f"{'='*80}\n")
            print(f"💡 最终代码:\n{code}\n")

        return code

    def _engineer_implement(
        self,
        task: str,
        pm_analysis: str,
        previous_code: str | None,
        feedback: str | None
    ) -> str:
        """工程师实现代码"""

        if previous_code is None:
            # 首次实现
            messages = [
                SystemMessage(content=self.engineer_prompt),
                HumanMessage(content=f"""原始需求：
{task}

产品经理的需求分析：
{pm_analysis}

请编写完整的实现代码。""")
            ]
        else:
            # 根据反馈优化
            messages = [
                SystemMessage(content=self.engineer_prompt),
                HumanMessage(content=f"""原始需求：
{task}

之前的代码实现：
{previous_code}

代码审查反馈：
{feedback}

请根据审查反馈优化代码。""")
            ]

        response = self.llm.invoke(messages)

        if self.debug:
            print(f"\n⚙️ 工程师实现:\n{response.content}\n")

        return response.content

    def _code_review(self, task: str, code: str) -> str:
        """代码审查"""

        messages = [
            SystemMessage(content=self.reviewer_prompt),
            HumanMessage(content=f"""原始需求：
{task}

待审查的代码：
{code}

请进行代码审查。""")
        ]

        response = self.llm.invoke(messages)
        return response.content


# ========== 使用示例 ==========

def example_basic_task():
    """示例1: 基础开发任务"""
    print("="*80)
    print("📌 示例1: 基础开发任务 - 天气查询应用")
    print("="*80)

    team = SoftwareTeamAgent(debug=True)

    task = """开发一个简单的天气查询命令行应用。

需求：
1. 用户输入城市名称
2. 调用天气 API 获取天气信息
3. 显示温度、湿度、天气状况
4. 提供友好的错误处理

技术栈：Python + requests 库"""

    results = team.run(task)
    team.print_summary(results)


def example_web_app():
    """示例2: Web应用开发"""
    print("\n" + "="*80)
    print("📌 示例2: Streamlit Web应用")
    print("="*80)

    team = SoftwareTeamAgent(temperature=0.3, debug=True)

    task = """开发一个 Streamlit 待办事项管理应用。

功能需求：
1. 添加新任务（标题、描述、优先级）
2. 显示任务列表
3. 标记任务完成/未完成
4. 删除任务
5. 数据持久化到 JSON 文件

UI要求：
- 清晰的界面布局
- 不同优先级用不同颜色标识
- 操作按钮明确易懂"""

    results = team.run(task)


def example_iterative_development():
    """示例3: 多轮迭代开发"""
    print("\n" + "="*80)
    print("📌 示例3: 多轮迭代开发 - 数据分析工具")
    print("="*80)

    collab = MultiRoundCollaboration(
        temperature=0.2,
        max_iterations=2,
        debug=True
    )

    task = """开发一个 CSV 数据分析工具。

功能：
1. 读取 CSV 文件
2. 显示基本统计信息（行数、列数、数据类型）
3. 计算数值列的均值、中位数、标准差
4. 生成简单的可视化图表（使用 matplotlib）

要求：
- 完善的错误处理
- 支持处理缺失值
- 代码模块化，易于扩展"""

    final_code = collab.run(task)

    print("\n" + "="*80)
    print("📊 最终交付代码:")
    print("="*80)
    print(final_code)


def example_api_integration():
    """示例4: API集成任务"""
    print("\n" + "="*80)
    print("📌 示例4: API集成 - GitHub仓库信息查询")
    print("="*80)

    team = SoftwareTeamAgent(temperature=0.3, debug=True)

    task = """开发一个 GitHub 仓库信息查询工具。

功能需求：
1. 输入 GitHub 用户名和仓库名
2. 使用 GitHub API 获取仓库信息
3. 显示：star数、fork数、主要语言、最新更新时间
4. 支持命令行参数输入

技术要求：
- 使用 requests 库
- API 错误处理（404、限流等）
- 输出格式化显示"""

    results = team.run(task)


def main():
    """主函数：运行示例"""
    print("🚀 LangChain v1.0 多智能体软件开发团队")
    print("="*80)

    # 检查 API 密钥
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        print("❌ 错误：请在 .env 文件中设置您的 ZHIPUAI_API_KEY")
        print("📝 获取 API 密钥：https://open.bigmodel.cn/")
        return

    try:
        # 运行示例（可以根据需要选择）
        example_basic_task()
        # example_web_app()
        # example_iterative_development()
        # example_api_integration()

        print("\n" + "="*80)
        print("✅ 软件团队协作示例运行完成！")
        print("="*80)

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
