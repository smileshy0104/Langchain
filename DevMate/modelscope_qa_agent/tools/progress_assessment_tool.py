"""对话进度评估工具

实现 Phase 4.4 对话进度评估功能:
- 评估问题解决进度
- 总结已尝试的方法
- 识别已排除的可能性
- 建议后续行动（继续排查、转向其他路径、寻求人工支持）

Author: Claude Code
Created: 2025-12-01
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatTongyi


class ProgressAssessment(BaseModel):
    """进度评估结果模型

    用于记录对话进度评估的结果，包括尝试的方法、排除的可能性和建议。
    """

    # 评估指标
    turn_count: int = Field(description="对话轮次")
    problem_resolved: bool = Field(description="问题是否已解决")
    confidence_score: float = Field(ge=0, le=1, description="解决置信度 (0-1)")

    # 进度总结
    attempted_solutions: List[str] = Field(
        default_factory=list,
        description="已尝试的解决方案列表"
    )
    excluded_causes: List[str] = Field(
        default_factory=list,
        description="已排除的可能原因列表"
    )
    remaining_options: List[str] = Field(
        default_factory=list,
        description="剩余可尝试的选项"
    )

    # 建议
    recommendation: str = Field(description="后续建议 (continue/pivot/escalate)")
    recommendation_reason: str = Field(description="建议理由")
    next_steps: List[str] = Field(
        default_factory=list,
        description="建议的下一步行动"
    )

    # 是否需要人工支持
    needs_human_support: bool = Field(
        default=False,
        description="是否建议转向人工支持"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "turn_count": 6,
                "problem_resolved": False,
                "confidence_score": 0.3,
                "attempted_solutions": [
                    "降低学习率到 0.0001",
                    "调整 batch_size 到 16",
                    "启用梯度裁剪"
                ],
                "excluded_causes": [
                    "学习率过高（已调整）",
                    "显存不足（已优化）"
                ],
                "remaining_options": [
                    "检查数据质量",
                    "更换优化器",
                    "调整模型架构"
                ],
                "recommendation": "pivot",
                "recommendation_reason": "已尝试常规超参数调整，建议从数据质量角度排查",
                "next_steps": [
                    "检查训练数据标注准确性",
                    "验证数据预处理流程",
                    "尝试使用不同的数据集子集"
                ],
                "needs_human_support": False
            }
        }
    }


class ProgressAssessmentTool:
    """对话进度评估工具

    基于对话历史评估问题解决进度，提供后续行动建议。

    Attributes:
        llm: ChatTongyi LLM 客户端
        turn_threshold: 触发主动总结的轮次阈值（默认 5）

    Example:
        >>> tool = ProgressAssessmentTool(llm_api_key="sk-xxx")
        >>> assessment = tool.assess_progress(messages, turn_count=6)
        >>> if assessment.needs_human_support:
        ...     print("建议寻求人工支持")
    """

    def __init__(
        self,
        llm_api_key: str,
        model: str = "qwen-plus",
        temperature: float = 0.3,
        turn_threshold: int = 5
    ):
        """初始化进度评估工具

        Args:
            llm_api_key: 通义千问 API 密钥
            model: 模型名称（默认 qwen-plus）
            temperature: 温度参数（默认 0.3）
            turn_threshold: 触发主动总结的轮次阈值（默认 5）

        Raises:
            ValueError: 如果 API 密钥为空
        """
        if not llm_api_key or not llm_api_key.strip():
            raise ValueError("llm_api_key 不能为空")

        self.llm = ChatTongyi(
            model=model,
            temperature=temperature,
            dashscope_api_key=llm_api_key
        )

        self.turn_threshold = turn_threshold

        print(f"✅ ProgressAssessmentTool 初始化成功")
        print(f"   - 轮次阈值: {turn_threshold}")

    def should_assess(self, turn_count: int) -> bool:
        """判断是否应该进行进度评估

        Args:
            turn_count: 当前对话轮次

        Returns:
            bool: 如果轮次 >= 阈值，返回 True

        Example:
            >>> tool.should_assess(6)  # True
            >>> tool.should_assess(3)  # False
        """
        return turn_count >= self.turn_threshold

    def assess_progress(
        self,
        messages: List[BaseMessage],
        turn_count: int,
        current_question: str = ""
    ) -> ProgressAssessment:
        """评估对话进度

        基于对话历史分析问题解决进度，生成评估报告。

        Args:
            messages: 对话消息历史
            turn_count: 当前对话轮次
            current_question: 当前问题（可选）

        Returns:
            ProgressAssessment: 进度评估结果

        Example:
            >>> messages = [
            ...     HumanMessage(content="模型训练 loss 不下降"),
            ...     AIMessage(content="建议降低学习率"),
            ...     HumanMessage(content="降低了还是不行")
            ... ]
            >>> assessment = tool.assess_progress(messages, turn_count=6)
            >>> print(assessment.recommendation)  # "pivot"
        """
        print(f"\n📊 开始评估对话进度 (轮次: {turn_count})")

        # 构建对话历史摘要
        conversation_summary = self._build_conversation_summary(messages)

        # 使用 LLM 分析进度
        try:
            assessment_result = self._generate_assessment(
                conversation_summary,
                turn_count,
                current_question
            )

            print(f"✅ 进度评估完成")
            print(f"   - 问题解决: {assessment_result.problem_resolved}")
            print(f"   - 置信度: {assessment_result.confidence_score:.2f}")
            print(f"   - 建议: {assessment_result.recommendation}")

            return assessment_result

        except Exception as e:
            print(f"⚠️  进度评估失败: {e}")
            # 返回降级评估
            return self._create_fallback_assessment(turn_count)

    def _build_conversation_summary(self, messages: List[BaseMessage]) -> str:
        """构建对话历史摘要

        Args:
            messages: 对话消息列表

        Returns:
            str: 格式化的对话摘要
        """
        summary_lines = []

        for i, msg in enumerate(messages, 1):
            if isinstance(msg, HumanMessage):
                summary_lines.append(f"[第{i}轮] 用户: {msg.content[:200]}")
            elif isinstance(msg, AIMessage):
                # 提取主要内容（避免过长）
                content = msg.content[:300]
                summary_lines.append(f"[第{i}轮] Agent: {content}...")

        return "\n".join(summary_lines)

    def _generate_assessment(
        self,
        conversation_summary: str,
        turn_count: int,
        current_question: str
    ) -> ProgressAssessment:
        """使用 LLM 生成进度评估

        Args:
            conversation_summary: 对话历史摘要
            turn_count: 对话轮次
            current_question: 当前问题

        Returns:
            ProgressAssessment: 评估结果
        """
        prompt = f"""你是一个对话进度评估专家。请基于以下对话历史，评估问题解决的进度。

**对话轮次**: {turn_count}

**对话历史**:
{conversation_summary}

**当前问题**: {current_question if current_question else "无"}

**评估任务**:
1. 判断问题是否已解决（true/false）
2. 评估解决置信度（0-1）
3. 总结已尝试的解决方案
4. 列出已排除的可能原因
5. 列出剩余可尝试的选项
6. 提供后续建议:
   - "continue": 继续当前排查路径
   - "pivot": 转向其他排查角度
   - "escalate": 建议人工支持
7. 说明建议理由
8. 列出建议的下一步行动

**输出格式**: 请以自然语言描述你的评估，包含以上所有要点。
"""

        # 调用 LLM
        response = self.llm.invoke(prompt)
        assessment_text = response.content

        # 解析 LLM 响应（简化版本 - 提取关键信息）
        return self._parse_assessment_response(assessment_text, turn_count)

    def _parse_assessment_response(
        self,
        assessment_text: str,
        turn_count: int
    ) -> ProgressAssessment:
        """解析 LLM 的评估响应

        Args:
            assessment_text: LLM 生成的评估文本
            turn_count: 对话轮次

        Returns:
            ProgressAssessment: 结构化的评估结果
        """
        # 简化实现：基于关键词和启发式规则解析

        # 判断问题是否解决
        problem_resolved = any(keyword in assessment_text for keyword in [
            "问题已解决", "成功解决", "已经解决", "解决了"
        ])

        # 评估置信度（基于轮次和解决状态）
        if problem_resolved:
            confidence_score = 0.9
        elif turn_count >= 8:
            confidence_score = 0.2  # 多轮未解决，置信度低
        elif turn_count >= 6:
            confidence_score = 0.4
        else:
            confidence_score = 0.6

        # 提取尝试的方案（简化）
        attempted_solutions = []
        if "降低学习率" in assessment_text or "调整学习率" in assessment_text:
            attempted_solutions.append("调整学习率")
        if "batch_size" in assessment_text or "批次大小" in assessment_text:
            attempted_solutions.append("调整批次大小")
        if "优化器" in assessment_text:
            attempted_solutions.append("尝试不同优化器")

        # 提取排除的原因
        excluded_causes = []
        if "不是学习率" in assessment_text or "学习率已调整" in assessment_text:
            excluded_causes.append("学习率问题已排除")
        if "不是显存" in assessment_text or "显存足够" in assessment_text:
            excluded_causes.append("显存问题已排除")

        # 剩余选项
        remaining_options = []
        if "数据" in assessment_text and "数据" not in "".join(attempted_solutions):
            remaining_options.append("检查数据质量")
        if "模型" in assessment_text and "模型" not in "".join(attempted_solutions):
            remaining_options.append("调整模型架构")
        if not remaining_options:
            remaining_options = ["尝试其他超参数组合", "检查代码实现", "咨询专家"]

        # 确定建议
        if problem_resolved:
            recommendation = "continue"
            recommendation_reason = "问题已解决，可继续使用该方案"
            needs_human_support = False
        elif turn_count >= 8:
            recommendation = "escalate"
            recommendation_reason = f"已尝试 {turn_count} 轮对话仍未解决，建议人工支持"
            needs_human_support = True
        elif turn_count >= 6:
            recommendation = "pivot"
            recommendation_reason = "常规方法效果不佳，建议尝试其他排查角度"
            needs_human_support = False
        else:
            recommendation = "continue"
            recommendation_reason = "继续当前排查路径"
            needs_human_support = False

        # 下一步建议
        if recommendation == "escalate":
            next_steps = [
                "整理已尝试的所有方法",
                "准备完整的问题描述和日志",
                "联系技术支持团队"
            ]
        elif recommendation == "pivot":
            next_steps = remaining_options[:3]
        else:
            next_steps = ["继续按照当前方案排查", "收集更多信息", "验证解决效果"]

        return ProgressAssessment(
            turn_count=turn_count,
            problem_resolved=problem_resolved,
            confidence_score=confidence_score,
            attempted_solutions=attempted_solutions if attempted_solutions else ["多种方案"],
            excluded_causes=excluded_causes if excluded_causes else ["部分原因已排除"],
            remaining_options=remaining_options,
            recommendation=recommendation,
            recommendation_reason=recommendation_reason,
            next_steps=next_steps,
            needs_human_support=needs_human_support
        )

    def _create_fallback_assessment(self, turn_count: int) -> ProgressAssessment:
        """创建降级评估结果（当 LLM 调用失败时）

        Args:
            turn_count: 对话轮次

        Returns:
            ProgressAssessment: 基于轮次的简单评估
        """
        if turn_count >= 8:
            return ProgressAssessment(
                turn_count=turn_count,
                problem_resolved=False,
                confidence_score=0.2,
                attempted_solutions=["多种尝试"],
                excluded_causes=["部分原因已排除"],
                remaining_options=["其他排查路径"],
                recommendation="escalate",
                recommendation_reason="对话轮次过多，建议人工支持",
                next_steps=["联系技术支持"],
                needs_human_support=True
            )
        elif turn_count >= 5:
            return ProgressAssessment(
                turn_count=turn_count,
                problem_resolved=False,
                confidence_score=0.4,
                attempted_solutions=["常规方案"],
                excluded_causes=["基本原因已排查"],
                remaining_options=["深入排查", "尝试其他角度"],
                recommendation="pivot",
                recommendation_reason="建议转向其他排查角度",
                next_steps=["检查数据质量", "验证代码逻辑"],
                needs_human_support=False
            )
        else:
            return ProgressAssessment(
                turn_count=turn_count,
                problem_resolved=False,
                confidence_score=0.6,
                attempted_solutions=["初步尝试"],
                excluded_causes=[],
                remaining_options=["继续排查"],
                recommendation="continue",
                recommendation_reason="继续当前排查路径",
                next_steps=["按照建议继续尝试"],
                needs_human_support=False
            )

    def format_assessment_summary(self, assessment: ProgressAssessment) -> str:
        """格式化评估摘要为可读文本

        Args:
            assessment: 评估结果

        Returns:
            str: 格式化的摘要文本

        Example:
            >>> summary = tool.format_assessment_summary(assessment)
            >>> print(summary)
        """
        lines = [
            f"\n{'='*70}",
            f"📊 对话进度评估报告（第 {assessment.turn_count} 轮）",
            f"{'='*70}",
            "",
            f"**问题状态**: {'✅ 已解决' if assessment.problem_resolved else '⏳ 进行中'}",
            f"**解决置信度**: {assessment.confidence_score:.0%}",
            "",
            "**已尝试的方案**:",
        ]

        for i, solution in enumerate(assessment.attempted_solutions, 1):
            lines.append(f"  {i}. {solution}")

        if assessment.excluded_causes:
            lines.append("")
            lines.append("**已排除的可能性**:")
            for i, cause in enumerate(assessment.excluded_causes, 1):
                lines.append(f"  {i}. {cause}")

        if assessment.remaining_options:
            lines.append("")
            lines.append("**剩余可尝试选项**:")
            for i, option in enumerate(assessment.remaining_options, 1):
                lines.append(f"  {i}. {option}")

        lines.extend([
            "",
            f"**建议行动**: {self._get_recommendation_label(assessment.recommendation)}",
            f"**理由**: {assessment.recommendation_reason}",
            "",
            "**下一步建议**:"
        ])

        for i, step in enumerate(assessment.next_steps, 1):
            lines.append(f"  {i}. {step}")

        if assessment.needs_human_support:
            lines.extend([
                "",
                "⚠️  **建议**: 问题较为复杂，建议寻求人工技术支持"
            ])

        lines.append(f"{'='*70}\n")

        return "\n".join(lines)

    def _get_recommendation_label(self, recommendation: str) -> str:
        """获取建议的中文标签

        Args:
            recommendation: 建议类型

        Returns:
            str: 中文标签
        """
        labels = {
            "continue": "继续当前路径",
            "pivot": "转向其他角度",
            "escalate": "寻求人工支持"
        }
        return labels.get(recommendation, recommendation)
