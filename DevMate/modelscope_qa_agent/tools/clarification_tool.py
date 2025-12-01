"""
澄清问题工具

实现问题澄清机制,检测用户问题中缺失的关键信息并生成针对性澄清问题。

核心功能:
- 检测缺失关键信息 (版本号、环境配置、错误信息等)
- 生成针对性澄清问题
- 支持多种技术场景的澄清模板
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class MissingInfo(BaseModel):
    """缺失信息模型

    Attributes:
        category: 缺失信息类别
        description: 缺失信息描述
        importance: 重要性 (high/medium/low)
    """
    category: str = Field(description="缺失信息类别")
    description: str = Field(description="缺失信息的具体描述")
    importance: str = Field(description="重要性等级: high, medium, low")


class ClarificationResult(BaseModel):
    """澄清结果模型

    Attributes:
        needs_clarification: 是否需要澄清
        missing_info_list: 缺失信息列表
        clarification_questions: 澄清问题列表
        confidence: 检测置信度
    """
    needs_clarification: bool = Field(description="是否需要澄清")
    missing_info_list: List[MissingInfo] = Field(default_factory=list, description="缺失的关键信息列表")
    clarification_questions: List[str] = Field(default_factory=list, description="生成的澄清问题列表")
    confidence: float = Field(default=0.0, description="检测置信度 (0.0-1.0)")


class ClarificationTool:
    """澄清问题工具

    检测用户问题中的缺失信息并生成澄清问题。
    """

    def __init__(
        self,
        llm_api_key: str,
        model: str = "qwen-plus",
        temperature: float = 0.3
    ):
        """初始化澄清工具"""
        if not llm_api_key or not llm_api_key.strip():
            raise ValueError("llm_api_key 不能为空")

        self.llm = ChatTongyi(
            model=model,
            temperature=temperature,
            dashscope_api_key=llm_api_key
        )

        print(f"✅ ClarificationTool 初始化成功")

    def detect_missing_info(self, question: str) -> List[MissingInfo]:
        """检测缺失的关键信息"""
        if not question or not question.strip():
            return []

        system_prompt = """你是一个技术问题分析专家。分析用户的技术问题,识别缺失的关键信息。

关键信息类别:
1. **版本信息**: transformers版本、Python版本、CUDA版本、模型版本等
2. **环境配置**: 操作系统、GPU型号、内存大小、环境变量等
3. **错误信息**: 完整错误提示、堆栈跟踪、错误代码等
4. **模型信息**: 模型名称、模型路径、模型来源、模型配置等
5. **代码信息**: 完整代码片段、调用方式、参数设置等
6. **数据信息**: 数据格式、数据大小、数据样例、数据来源等

分析规则:
- 如果问题涉及错误或失败,错误信息是**必需**的 (importance: high)
- 如果问题涉及安装或版本问题,版本信息是**必需**的 (importance: high)
- 如果问题涉及模型使用,模型信息是**必需**的 (importance: high)

请识别问题中缺失的关键信息,并评估其重要性。

{format_instructions}
"""

        user_prompt = """用户问题: {question}

请分析这个问题,识别缺失的关键信息。只列出**确实缺失且对解决问题重要**的信息。"""

        class MissingInfoList(BaseModel):
            """缺失信息列表包装器"""
            missing_info_items: List[MissingInfo] = Field(default_factory=list)

        parser = PydanticOutputParser(pydantic_object=MissingInfoList)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt)
        ])

        chain = prompt | self.llm | parser

        try:
            result = chain.invoke({
                "question": question,
                "format_instructions": parser.get_format_instructions()
            })

            missing_list = result.missing_info_items

            print(f"📋 检测到 {len(missing_list)} 个缺失信息")
            for info in missing_list:
                print(f"   - [{info.importance}] {info.category}: {info.description}")

            return missing_list

        except Exception as e:
            print(f"⚠️  缺失信息检测失败: {e}")
            return []

    def generate_clarification_questions(
        self,
        question: str,
        missing_info_list: List[MissingInfo]
    ) -> List[str]:
        """生成澄清问题"""
        if not missing_info_list:
            return []

        # 按重要性排序
        importance_order = {"high": 0, "medium": 1, "low": 2}
        sorted_missing = sorted(
            missing_info_list,
            key=lambda x: importance_order.get(x.importance, 3)
        )

        system_prompt = """你是一个友好的技术支持专家。基于缺失的关键信息,生成具体、易于回答的澄清问题。

生成规则:
1. 每个缺失信息生成**1个**澄清问题
2. 问题要**具体明确**,指向特定信息
3. 使用**友好、专业**的语气
4. 避免技术术语过于复杂
5. 优先询问 high importance 的信息

示例:
- ✅ 好: "您使用的transformers库版本是多少?"
- ✅ 好: "您的操作系统是 Windows、Mac 还是 Linux?"
- ✅ 好: "能否提供完整的错误信息或堆栈跟踪?"

请为每个缺失信息生成一个澄清问题,返回问题列表。

{format_instructions}
"""

        missing_info_desc = "\n".join([
            f"{i+1}. [{info.importance}] {info.category}: {info.description}"
            for i, info in enumerate(sorted_missing)
        ])

        user_prompt = """原始问题: {question}

缺失的关键信息:
{missing_info}

请为每个缺失信息生成一个澄清问题。"""

        class QuestionList(BaseModel):
            """澄清问题列表"""
            questions: List[str] = Field(description="澄清问题列表")

        parser = PydanticOutputParser(pydantic_object=QuestionList)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt)
        ])

        chain = prompt | self.llm | parser

        try:
            result = chain.invoke({
                "question": question,
                "missing_info": missing_info_desc,
                "format_instructions": parser.get_format_instructions()
            })

            questions = result.questions

            print(f"💬 生成 {len(questions)} 个澄清问题")
            for i, q in enumerate(questions, 1):
                print(f"   {i}. {q}")

            return questions

        except Exception as e:
            print(f"⚠️  澄清问题生成失败: {e}")
            # 降级: 使用简单模板生成
            fallback_questions = []
            for info in sorted_missing[:3]:
                if "版本" in info.category:
                    fallback_questions.append(f"您使用的相关库或工具的版本是多少?")
                elif "错误" in info.category:
                    fallback_questions.append(f"能否提供完整的错误信息?")
                elif "模型" in info.category:
                    fallback_questions.append(f"您使用的是哪个具体模型?")
                elif "环境" in info.category:
                    fallback_questions.append(f"您的开发环境配置是怎样的?")
                else:
                    fallback_questions.append(f"能否提供关于{info.category}的更多详细信息?")

            return fallback_questions

    def check_and_clarify(self, question: str) -> ClarificationResult:
        """检查并生成澄清问题(主方法)"""
        print(f"\n{'='*70}")
        print(f"🔍 澄清检查")
        print(f"{'='*70}")
        print(f"问题: {question}")
        print(f"{'='*70}\n")

        # Step 1: 检测缺失信息
        missing_info_list = self.detect_missing_info(question)

        # Step 2: 判断是否需要澄清
        needs_clarification = any(
            info.importance in ["high", "medium"]
            for info in missing_info_list
        )

        # Step 3: 生成澄清问题
        clarification_questions = []
        if needs_clarification:
            clarification_questions = self.generate_clarification_questions(
                question,
                missing_info_list
            )

        # Step 4: 计算置信度
        confidence = 0.0
        if missing_info_list:
            high_count = sum(1 for info in missing_info_list if info.importance == "high")
            medium_count = sum(1 for info in missing_info_list if info.importance == "medium")
            low_count = sum(1 for info in missing_info_list if info.importance == "low")

            confidence = min(1.0, (high_count * 0.4 + medium_count * 0.3 + low_count * 0.1) / 2.0)

        result = ClarificationResult(
            needs_clarification=needs_clarification,
            missing_info_list=missing_info_list,
            clarification_questions=clarification_questions,
            confidence=confidence
        )

        print(f"\n{'='*70}")
        print(f"✅ 澄清检查完成")
        print(f"{'='*70}")
        print(f"需要澄清: {result.needs_clarification}")
        print(f"缺失信息: {len(result.missing_info_list)} 个")
        print(f"澄清问题: {len(result.clarification_questions)} 个")
        print(f"置信度: {result.confidence:.2f}")
        print(f"{'='*70}\n")

        return result
