"""
测试魔搭社区问答 Prompt 模板

测试 Prompt 模板的功能:
- Prompt 模板获取
- 变量验证
- 输出解析器配置
- Prompt 统计
- 实际场景测试
"""

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from prompts.qa_prompts import (
    get_qa_prompt,
    get_qa_prompt_with_parser,
    validate_prompt_variables,
    get_prompt_stats,
    QA_SYSTEM_PROMPT,
    CLARIFICATION_PROMPT,
    RERANK_PROMPT,
    ANSWER_VALIDATION_PROMPT
)
from models.schemas import TechnicalAnswer


class TestGetQAPrompt:
    """测试 get_qa_prompt() 函数"""

    def test_get_basic_prompt(self):
        """测试获取基本 Prompt 模板"""
        prompt = get_qa_prompt()

        assert isinstance(prompt, ChatPromptTemplate)
        assert "context" in prompt.input_variables
        assert "question" in prompt.input_variables
        assert "format_instructions" in prompt.input_variables
        assert len(prompt.messages) == 2  # system + human

        print("✅ 基本 Prompt 模板测试通过")

    def test_prompt_with_format_instructions(self):
        """测试包含格式化指令的 Prompt"""
        prompt = get_qa_prompt(include_format_instructions=True)

        # 格式化一个示例
        formatted = prompt.format(
            context="示例上下文",
            question="示例问题",
            format_instructions="示例格式"
        )

        assert "示例上下文" in formatted
        assert "示例问题" in formatted
        assert "示例格式" in formatted

        print("✅ 格式化指令 Prompt 测试通过")

    def test_prompt_without_format_instructions(self):
        """测试不包含格式化指令的 Prompt"""
        prompt = get_qa_prompt(include_format_instructions=False)

        # 不应该有 format_instructions 变量
        assert "format_instructions" not in prompt.input_variables
        assert "context" in prompt.input_variables
        assert "question" in prompt.input_variables

        print("✅ 无格式化指令 Prompt 测试通过")

    def test_custom_system_prompt(self):
        """测试自定义系统 Prompt"""
        custom_prompt = "你是一个测试专家。\n{context}\n{format_instructions}"

        prompt = get_qa_prompt(custom_system_prompt=custom_prompt)

        formatted = prompt.format(
            context="测试上下文",
            question="测试问题",
            format_instructions="测试格式"
        )

        assert "测试专家" in formatted
        assert "魔搭社区" not in formatted  # 不应该包含默认 Prompt

        print("✅ 自定义系统 Prompt 测试通过")


class TestGetQAPromptWithParser:
    """测试 get_qa_prompt_with_parser() 函数"""

    def test_get_prompt_and_parser(self):
        """测试获取 Prompt 和解析器"""
        prompt, parser = get_qa_prompt_with_parser()

        assert isinstance(prompt, ChatPromptTemplate)
        assert isinstance(parser, PydanticOutputParser)
        assert parser.pydantic_object == TechnicalAnswer

        print("✅ Prompt + Parser 获取测试通过")

    def test_parser_format_instructions(self):
        """测试解析器格式化指令"""
        _, parser = get_qa_prompt_with_parser()

        format_instructions = parser.get_format_instructions()

        # TechnicalAnswer 的字段
        assert "problem_analysis" in format_instructions
        assert "solutions" in format_instructions
        assert "code_examples" in format_instructions
        assert "references" in format_instructions
        assert "confidence_score" in format_instructions
        assert "configuration_notes" in format_instructions or "warnings" in format_instructions

        print("✅ 解析器格式化指令测试通过")

    def test_custom_pydantic_model(self):
        """测试自定义 Pydantic 模型"""
        from pydantic import BaseModel

        class CustomModel(BaseModel):
            answer: str

        prompt, parser = get_qa_prompt_with_parser(pydantic_model=CustomModel)

        assert parser.pydantic_object == CustomModel
        format_instructions = parser.get_format_instructions()
        assert "answer" in format_instructions

        print("✅ 自定义 Pydantic 模型测试通过")


class TestValidatePromptVariables:
    """测试 validate_prompt_variables() 函数"""

    def test_valid_variables(self):
        """测试有效变量验证"""
        prompt = get_qa_prompt()
        result = validate_prompt_variables(prompt, ["context", "question", "format_instructions"])

        assert result is True
        print("✅ 有效变量验证测试通过")

    def test_missing_variables(self):
        """测试缺失变量验证"""
        prompt = get_qa_prompt()
        result = validate_prompt_variables(prompt, ["context", "question", "nonexistent"])

        assert result is False
        print("✅ 缺失变量验证测试通过")

    def test_subset_validation(self):
        """测试子集验证"""
        prompt = get_qa_prompt()
        result = validate_prompt_variables(prompt, ["context"])

        assert result is True
        print("✅ 子集验证测试通过")


class TestGetPromptStats:
    """测试 get_prompt_stats() 函数"""

    def test_basic_stats(self):
        """测试基本统计信息"""
        prompt = get_qa_prompt()
        stats = get_prompt_stats(prompt)

        assert "num_variables" in stats
        assert "variables" in stats
        assert "num_messages" in stats
        assert "estimated_tokens" in stats
        assert "template_type" in stats

        assert stats["num_variables"] == 3
        assert stats["num_messages"] == 2
        assert stats["template_type"] == "ChatPromptTemplate"

        print("✅ 基本统计信息测试通过")

    def test_variables_list(self):
        """测试变量列表"""
        prompt = get_qa_prompt()
        stats = get_prompt_stats(prompt)

        variables = stats["variables"]
        assert "context" in variables
        assert "question" in variables
        assert "format_instructions" in variables

        print("✅ 变量列表测试通过")

    def test_token_estimation(self):
        """测试 Token 估算"""
        prompt = get_qa_prompt()
        stats = get_prompt_stats(prompt)

        # Token 数应该是正数
        assert stats["estimated_tokens"] > 0
        print(f"✅ Token 估算测试通过 (估算: {stats['estimated_tokens']:.0f} tokens)")


class TestPromptConstants:
    """测试 Prompt 常量"""

    def test_qa_system_prompt_exists(self):
        """测试 QA 系统 Prompt 存在"""
        assert isinstance(QA_SYSTEM_PROMPT, str)
        assert len(QA_SYSTEM_PROMPT) > 0
        assert "魔搭社区" in QA_SYSTEM_PROMPT
        assert "{context}" in QA_SYSTEM_PROMPT
        assert "{format_instructions}" in QA_SYSTEM_PROMPT

        print("✅ QA 系统 Prompt 常量测试通过")

    def test_clarification_prompt_exists(self):
        """测试澄清 Prompt 存在"""
        assert isinstance(CLARIFICATION_PROMPT, str)
        assert "{question}" in CLARIFICATION_PROMPT

        print("✅ 澄清 Prompt 常量测试通过")

    def test_rerank_prompt_exists(self):
        """测试重排序 Prompt 存在"""
        assert isinstance(RERANK_PROMPT, str)
        assert "{question}" in RERANK_PROMPT
        assert "{document}" in RERANK_PROMPT

        print("✅ 重排序 Prompt 常量测试通过")

    def test_validation_prompt_exists(self):
        """测试验证 Prompt 存在"""
        assert isinstance(ANSWER_VALIDATION_PROMPT, str)
        assert "{question}" in ANSWER_VALIDATION_PROMPT
        assert "{answer}" in ANSWER_VALIDATION_PROMPT
        assert "{context}" in ANSWER_VALIDATION_PROMPT

        print("✅ 验证 Prompt 常量测试通过")


class TestPromptIntegration:
    """测试 Prompt 集成场景"""

    def test_format_with_real_example(self):
        """测试使用真实示例格式化 Prompt"""
        prompt = get_qa_prompt()

        context = """
        Qwen 是阿里云开发的大语言模型系列。支持多轮对话、代码生成、数学推理等任务。

        使用示例:
        ```python
        from modelscope import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat")
        tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat")
        ```
        """

        question = "如何使用 Qwen 模型进行对话?"

        formatted = prompt.format(
            context=context,
            question=question,
            format_instructions="JSON 格式"
        )

        assert "Qwen" in formatted
        assert "from modelscope import" in formatted
        assert "如何使用 Qwen 模型进行对话?" in formatted

        print("✅ 真实示例格式化测试通过")

    def test_prompt_chain_compatibility(self):
        """测试 Prompt 与 LangChain 链的兼容性"""
        from unittest.mock import Mock

        prompt, parser = get_qa_prompt_with_parser()

        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=Mock(content='{"summary": "测试", "problem_analysis": "分析", "solutions": ["方案"], "code_examples": [], "references": [], "confidence_score": 0.8}'))

        # 构建链
        chain = prompt | mock_llm

        # 测试调用
        result = chain.invoke({
            "context": "测试上下文",
            "question": "测试问题",
            "format_instructions": parser.get_format_instructions()
        })

        assert result is not None
        print("✅ Prompt 链兼容性测试通过")

    def test_empty_context_handling(self):
        """测试空上下文处理"""
        prompt = get_qa_prompt()

        formatted = prompt.format(
            context="",
            question="测试问题",
            format_instructions="格式说明"
        )

        # 应该仍然可以格式化,只是上下文为空
        assert "测试问题" in formatted
        assert formatted is not None

        print("✅ 空上下文处理测试通过")

    def test_long_context_handling(self):
        """测试长上下文处理"""
        prompt = get_qa_prompt()

        # 创建一个很长的上下文
        long_context = "这是一个很长的文档。" * 1000

        formatted = prompt.format(
            context=long_context,
            question="测试问题",
            format_instructions="格式说明"
        )

        # 验证可以处理长文本
        assert len(formatted) > len(long_context)
        assert "测试问题" in formatted

        print(f"✅ 长上下文处理测试通过 (长度: {len(formatted)} 字符)")

    def test_special_characters_handling(self):
        """测试特殊字符处理"""
        prompt = get_qa_prompt()

        special_context = """
        特殊字符测试: {}[]()'"<>&
        中文标点: 《》【】""''、。！？
        代码符号: $@#%^*+=|\\`~
        """

        formatted = prompt.format(
            context=special_context,
            question="特殊字符问题?",
            format_instructions="格式 {test}"
        )

        # 应该正确处理所有特殊字符
        assert "特殊字符测试" in formatted
        assert "中文标点" in formatted

        print("✅ 特殊字符处理测试通过")


class TestPromptQuality:
    """测试 Prompt 质量"""

    def test_system_prompt_completeness(self):
        """测试系统 Prompt 完整性"""
        # 检查关键元素
        essential_elements = [
            "魔搭社区",
            "技术支持",
            "角色",
            "任务",
            "要求",
            "准确性",
            "完整性",
            "实用性",
            "上下文文档",
            "输出格式"
        ]

        for element in essential_elements:
            assert element in QA_SYSTEM_PROMPT, f"缺少关键元素: {element}"

        print("✅ 系统 Prompt 完整性测试通过")

    def test_prompt_clarity(self):
        """测试 Prompt 清晰度"""
        # Prompt 应该包含明确的指令
        assert "必须基于文档内容" in QA_SYSTEM_PROMPT
        assert "不得编造" in QA_SYSTEM_PROMPT
        assert "至少 1 种" in QA_SYSTEM_PROMPT or "至少1种" in QA_SYSTEM_PROMPT

        print("✅ Prompt 清晰度测试通过")

    def test_output_format_specification(self):
        """测试输出格式规范"""
        prompt = get_qa_prompt()
        formatted = prompt.format(
            context="测试",
            question="测试",
            format_instructions="测试格式"
        )

        # 应该包含格式说明
        assert "JSON" in formatted or "json" in formatted.lower()

        print("✅ 输出格式规范测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
