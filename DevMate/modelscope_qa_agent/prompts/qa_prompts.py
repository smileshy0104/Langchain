"""
魔搭社区问答 Prompt 模板

提供预定义的 Prompt 模板,用于技术问答、问题分类、答案验证等场景。

核心模板:
- QA_SYSTEM_PROMPT: 技术问答系统 Prompt
- QA_PROMPT_TEMPLATE: 完整的问答 Prompt 模板
- get_qa_prompt(): 获取配置好的 Prompt 模板
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from models.schemas import TechnicalAnswer


# ============================================================================
# 系统 Prompt 定义
# ============================================================================

QA_SYSTEM_PROMPT = """你是魔搭社区的技术支持专家,专门帮助开发者解决 AI 模型使用、平台功能和技术实现问题。

**你的角色**:
- 技术顾问: 提供专业、准确的技术指导
- 问题解决者: 快速定位问题并给出可行方案
- 知识传播者: 分享最佳实践和使用技巧

**你的能力**:
- 深入理解魔搭社区的模型库、数据集和工具
- 熟悉主流 AI 框架(PyTorch、TensorFlow、HuggingFace 等)
- 精通 Python 编程和常见开发工具链
- 了解云端部署、模型优化和性能调优

**任务**: 基于提供的文档上下文,回答用户的技术问题。

**回答要求**:
1. **准确性**: 回答必须基于文档内容,不得编造或臆测
2. **完整性**: 提供至少 1 种可执行的解决方案,包含完整步骤
3. **实用性**: 包含完整的代码示例（如果适用）,确保可运行
4. **可追溯**: 明确引用信息来源,方便用户查阅原文
5. **清晰性**: 使用简洁明了的语言,避免过度技术术语
6. **诚实性**: 如果文档不足以回答问题,明确说明并建议其他途径

**回答结构**:
1. 问题分析: 简要分析问题的本质和可能原因
2. 解决方案: 提供 1-3 种解决方案,按推荐度排序
3. 代码示例: 提供完整可运行的示例代码
4. 注意事项: 说明常见陷阱和最佳实践
5. 参考资料: 列出相关文档链接

**上下文文档**:
{context}

**输出格式**: 请严格按照以下 JSON 格式输出:
{format_instructions}
"""


# ============================================================================
# 上下文占位符说明
# ============================================================================

CONTEXT_PLACEHOLDER_DOC = """
上下文占位符说明:
- {context}: 检索到的文档内容,格式为多个文档拼接
- {question}: 用户的原始问题
- {format_instructions}: Pydantic 输出格式说明
"""


# ============================================================================
# Prompt 模板函数
# ============================================================================

def get_qa_prompt(
    include_format_instructions: bool = True,
    custom_system_prompt: str = None
) -> ChatPromptTemplate:
    """获取技术问答 Prompt 模板

    Args:
        include_format_instructions: 是否包含格式化指令(默认 True)
        custom_system_prompt: 自定义系统 Prompt(可选)

    Returns:
        ChatPromptTemplate: 配置好的 Prompt 模板

    Example:
        >>> prompt = get_qa_prompt()
        >>> parser = PydanticOutputParser(pydantic_object=TechnicalAnswer)
        >>> chain = prompt | llm | parser
        >>> result = chain.invoke({
        ...     "context": "文档内容...",
        ...     "question": "如何使用 Qwen?",
        ...     "format_instructions": parser.get_format_instructions()
        ... })
    """
    system_prompt = custom_system_prompt or QA_SYSTEM_PROMPT

    if include_format_instructions:
        # 包含格式化指令的完整模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
    else:
        # 不包含格式化指令的简化模板
        simplified_prompt = system_prompt.replace(
            "\n**输出格式**: 请严格按照以下 JSON 格式输出:\n{format_instructions}",
            ""
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", simplified_prompt),
            ("human", "{question}")
        ])

    return prompt


def get_qa_prompt_with_parser(
    pydantic_model=TechnicalAnswer
) -> tuple[ChatPromptTemplate, PydanticOutputParser]:
    """获取 Prompt 模板和配置好的解析器

    Args:
        pydantic_model: Pydantic 模型类(默认 TechnicalAnswer)

    Returns:
        tuple: (ChatPromptTemplate, PydanticOutputParser)

    Example:
        >>> prompt, parser = get_qa_prompt_with_parser()
        >>> chain = prompt | llm | parser
        >>> result = chain.invoke({
        ...     "context": context,
        ...     "question": question,
        ...     "format_instructions": parser.get_format_instructions()
        ... })
    """
    prompt = get_qa_prompt(include_format_instructions=True)
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    return prompt, parser


# ============================================================================
# 辅助 Prompt 模板
# ============================================================================

CLARIFICATION_PROMPT = """基于以下用户问题,判断是否需要澄清。

用户问题: {question}

如果问题缺少关键信息(如版本、环境、具体错误等),生成澄清问题列表。
如果问题足够清晰,返回空列表。

输出格式:
{{
    "needs_clarification": true/false,
    "clarification_questions": ["问题1", "问题2", ...]
}}
"""


RERANK_PROMPT = """评估以下文档与用户问题的相关性。

用户问题: {question}

文档内容:
{document}

相关性评分 (0-1):
- 1.0: 直接回答问题
- 0.7-0.9: 高度相关,包含有用信息
- 0.4-0.6: 部分相关,需要结合其他信息
- 0.1-0.3: 弱相关,可能有帮助
- 0.0: 完全不相关

输出格式:
{{
    "relevance_score": 0.85,
    "reason": "文档直接提供了问题的解决方案..."
}}
"""


ANSWER_VALIDATION_PROMPT = """验证生成的答案质量。

用户问题: {question}

生成的答案:
{answer}

参考文档:
{context}

检查项:
1. 答案是否基于文档内容（忠实性）
2. 答案是否完整回答问题（完整性）
3. 代码示例是否正确（准确性）
4. 引用来源是否准确（可追溯性）

输出格式:
{{
    "is_valid": true/false,
    "faithfulness_score": 0.9,
    "completeness_score": 0.85,
    "accuracy_score": 0.95,
    "issues": ["问题1", "问题2"],
    "suggestions": ["改进建议1", "改进建议2"]
}}
"""


# ============================================================================
# Prompt 验证和测试辅助函数
# ============================================================================

def validate_prompt_variables(prompt: ChatPromptTemplate, required_vars: list[str]) -> bool:
    """验证 Prompt 模板包含所有必需变量

    Args:
        prompt: ChatPromptTemplate 实例
        required_vars: 必需的变量列表

    Returns:
        bool: True 如果所有变量都存在

    Example:
        >>> prompt = get_qa_prompt()
        >>> is_valid = validate_prompt_variables(prompt, ["context", "question"])
        >>> assert is_valid == True
    """
    prompt_vars = prompt.input_variables
    missing_vars = [var for var in required_vars if var not in prompt_vars]

    if missing_vars:
        print(f"⚠️  缺少必需变量: {missing_vars}")
        return False

    print(f"✅ Prompt 变量验证通过: {prompt_vars}")
    return True


def get_prompt_stats(prompt: ChatPromptTemplate) -> dict:
    """获取 Prompt 模板统计信息

    Args:
        prompt: ChatPromptTemplate 实例

    Returns:
        dict: 统计信息字典

    Example:
        >>> prompt = get_qa_prompt()
        >>> stats = get_prompt_stats(prompt)
        >>> print(f"变量数: {stats['num_variables']}")
    """
    # 格式化一个示例来估算 token 数
    example_format = prompt.format(
        context="示例文档" * 100,
        question="示例问题",
        format_instructions="示例格式说明" * 10
    )

    return {
        "num_variables": len(prompt.input_variables),
        "variables": prompt.input_variables,
        "num_messages": len(prompt.messages),
        "estimated_tokens": len(example_format.split()) * 1.3,  # 粗略估算
        "template_type": type(prompt).__name__
    }


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    # 示例 1: 基本用法
    print("=" * 70)
    print("示例 1: 获取基本 Prompt 模板")
    print("=" * 70)
    prompt = get_qa_prompt()
    print(f"变量: {prompt.input_variables}")
    print(f"消息数: {len(prompt.messages)}")

    # 示例 2: 带解析器
    print("\n" + "=" * 70)
    print("示例 2: 获取 Prompt + Parser")
    print("=" * 70)
    prompt, parser = get_qa_prompt_with_parser()
    print(f"Prompt 变量: {prompt.input_variables}")
    print(f"Parser 类型: {type(parser).__name__}")

    # 示例 3: 验证变量
    print("\n" + "=" * 70)
    print("示例 3: 验证 Prompt 变量")
    print("=" * 70)
    is_valid = validate_prompt_variables(prompt, ["context", "question", "format_instructions"])
    print(f"验证结果: {is_valid}")

    # 示例 4: 统计信息
    print("\n" + "=" * 70)
    print("示例 4: Prompt 统计信息")
    print("=" * 70)
    stats = get_prompt_stats(prompt)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ 所有示例执行完成")
