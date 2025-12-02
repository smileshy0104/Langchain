"""
Agent Prompt Templates
Agent 使用的提示词模板
"""

# 系统提示词:定义 Agent 角色
SYSTEM_PROMPT = """你是魔搭社区(ModelScope)的智能技术助手。

你的职责是:
1. 准确回答用户关于魔搭社区平台、模型、API的技术问题
2. 基于提供的文档上下文给出答案,不编造信息
3. 当信息不足时,主动询问澄清问题
4. 提供清晰、结构化的答案,包含代码示例和文档链接

回答要求:
- 使用中文回答
- 引用具体文档来源
- 代码示例需要完整可运行
- 不确定时明确告知用户
"""

# 答案生成提示词模板
ANSWER_GENERATION_PROMPT = """基于以下检索到的文档,回答用户问题。

用户问题:
{question}

检索文档:
{context}

要求:
1. 仅基于提供的文档内容回答
2. 包含具体代码示例(如果适用)
3. 引用文档来源
4. 如果文档不足以回答问题,明确说明

请给出答案:
"""

# 澄清问题生成提示词模板
CLARIFICATION_PROMPT = """用户的问题不够明确,需要进一步澄清。

用户问题:
{question}

当前理解:
{current_understanding}

请生成2-3个澄清问题,帮助理解用户真正的需求:
"""

# 置信度评估提示词模板
CONFIDENCE_EVALUATION_PROMPT = """评估以下检索结果对回答用户问题的充分性。

用户问题:
{question}

检索到的文档:
{retrieved_docs}

请评估:
1. 文档是否包含足够信息回答问题
2. 给出置信度分数(0.0-1.0)
3. 如果置信度低,说明缺少哪些信息

输出JSON格式:
{{
    "confidence": 0.0-1.0,
    "is_sufficient": true/false,
    "missing_info": "..."
}}
"""

# 多轮对话上下文总结提示词
CONVERSATION_SUMMARY_PROMPT = """总结以下对话历史的关键信息。

对话历史:
{conversation_history}

请提取:
1. 用户的核心问题
2. 已经澄清的信息
3. 仍需解答的部分

总结:
"""
