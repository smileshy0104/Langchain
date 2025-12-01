"""
魔搭社区问答 Agent

实现基于 LangGraph 的技术问答智能代理,包含文档检索、答案生成和验证节点。

核心功能:
- 文档检索: 使用混合检索器获取相关文档
- 答案生成: 基于 LLM 生成结构化技术回答
- 答案验证: Self-RAG 验证答案质量(可选)
- 对话持久化: MemorySaver 支持多轮对话
"""

from typing import Optional, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from models.schemas import ConversationState, TechnicalAnswer
from retrievers.hybrid_retriever import HybridRetriever
from tools.clarification_tool import ClarificationTool
from core.memory_manager import MemoryManager


class ModelScopeQAAgent:
    """魔搭社区问答 Agent

    基于 LangGraph 实现的技术问答代理,支持:
    - RAG (Retrieval-Augmented Generation) 架构
    - 多轮对话状态管理
    - 答案质量验证
    - 检查点持久化

    Attributes:
        retriever: HybridRetriever 混合检索器实例
        llm: ChatTongyi LLM 客户端
        clarification_tool: ClarificationTool 澄清问题工具
        workflow: StateGraph LangGraph 工作流
        checkpointer: MemorySaver 检查点持久化器
        app: CompiledGraph 编译后的工作流应用

    Example:
        >>> from core.vector_store import VectorStoreManager
        >>> from retrievers.hybrid_retriever import HybridRetriever
        >>>
        >>> # 初始化检索器
        >>> manager = VectorStoreManager()
        >>> vector_store = manager.get_vector_store()
        >>> retriever = HybridRetriever(vector_store, documents)
        >>>
        >>> # 创建 Agent
        >>> agent = ModelScopeQAAgent(retriever, api_key="your-api-key")
        >>>
        >>> # 单轮问答
        >>> answer = agent.invoke("如何使用 Qwen 模型?")
        >>> print(answer["summary"])
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_api_key: str,
        model: str = "qwen-plus",
        temperature: float = 0.3,
        top_p: float = 0.8
    ):
        """初始化魔搭社区问答 Agent

        Args:
            retriever: HybridRetriever 实例,用于文档检索
            llm_api_key: 通义千问 API 密钥
            model: 模型名称(默认 qwen-plus)
            temperature: 温度参数,控制随机性(默认 0.3)
            top_p: Top-p 采样参数(默认 0.8)

        Raises:
            ValueError: 如果 retriever 为 None 或 API 密钥为空

        Example:
            >>> agent = ModelScopeQAAgent(
            ...     retriever=my_retriever,
            ...     llm_api_key="sk-xxx",
            ...     temperature=0.2
            ... )
        """
        if retriever is None:
            raise ValueError("retriever 不能为 None")
        if not llm_api_key or not llm_api_key.strip():
            raise ValueError("llm_api_key 不能为空")

        self.retriever = retriever

        # 存储配置参数(因为 ChatTongyi 不暴露这些属性)
        self._model = model
        self._temperature = temperature
        self._top_p = top_p

        # 初始化通义千问 LLM
        self.llm = ChatTongyi(
            model=model,
            temperature=temperature,
            top_p=top_p,
            dashscope_api_key=llm_api_key
        )

        # 初始化澄清工具 (Phase 3.6: 主动澄清机制)
        self.clarification_tool = ClarificationTool(
            llm_api_key=llm_api_key,
            model=model,
            temperature=temperature
        )

        # 初始化对话记忆管理器 (Phase 4.1: 对话历史管理)
        self.memory_manager = MemoryManager(
            llm=self.llm,
            max_turns=10,  # 保留最近10轮对话
            max_tokens=4000
        )

        # 构建 LangGraph 工作流
        self.workflow = StateGraph(ConversationState)
        self._build_graph()

        # 添加检查点器支持对话持久化
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

        print(f"✅ ModelScopeQAAgent 初始化成功")
        print(f"   - LLM 模型: {model}")
        print(f"   - 温度: {temperature}")
        print(f"   - Top-P: {top_p}")
        print(f"   - 检索器: {type(retriever).__name__}")

    def _build_graph(self):
        """构建 LangGraph 工作流

        工作流节点:
        1. clarify: 澄清问题节点 (Phase 3.6)
        2. retrieve: 文档检索节点
        3. generate: 答案生成节点
        4. validate: 答案验证节点(可选)

        工作流:
        START → clarify → [条件分支]
                           ├─> END (需要澄清, 返回澄清问题)
                           └─> retrieve → generate → [条件分支]
                                                       ├─> validate → END (置信度 < 0.8)
                                                       └─> END (置信度 ≥ 0.8)
        """
        # 添加节点
        self.workflow.add_node("clarify", self._clarify_question)
        self.workflow.add_node("retrieve", self._retrieve_documents)
        self.workflow.add_node("generate", self._generate_answer)
        self.workflow.add_node("validate", self._validate_answer)

        # 设置入口点: 从澄清节点开始
        self.workflow.set_entry_point("clarify")

        # 条件分支1: 澄清后决定是继续还是返回澄清问题
        self.workflow.add_conditional_edges(
            "clarify",
            self._should_retrieve_or_clarify,
            {
                "retrieve": "retrieve",  # 不需要澄清,继续检索
                "end": END  # 需要澄清,返回澄清问题
            }
        )

        # 添加边: retrieve → generate
        self.workflow.add_edge("retrieve", "generate")

        # 条件分支2: 根据置信度决定是否验证
        self.workflow.add_conditional_edges(
            "generate",
            self._should_validate,
            {
                "validate": "validate",
                "end": END
            }
        )

        self.workflow.add_edge("validate", END)

        print("✅ LangGraph 工作流构建完成")
        print("   节点: clarify → [retrieve → generate → validate]")

    def _clarify_question(self, state: ConversationState) -> ConversationState:
        """澄清问题节点 (Phase 3.6: 主动澄清机制)

        检测用户问题是否缺失关键信息,如果需要则生成澄清问题。

        Args:
            state: 当前对话状态

        Returns:
            ConversationState: 更新后的状态,包含澄清检查结果

        Updates:
            - needs_clarification: 是否需要澄清
            - clarification_questions: 澄清问题列表
        """
        # 获取用户问题
        question = state["messages"][-1].content

        # 使用澄清工具检查问题
        try:
            clarification_result = self.clarification_tool.check_and_clarify(question)

            # 更新状态
            state["needs_clarification"] = clarification_result.needs_clarification
            state["clarification_questions"] = clarification_result.clarification_questions

            if clarification_result.needs_clarification:
                print(f"❓ 需要澄清, 生成了 {len(clarification_result.clarification_questions)} 个问题")
            else:
                print(f"✅ 问题信息充分, 无需澄清")

        except Exception as e:
            print(f"⚠️  澄清检查失败: {e}")
            # 降级: 假设不需要澄清,继续处理
            state["needs_clarification"] = False
            state["clarification_questions"] = []

        return state

    def _should_retrieve_or_clarify(self, state: ConversationState) -> str:
        """条件分支: 判断是继续检索还是返回澄清问题

        Args:
            state: 当前对话状态

        Returns:
            str: "retrieve" 或 "end"

        逻辑:
            - 如果需要澄清: 返回 "end" (结束流程,返回澄清问题)
            - 如果不需要澄清: 返回 "retrieve" (继续检索流程)
        """
        if state["needs_clarification"]:
            print(f"🔀 需要澄清, 终止检索流程")
            return "end"
        else:
            print(f"🔀 无需澄清, 继续检索流程")
            return "retrieve"

    def _retrieve_documents(self, state: ConversationState) -> ConversationState:
        """检索相关文档节点

        从用户消息中提取问题,使用混合检索器获取相关文档。

        Args:
            state: 当前对话状态

        Returns:
            ConversationState: 更新后的状态,包含检索到的文档

        Updates:
            - current_question: 当前用户问题
            - retrieved_documents: 检索到的相关文档列表
        """
        # 获取最后一条消息作为问题
        question = state["messages"][-1].content

        # 执行混合检索
        try:
            docs = self.retriever.retrieve(question, k=3)
            print(f"📥 检索到 {len(docs)} 个相关文档")
        except Exception as e:
            print(f"⚠️  检索失败: {e}")
            docs = []

        # 更新状态
        state["current_question"] = question
        state["retrieved_documents"] = docs

        return state

    def _generate_answer(self, state: ConversationState) -> ConversationState:
        """生成技术回答节点

        基于检索到的文档、用户问题和对话历史,使用 LLM 生成结构化技术回答。

        Args:
            state: 当前对话状态

        Returns:
            ConversationState: 更新后的状态,包含生成的答案

        Updates:
            - generated_answer: 生成的 TechnicalAnswer 字典

        Prompt 结构:
            - System: 定义角色、任务和输出格式
            - Conversation History: 对话历史摘要（如果有）
            - Context: 检索到的文档内容
            - Human: 当前用户问题
        """
        # 构建文档上下文
        context = "\n\n".join([
            f"文档 {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(state["retrieved_documents"])
        ])

        if not context.strip():
            context = "未检索到相关文档。"

        # 构建对话历史上下文（T107: 支持对话历史引用）
        conversation_history = self._build_conversation_history(state)

        # 系统提示词（T108: 添加对话历史占位符）
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是魔搭社区的技术支持专家。

**任务**: 基于提供的文档上下文和对话历史,回答用户的技术问题。

**要求**:
1. 回答必须基于文档内容,不得编造
2. 如果用户问题引用了之前的对话内容（如"刚才你建议的方法"、"你刚说的"、"之前提到的"),要准确理解代词指向,结合对话历史给出回答
3. 提供至少1种可执行的解决方案
4. 包含完整的代码示例（如果适用）
5. 引用信息来源
6. 如果文档不足以回答问题,明确说明

{conversation_history_section}

**上下文文档**:
{context}

**输出格式**: 请使用以下 JSON 格式:
{format_instructions}
"""),
            ("human", "{question}")
        ])

        # 配置 Pydantic 输出解析器
        parser = PydanticOutputParser(pydantic_object=TechnicalAnswer)

        # 构建生成链
        chain = prompt | self.llm | parser

        try:
            # 生成答案（包含对话历史）
            answer = chain.invoke({
                "context": context,
                "conversation_history_section": conversation_history,
                "question": state["current_question"],
                "format_instructions": parser.get_format_instructions()
            })

            print(f"✅ 答案生成成功")
            print(f"   - 解决方案数: {len(answer.solutions)}")
            print(f"   - 置信度: {answer.confidence_score}")

            # 更新状态
            state["generated_answer"] = answer.model_dump()

        except Exception as e:
            print(f"⚠️  答案生成失败: {e}")
            # 降级: 返回基本回答
            fallback_answer = TechnicalAnswer(
                summary=f"抱歉,生成答案时出现错误: {str(e)}",
                problem_analysis="答案生成失败",
                solutions=["请稍后重试或联系技术支持"],
                code_examples=[],
                references=[],
                confidence_score=0.0
            )
            state["generated_answer"] = fallback_answer.model_dump()

        return state

    def _validate_answer(self, state: ConversationState) -> ConversationState:
        """验证答案质量节点

        对生成的答案进行质量验证(Self-RAG)。
        当置信度较低时,可以重新检索或优化答案。

        Args:
            state: 当前对话状态

        Returns:
            ConversationState: 更新后的状态

        TODO: 实现 Self-RAG 验证逻辑
            - 检查答案与文档的相关性
            - 验证引用来源的准确性
            - 评估解决方案的可行性
        """
        # TODO: 实现答案验证逻辑
        # 目前仅作为占位节点
        print("🔍 执行答案验证...")

        # 未来可以添加:
        # 1. 相关性评分
        # 2. 引用验证
        # 3. 代码可执行性检查
        # 4. 如果验证失败,触发重新检索

        return state

    def _should_validate(self, state: ConversationState) -> str:
        """条件分支: 判断是否需要验证答案

        Args:
            state: 当前对话状态

        Returns:
            str: "validate" 或 "end"

        逻辑:
            - 置信度 < 0.8: 需要验证
            - 置信度 ≥ 0.8: 直接结束
        """
        confidence = state["generated_answer"].get("confidence_score", 0.0)

        if confidence < 0.8:
            print(f"⚠️  置信度较低 ({confidence:.2f}), 执行验证")
            return "validate"
        else:
            print(f"✅ 置信度较高 ({confidence:.2f}), 直接返回")
            return "end"

    def _build_conversation_history(self, state: ConversationState) -> str:
        """构建对话历史上下文（T107: 支持对话历史引用）

        使用 MemoryManager 获取优化的对话窗口，包含早期对话摘要和最近对话。

        Args:
            state: 当前对话状态

        Returns:
            str: 格式化的对话历史文本，如果是首轮对话则返回空字符串

        处理流程:
            1. 检查是否有对话历史（消息数 > 2: System + 当前问题）
            2. 如果需要摘要，生成早期对话摘要
            3. 使用 MemoryManager 获取优化的对话窗口
            4. 格式化为可读文本

        Example Output:
            **对话历史**:
            早期对话摘要: 用户询问了模型加载问题...

            用户: 如何加载 Qwen-7B 模型?
            Agent: 使用 AutoModelForCausalLM.from_pretrained()...
            用户: CUDA 内存不足怎么办?
            Agent: 可以降低 batch_size...
        """
        messages = state.get("messages", [])

        # 如果只有当前问题（没有历史对话），返回空
        # 通常第一轮对话会有: SystemMessage (if any) + HumanMessage
        if len(messages) <= 2:
            return ""

        # 检查是否需要生成摘要
        conversation_summary = state.get("conversation_summary")
        if self.memory_manager.should_generate_summary(messages):
            # 获取需要摘要的早期消息
            early_messages = self.memory_manager.get_early_messages(messages)
            if early_messages:
                # 生成或更新摘要
                conversation_summary = self.memory_manager.summarize_early_messages(
                    early_messages,
                    current_summary=conversation_summary
                )
                # 更新状态中的摘要（供下次使用）
                state["conversation_summary"] = conversation_summary

        # 获取优化的对话窗口（包含摘要 + 最近对话）
        conversation_window = self.memory_manager.get_conversation_window(
            messages[:-1],  # 排除当前问题（最后一条消息）
            summary=conversation_summary
        )

        # 格式化对话历史
        if not conversation_window:
            return ""

        history_lines = ["**对话历史**:"]

        for msg in conversation_window:
            if hasattr(msg, '__class__') and msg.__class__.__name__ == 'SystemMessage':
                # 系统消息（包括摘要）
                if "早期对话摘要" in msg.content:
                    history_lines.append(msg.content)
                # 跳过其他系统消息（如初始的 System Prompt）
            elif hasattr(msg, '__class__') and msg.__class__.__name__ == 'HumanMessage':
                history_lines.append(f"用户: {msg.content}")
            elif hasattr(msg, '__class__') and msg.__class__.__name__ == 'AIMessage':
                # 简化 AI 回答（只显示摘要，不显示完整的结构化输出）
                content = msg.content
                if len(content) > 200:
                    content = content[:200] + "..."
                history_lines.append(f"Agent: {content}")

        return "\n".join(history_lines) if len(history_lines) > 1 else ""

    def invoke(
        self,
        question: str,
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        """调用 Agent 进行单轮问答

        Args:
            question: 用户问题
            thread_id: 线程ID,用于多轮对话管理(默认 "default")

        Returns:
            Dict[str, Any]: TechnicalAnswer 字典,包含:
                - summary: 答案摘要
                - problem_analysis: 问题分析
                - solutions: 解决方案列表
                - code_examples: 代码示例
                - references: 引用来源
                - confidence_score: 置信度评分

        Raises:
            Exception: 工作流执行失败时抛出异常

        Example:
            >>> answer = agent.invoke("如何使用 Qwen 模型?")
            >>> print(answer["summary"])
            >>> for solution in answer["solutions"]:
            ...     print(f"- {solution}")
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        print(f"\n{'='*70}")
        print(f"🤖 ModelScopeQAAgent 处理问题")
        print(f"{'='*70}")
        print(f"问题: {question}")
        print(f"线程ID: {thread_id}")
        print(f"{'='*70}\n")

        try:
            # 调用工作流
            result = self.app.invoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "current_question": "",
                    "retrieved_documents": [],
                    "generated_answer": {},
                    "needs_clarification": False,  # Phase 3.6: 澄清标记
                    "clarification_questions": [],  # Phase 3.6: 澄清问题列表
                    "turn_count": 0
                },
                config={"configurable": {"thread_id": thread_id}}
            )

            print(f"\n{'='*70}")
            print(f"✅ 处理完成")
            print(f"{'='*70}\n")

            # 如果需要澄清,返回澄清问题而不是答案
            if result["needs_clarification"]:
                print(f"❓ 需要用户澄清信息")
                return {
                    "needs_clarification": True,
                    "clarification_questions": result["clarification_questions"],
                    "summary": "为了更好地帮助您,我需要了解以下信息:",
                    "problem_analysis": "问题描述不够清晰",
                    "solutions": result["clarification_questions"],
                    "code_examples": [],
                    "references": [],
                    "confidence_score": 0.0
                }
            else:
                return result["generated_answer"]

        except Exception as e:
            print(f"\n⚠️  工作流执行失败: {e}")
            raise

    def get_state(self, thread_id: str = "default") -> Optional[ConversationState]:
        """获取指定线程的对话状态

        Args:
            thread_id: 线程ID

        Returns:
            Optional[ConversationState]: 对话状态,如果不存在返回 None

        Example:
            >>> state = agent.get_state("user123")
            >>> if state:
            ...     print(f"历史消息数: {len(state['messages'])}")
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = self.app.get_state(config)
            return snapshot.values if snapshot else None
        except Exception as e:
            print(f"⚠️  获取状态失败: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """获取 Agent 统计信息

        Returns:
            Dict[str, Any]: 统计信息字典

        Example:
            >>> stats = agent.get_stats()
            >>> print(f"检索器: {stats['retriever_type']}")
            >>> print(f"LLM: {stats['llm_model']}")
        """
        return {
            "retriever_type": type(self.retriever).__name__,
            "retriever_stats": self.retriever.get_stats(),
            "llm_model": self._model,
            "llm_temperature": self._temperature,
            "llm_top_p": self._top_p,
            "has_checkpointer": self.checkpointer is not None,
            "workflow_nodes": ["retrieve", "generate", "validate"]
        }
