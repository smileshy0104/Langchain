"""
LangChain Messages - 最佳实践示例
演示消息使用的最佳实践、常见模式和注意事项
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import List, Optional
import os

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 清晰的系统提示 ====================

def clear_system_prompts():
    """清晰的系统提示最佳实践"""
    print("=" * 60)
    print("清晰的系统提示最佳实践")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    # ❌ 不好的系统提示
    bad_prompt = SystemMessage(content="帮我")

    # ✅ 好的系统提示
    good_prompt = SystemMessage(
        content="""你是一个专业的 Python 编程助手。

角色定位:
- 提供准确的 Python 代码示例
- 解释概念时简洁清晰
- 遵循 PEP 8 编码规范

回答风格:
- 使用中文回答
- 提供可运行的代码示例
- 指出常见错误和最佳实践

限制:
- 不讨论其他编程语言
- 不提供完整项目代码
"""
    )

    print("\n❌ 不好的系统提示:")
    print(f"   '{bad_prompt.content}'")
    print("   问题: 过于模糊,没有明确角色和行为")

    print("\n✅ 好的系统提示:")
    print(f"   {good_prompt.content}")
    print("\n   优点:")
    print("   - 明确角色定位")
    print("   - 规定回答风格")
    print("   - 设置明确限制")


# ==================== 2. 结构化消息内容 ====================

def structured_message_content():
    """结构化消息内容最佳实践"""
    print("\n" + "=" * 60)
    print("结构化消息内容最佳实践")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6")

    # ❌ 不好的提问方式
    bad_message = HumanMessage(
        content="我想要一个函数处理数据"
    )

    # ✅ 好的提问方式
    good_message = HumanMessage(
        content="""请帮我编写一个 Python 函数:

功能: 处理用户数据
输入: 包含 name, age, email 的字典列表
输出: 过滤掉 age < 18 的用户
要求: 使用列表推导式

示例输入:
[{"name": "张三", "age": 25, "email": "zhang@example.com"},
 {"name": "李四", "age": 16, "email": "li@example.com"}]
"""
    )

    print("\n❌ 不好的提问:")
    print(f"   {bad_message.content}")
    print("   问题: 过于模糊,缺少关键信息")

    print("\n✅ 好的提问:")
    print(f"   {good_message.content}")
    print("\n   优点:")
    print("   - 明确功能需求")
    print("   - 指定输入输出格式")
    print("   - 提供具体示例")


# ==================== 3. 合理使用消息类型 ====================

def proper_message_types():
    """合理使用消息类型最佳实践"""
    print("\n" + "=" * 60)
    print("合理使用消息类型最佳实践")
    print("=" * 60)

    print("\n消息类型选择指南:")

    print("\n✅ SystemMessage - 用于:")
    print("   - 设置 AI 角色和行为")
    print("   - 定义对话规则")
    print("   - 提供背景信息")

    print("\n✅ HumanMessage - 用于:")
    print("   - 用户的问题和请求")
    print("   - 提供输入数据")
    print("   - 给出反馈")

    print("\n✅ AIMessage - 用于:")
    print("   - AI 的回复和输出")
    print("   - 工具调用决策")
    print("   - 对话历史记录")

    print("\n✅ ToolMessage - 用于:")
    print("   - 工具执行结果")
    print("   - 必须关联 tool_call_id")

    # 示例:正确的消息组合
    correct_usage = [
        SystemMessage(content="你是Python助手"),
        HumanMessage(content="如何读取文件？"),
        AIMessage(content="使用 open() 函数..."),
        HumanMessage(content="能举个例子吗？"),
        AIMessage(content="当然可以:...")
    ]

    print("\n正确的消息流:")
    for i, msg in enumerate(correct_usage, 1):
        print(f"   {i}. {type(msg).__name__}: {msg.content[:30]}...")


# ==================== 4. 上下文管理 ====================

def context_management():
    """上下文管理最佳实践"""
    print("\n" + "=" * 60)
    print("上下文管理最佳实践")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6")

    class SmartContextManager:
        """智能上下文管理器"""

        def __init__(self, max_context_messages: int = 10):
            self.history = InMemoryChatMessageHistory()
            self.max_messages = max_context_messages
            self.system_message = None

        def add_system_message(self, content: str):
            """添加系统消息"""
            self.system_message = SystemMessage(content=content)

        def add_turn(self, user_msg: str, ai_msg: str):
            """添加一轮对话"""
            self.history.add_user_message(user_msg)
            self.history.add_ai_message(ai_msg)
            self._manage_context()

        def _manage_context(self):
            """管理上下文长度"""
            messages = self.history.messages

            if len(messages) > self.max_messages:
                # 保留最近的消息
                kept = messages[-self.max_messages:]
                self.history.clear()
                for msg in kept:
                    self.history.add_message(msg)

        def get_messages_for_llm(self) -> List[BaseMessage]:
            """获取用于 LLM 的消息"""
            result = []
            if self.system_message:
                result.append(self.system_message)
            result.extend(self.history.messages)
            return result

    # 使用示例
    manager = SmartContextManager(max_context_messages=4)
    manager.add_system_message("你是助手")

    print("\n✅ 智能上下文管理:")
    print("   - 自动限制消息数量")
    print("   - 始终保留系统消息")
    print("   - 保持最新对话")

    # 添加多轮对话
    for i in range(5):
        manager.add_turn(f"问题{i+1}", f"回答{i+1}")

    messages = manager.get_messages_for_llm()
    print(f"\n   添加了 5 轮对话")
    print(f"   实际保留: {len(messages)} 条消息 (含系统消息)")


# ==================== 5. 错误处理 ====================

def error_handling_patterns():
    """错误处理模式最佳实践"""
    print("\n" + "=" * 60)
    print("错误处理模式最佳实践")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6")

    def safe_invoke(model, messages: List[BaseMessage]) -> Optional[AIMessage]:
        """安全地调用模型"""
        try:
            # 验证消息
            if not messages:
                print("   ❌ 错误: 消息列表为空")
                return None

            # 检查消息格式
            for msg in messages:
                if not isinstance(msg, BaseMessage):
                    print(f"   ❌ 错误: 无效的消息类型 {type(msg)}")
                    return None

            # 调用模型
            response = model.invoke(messages)
            return response

        except Exception as e:
            print(f"   ❌ 调用失败: {str(e)}")
            return None

    print("\n✅ 错误处理最佳实践:")

    # 测试: 正常调用
    print("\n1. 正常调用:")
    result = safe_invoke(model, [HumanMessage(content="你好")])
    if result:
        print(f"   ✓ 成功: {result.content[:30]}...")

    # 测试: 空消息列表
    print("\n2. 空消息列表:")
    result = safe_invoke(model, [])

    # 测试: 无效消息
    print("\n3. 无效消息类型:")
    result = safe_invoke(model, ["invalid"])  # type: ignore

    print("\n   关键点:")
    print("   - 始终验证输入")
    print("   - 使用 try-except 捕获异常")
    print("   - 提供友好的错误信息")
    print("   - 返回合理的默认值")


# ==================== 6. 性能优化 ====================

def performance_optimization():
    """性能优化最佳实践"""
    print("\n" + "=" * 60)
    print("性能优化最佳实践")
    print("=" * 60)

    print("\n1. 减少 Token 使用")
    print("   ✓ 简化系统提示")
    print("   ✓ 修剪长对话历史")
    print("   ✓ 删除冗余消息")

    print("\n2. 批量处理")
    print("   ✓ 合并多个请求")
    print("   ✓ 使用异步调用")
    print("   ✓ 缓存常见响应")

    print("\n3. 上下文优化")
    print("   ✓ 只保留必要的历史")
    print("   ✓ 使用摘要替代详细历史")
    print("   ✓ 懒加载历史数据")

    # 示例:高效的消息构建
    def build_efficient_messages(question: str, context: Optional[str] = None) -> List[BaseMessage]:
        """高效构建消息"""
        messages = [
            SystemMessage(content="简洁回答")  # 简化的系统提示
        ]

        if context:
            # 只在需要时添加上下文
            messages.append(HumanMessage(content=f"背景: {context}"))

        messages.append(HumanMessage(content=question))

        return messages

    print("\n✅ 高效消息构建示例:")
    print("   - 最小化系统提示")
    print("   - 条件添加可选内容")
    print("   - 避免重复信息")


# ==================== 7. 安全和隐私 ====================

def security_and_privacy():
    """安全和隐私最佳实践"""
    print("\n" + "=" * 60)
    print("安全和隐私最佳实践")
    print("=" * 60)

    def sanitize_user_input(content: str) -> str:
        """清理用户输入"""
        # 移除潜在的注入攻击
        dangerous_patterns = ["<script>", "DROP TABLE", "'; DELETE"]

        sanitized = content
        for pattern in dangerous_patterns:
            if pattern.lower() in content.lower():
                print(f"   ⚠️  检测到可疑内容: {pattern}")
                sanitized = sanitized.replace(pattern, "[已过滤]")

        return sanitized

    def create_safe_message(user_input: str) -> HumanMessage:
        """创建安全的消息"""
        # 清理输入
        safe_content = sanitize_user_input(user_input)

        # 不包含敏感信息
        return HumanMessage(
            content=safe_content,
            additional_kwargs={
                "sanitized": safe_content != user_input
            }
        )

    print("\n✅ 安全措施:")
    print("   1. 输入验证和清理")
    print("   2. 不记录敏感信息")
    print("   3. 加密存储历史")
    print("   4. 定期清理数据")
    print("   5. 遵守隐私法规")

    # 测试
    print("\n测试输入清理:")
    test_inputs = [
        "正常问题",
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --"
    ]

    for inp in test_inputs:
        msg = create_safe_message(inp)
        print(f"   输入: {inp[:30]}")
        print(f"   清理后: {msg.content[:30]}")
        print()


# ==================== 8. 可维护性 ====================

def maintainability_patterns():
    """可维护性模式最佳实践"""
    print("\n" + "=" * 60)
    print("可维护性模式最佳实践")
    print("=" * 60)

    # 使用工厂函数
    def create_assistant_system_message(
        role: str,
        language: str = "中文",
        style: str = "友好"
    ) -> SystemMessage:
        """创建助手系统消息的工厂函数"""
        content = f"你是一个{role},使用{language}回答,风格{style}。"
        return SystemMessage(content=content)

    # 使用消息模板
    class MessageTemplates:
        """消息模板"""

        @staticmethod
        def greeting() -> HumanMessage:
            return HumanMessage(content="你好")

        @staticmethod
        def help_request(topic: str) -> HumanMessage:
            return HumanMessage(content=f"请帮我了解 {topic}")

        @staticmethod
        def code_request(language: str, task: str) -> HumanMessage:
            return HumanMessage(
                content=f"请用 {language} 编写代码完成: {task}"
            )

    print("\n✅ 可维护性最佳实践:")
    print("\n1. 使用工厂函数")
    print("   - 封装消息创建逻辑")
    print("   - 参数化配置")
    print("   - 便于修改和测试")

    # 示例
    msg1 = create_assistant_system_message("Python专家")
    msg2 = create_assistant_system_message("翻译助手", language="英文")

    print(f"\n   示例1: {msg1.content}")
    print(f"   示例2: {msg2.content}")

    print("\n2. 使用消息模板")
    print("   - 标准化常用消息")
    print("   - 减少重复代码")
    print("   - 集中管理")

    # 示例
    template_msg = MessageTemplates.code_request("Python", "读取 CSV 文件")
    print(f"\n   示例: {template_msg.content}")


# ==================== 9. 测试友好 ====================

def testing_friendly_patterns():
    """测试友好模式最佳实践"""
    print("\n" + "=" * 60)
    print("测试友好模式最佳实践")
    print("=" * 60)

    def create_test_messages() -> List[BaseMessage]:
        """创建测试消息"""
        return [
            SystemMessage(content="测试系统提示", id="test-sys-1"),
            HumanMessage(content="测试问题", id="test-human-1"),
            AIMessage(content="测试回答", id="test-ai-1")
        ]

    def validate_message_sequence(messages: List[BaseMessage]) -> bool:
        """验证消息序列"""
        if not messages:
            return False

        # 检查是否以系统消息开始
        if not isinstance(messages[0], SystemMessage):
            print("   ✗ 未以系统消息开始")
            return False

        # 检查消息交替
        for i in range(1, len(messages) - 1):
            current = messages[i]
            next_msg = messages[i + 1]

            if isinstance(current, HumanMessage):
                if not isinstance(next_msg, AIMessage):
                    print(f"   ✗ 消息 {i+1} 后应该是 AI 消息")
                    return False

        print("   ✓ 消息序列有效")
        return True

    print("\n✅ 测试友好设计:")
    print("\n1. 可预测的消息 ID")
    print("   - 使用固定 ID 便于测试")
    print("   - 方便断言和验证")

    print("\n2. 消息验证函数")
    print("   - 验证消息格式")
    print("   - 检查消息顺序")
    print("   - 确保数据完整性")

    # 测试
    print("\n测试消息验证:")
    test_msgs = create_test_messages()
    validate_message_sequence(test_msgs)


# ==================== 10. 综合最佳实践清单 ====================

def comprehensive_best_practices():
    """综合最佳实践清单"""
    print("\n" + "=" * 60)
    print("综合最佳实践清单")
    print("=" * 60)

    practices = {
        "消息构建": [
            "✓ 使用清晰具体的系统提示",
            "✓ 结构化用户输入",
            "✓ 选择正确的消息类型",
            "✓ 添加必要的元数据"
        ],
        "上下文管理": [
            "✓ 限制历史长度",
            "✓ 保留关键消息",
            "✓ 使用摘要压缩",
            "✓ 定期清理历史"
        ],
        "性能优化": [
            "✓ 减少 token 使用",
            "✓ 批量处理请求",
            "✓ 缓存常见响应",
            "✓ 异步调用"
        ],
        "安全隐私": [
            "✓ 验证和清理输入",
            "✓ 不记录敏感数据",
            "✓ 加密存储",
            "✓ 遵守法规"
        ],
        "代码质量": [
            "✓ 使用工厂函数",
            "✓ 模板化消息",
            "✓ 错误处理",
            "✓ 单元测试"
        ],
        "可维护性": [
            "✓ 清晰的代码结构",
            "✓ 详细的注释",
            "✓ 一致的命名",
            "✓ 文档完善"
        ]
    }

    for category, items in practices.items():
        print(f"\n【{category}】")
        for item in items:
            print(f"  {item}")

    print("\n" + "=" * 60)
    print("遵循这些实践可以:")
    print("  - 提高代码质量")
    print("  - 降低维护成本")
    print("  - 提升用户体验")
    print("  - 确保系统稳定")
    print("=" * 60)


if __name__ == "__main__":
    try:
        # clear_system_prompts()
        # structured_message_content()
        # proper_message_types()
        # context_management()
        # error_handling_patterns()
        # performance_optimization()
        security_and_privacy()
        # maintainability_patterns()
        # testing_friendly_patterns()
        # comprehensive_best_practices()

        print("\n" + "=" * 60)
        print("所有最佳实践示例完成!")
        print("=" * 60)
        print("\n建议:")
        print("  1. 从小处开始,逐步应用")
        print("  2. 根据项目需求调整")
        print("  3. 持续学习和改进")
        print("  4. 关注社区最佳实践")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
