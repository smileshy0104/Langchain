#!/usr/bin/env python3
"""
LangChain v1.0 Agent Middleware 使用示例

TODO Prebuilt middleware  预构建中间件
演示如何使用各种中间件来增强 Agent 功能：
1. PIIMiddleware - 保护个人身份信息
2. SummarizationMiddleware - 自动对话总结
3. HumanInTheLoopMiddleware - 人机交互审核

基于 GLM-4.6 模型实现
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)

# 从项目根目录加载 .env
dotenv.load_dotenv(dotenv_path="../../.env")


def _require_env_var(name: str) -> str:
    """确保必需的环境变量存在。"""
    value = os.getenv(name)
    if not value or value.startswith("your-"):
        raise EnvironmentError(
            f"未检测到有效的 {name}，请在项目根目录的 .env 中配置后重试。"
        )
    return value


@tool
def read_email(email_id: str) -> str:
    """读取邮件内容。

    Args:
        email_id: 邮件ID，例如 'email_001', 'email_002'
    """
    # 模拟邮件数据库
    emails = {
        "email_001": {
            "from": "张三 <zhangsan@example.com>",
            "to": "李四 <lisi@example.com>",
            "subject": "项目合作提案",
            "content": "亲爱的李四，我希望我们可以合作开发新的AI产品。请联系我：13800138001",
        },
        "email_002": {
            "from": "王五 <wangwu@company.com>",
            "to": "赵六 <zhaoliu@company.com>",
            "subject": "会议安排",
            "content": "明天下午3点的会议将在公司会议室举行，请准时参加。联系电话：010-12345678",
        },
    }

    if email_id in emails:
        email = emails[email_id]
        return f"""
邮件详情：
发件人：{email['from']}
收件人：{email['to']}
主题：{email['subject']}
内容：{email['content']}
        """.strip()
    else:
        return f"未找到邮件ID为 {email_id} 的邮件"


@tool
def send_email(to: str, subject: str, content: str) -> str:
    """发送邮件。

    Args:
        to: 收件人邮箱
        subject: 邮件主题
        content: 邮件内容
    """
    # 模拟发送邮件
    return f"""
✅ 邮件发送成功！
收件人：{to}
主题：{subject}
内容：{content}
"""


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。

    Args:
        city: 城市名称，例如 '北京', '上海', '厦门'
    """
    import random
    conditions = ["晴天", "多云", "小雨", "阴天"]
    temp = random.randint(15, 25)
    condition = random.choice(conditions)
    return f"{city}今天天气：{condition}，温度 {temp}°C"


def create_agent_with_pii_protection() -> Any:
    """创建带有PII保护的Agent

    PIIMiddleware 可以：
    1. 自动检测和脱敏个人身份信息
    2. 支持正则表达式自定义检测规则
    3. 提供多种处理策略：redact（脱敏）、block（阻止）、mask（掩码）
    """
    api_key = _require_env_var("ZHIPUAI_API_KEY")

    llm = ChatZhipuAI(
        model="glm-4.6",
        temperature=0.3,
        api_key=api_key,
    )

    # 创建带PII保护的Agent
    agent = create_agent(
        model=llm,
        tools=[read_email],
        system_prompt="你是一个邮件助手，可以帮助用户读取和管理邮件。",
        middleware=[
            # 保护邮箱地址
            PIIMiddleware(
                pii_type="email",
                strategy="redact",  # 脱敏处理
                apply_to_input=True,
                apply_to_output=True,
            ),
            # 保护电话号码
            PIIMiddleware(
                pii_type="phone_number",
                detector=r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}",
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True,
            ),
            # 保护身份证号
            PIIMiddleware(
                pii_type="id_number",
                detector=r"\b\d{17}[\dXx]\b",
                strategy="block",  # 阻止包含身份证号的请求
                apply_to_input=True,
            ),
        ],
        debug=False,
    )

    return agent


def create_agent_with_summary() -> Any:
    """创建带自动总结功能的Agent

    SummarizationMiddleware 可以：
    1. 自动总结长对话
    2. 减少token使用量
    3. 提高处理效率
    4. 保持对话上下文连贯性
    """
    api_key = _require_env_var("ZHIPUAI_API_KEY")

    llm = ChatZhipuAI(
        model="glm-4.6",
        temperature=0.3,
        api_key=api_key,
    )

    # 创建带自动总结的Agent
    agent = create_agent(
        model=llm,
        tools=[read_email, get_weather],
        system_prompt="你是一个智能助手，可以查询天气和处理邮件。",
        middleware=[
            SummarizationMiddleware(
                model=llm,
                max_tokens_before_summary=500,  # 超过500个token时自动总结
                summary_prompt="请将以上对话总结为要点",
            ),
        ],
        debug=False,
    )

    return agent


def create_agent_with_human_approval() -> Any:
    """创建需要人工审核的Agent

    HumanInTheLoopMiddleware 可以：
    1. 在执行敏感操作前暂停
    2. 要求人工确认
    3. 支持多种决策选项：approve、edit、reject
    4. 提供审核界面和流程
    """
    api_key = _require_env_var("ZHIPUAI_API_KEY")

    llm = ChatZhipuAI(
        model="glm-4.6",
        temperature=0.3,
        api_key=api_key,
    )

    # 创建需要人工审核的Agent
    agent = create_agent(
        model=llm,
        tools=[send_email, get_weather],
        system_prompt="你是一个智能助手，可以查询天气和发送邮件。发送邮件需要人工审核。",
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": {  # 发送邮件时需要审核
                        "allowed_decisions": ["approve", "edit", "reject"]
                    }
                },
                approval_required=lambda action: action.get("tool") == "send_email",
            ),
        ],
        debug=False,
    )

    return agent


def create_agent_with_all_middleware() -> Any:
    """创建集成所有中间件的Agent

    组合使用多个中间件：
    1. PII保护 + 自动总结 + 人工审核
    2. 提供全方位的安全和控制
    """
    api_key = _require_env_var("ZHIPUAI_API_KEY")

    llm = ChatZhipuAI(
        model="glm-4.6",
        temperature=0.3,
        api_key=api_key,
    )

    # 创建集成所有中间件的Agent
    agent = create_agent(
        model=llm,
        tools=[read_email, send_email, get_weather],
        system_prompt="""你是一个智能助手，具有以下能力：
1. 读取邮件
2. 发送邮件（需审核）
3. 查询天气

安全规则：
- 保护用户隐私信息
- 自动总结长对话
- 发送邮件前需要人工确认
""",
        middleware=[
            # PII保护
            PIIMiddleware(
                pii_type="email",
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True,
            ),
            PIIMiddleware(
                pii_type="phone_number",
                detector=r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}",
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True,
            ),
            # 自动总结
            SummarizationMiddleware(
                model=llm,
                max_tokens_before_summary=400,
                summary_prompt="总结对话要点，保留重要信息",
            ),
            # 人工审核
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": {
                        "allowed_decisions": ["approve", "edit", "reject"]
                    }
                },
                approval_required=lambda action: action.get("tool") == "send_email",
            ),
        ],
        debug=True,
    )

    return agent


def demo_pii_protection():
    """演示PII保护功能"""
    print("=" * 70)
    print("🔒 PII保护演示")
    print("=" * 70)
    print("""
功能说明：
- 自动检测和脱敏个人身份信息
- 保护邮箱地址、电话号码、身份证号等
- 支持多种处理策略：redact、block、mask
    """)

    agent = create_agent_with_pii_protection()

    # 测试PII保护
    test_inputs = [
        "请帮我读取邮件 email_001",
        "我的邮箱是 zhangsan@example.com，请帮我处理邮件",
        "请不要泄露我的电话号码 13800138001",
        "你的电话号码是什么？",
        "输出邮件内容，包括发件人联系方式",
    ]

    messages = []
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n🧪 测试 {i}: {user_input}")
        messages.append(HumanMessage(content=user_input))

        try:
            result = agent.invoke({"messages": messages})
            messages = result["messages"]
            response = messages[-1]
            print(f"✅ Agent 回答: {response.content[:200]}...")
        except Exception as e:
            print(f"❌ 处理失败: {e}")


def demo_summary():
    """演示自动总结功能"""
    print("\n" + "=" * 70)
    print("📝 自动总结演示")
    print("=" * 70)
    print("""
功能说明：
- 自动检测对话长度
- 超过阈值时生成对话总结
- 减少token使用量
- 保持上下文连贯性
    """)

    agent = create_agent_with_summary()

    # 模拟多轮对话
    conversation = [
        "你好，我想查看邮件",
        "请帮我读取 email_001",
        "今天的天气怎么样？",
        "帮我查询北京的天气",
        "我再看一下 email_002",
        "上海天气如何？",
    ]

    messages = []
    # 多轮对话测试
    for i, user_input in enumerate(conversation, 1):
        print(f"\n💬 对话 {i}: {user_input}")
        messages.append(HumanMessage(content=user_input))

        try:
            result = agent.invoke({"messages": messages})
            messages = result["messages"]
            response = messages[-1]
            print(f"🤖 Agent: {response.content[:150]}...")
        except Exception as e:
            print(f"❌ 处理失败: {e}")


def demo_human_approval():
    """演示人工审核功能"""
    print("\n" + "=" * 70)
    print("👤 人工审核演示")
    print("=" * 70)
    print("""
功能说明：
- 敏感操作前自动暂停
- 请求人工确认
- 支持 approve、edit、reject 决策
- 提供审核流程
    """)

    agent = create_agent_with_human_approval()

    # 测试发送邮件（需要审核）
    test_input = "请给 zhangsan@example.com 发送一封主题为'测试邮件'的邮件，内容是'这是一封测试邮件'"

    print(f"\n🧪 测试: {test_input}")
    messages = [HumanMessage(content=test_input)]

    try:
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        response = messages[-1]
        print(f"🤖 Agent: {response.content[:200]}...")
        print("\n💡 注意：实际使用中，发送邮件会触发人工审核流程")
    except Exception as e:
        print(f"❌ 处理失败: {e}")


def demo_all_middleware():
    """演示所有中间件组合"""
    print("\n" + "=" * 70)
    print("🎯 所有中间件组合演示")
    print("=" * 70)
    print("""
组合功能：
1. 🔒 PII保护 - 保护个人隐私
2. 📝 自动总结 - 优化对话效率
3. 👤 人工审核 - 控制敏感操作
    """)

    agent = create_agent_with_all_middleware()

    # 复杂测试场景
    test_scenario = """
请帮我做以下事情：
1. 读取邮件 email_001
2. 给 zhangsan@example.com 发送邮件
3. 主题：关于项目的讨论
4. 内容：我想讨论一下我们之前提到的AI项目进展
    """

    print(f"\n🧪 综合测试: {test_scenario.strip()}")
    messages = [HumanMessage(content=test_scenario)]

    try:
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        response = messages[-1]
        print(f"🤖 Agent: {response.content[:200]}...")
        print("\n💡 响应中包含了PII保护、可能的总结和审核标记")
    except Exception as e:
        print(f"❌ 处理失败: {e}")


def explain_middleware():
    """详细解释中间件机制"""
    print("\n" + "=" * 70)
    print("📚 Middleware 机制详解")
    print("=" * 70)

    print("""
🔧 Middleware 工作原理：

1. 请求处理流程：
   用户输入 → PII检测 → 内容分析 → 工具调用 → 人工审核 → 响应输出

2. 中间件类型：

   📌 PIIMiddleware：
   - 用途：保护个人身份信息
   - 支持类型：email, phone_number, id_number, credit_card, etc.
   - 处理策略：
     * redact: 脱敏处理（如：zhangsan@example.com → z***@***.com）
     * block: 阻止请求
     * mask: 部分掩码

   📌 SummarizationMiddleware：
   - 用途：自动总结长对话
   - 触发条件：超过 max_tokens_before_summary
   - 优势：节省token、保持上下文、提升性能

   📌 HumanInTheLoopMiddleware：
   - 用途：人工审核关键操作
   - 审核点：工具调用前、发送邮件前等
   - 决策选项：approve、edit、reject

3. 使用场景：
   - 客户服务系统：保护客户隐私
   - 企业邮件助手：需要审核敏感操作
   - 长对话应用：自动总结提升效率
   - 金融/医疗：严格的合规要求

4. 自定义中间件：
   可以继承 BaseMiddleware 类创建自定义中间件

   from langchain.agents.middleware import BaseMiddleware

   class CustomMiddleware(BaseMiddleware):
       async def acall_ms(
           self,
           request: AgentScratchPadCallRequest,
           *,
           start_time: float | None = None,
           **kwargs: Any,
       ) -> AgentScratchPadCallRequest:
           # 自定义处理逻辑
           return request
    """)


def main():
    """主函数：运行所有演示"""
    print("🚀 LangChain v1.0 Agent Middleware 完整示例")
    print("=" * 80)
    print("""
✨ Middleware 功能展示：
1. 🔒 PIIMiddleware - 个人身份信息保护
2. 📝 SummarizationMiddleware - 自动对话总结
3. 👤 HumanInTheLoopMiddleware - 人工审核
4. 🎯 多中间件组合使用

基于 GLM-4.6 模型
    """)

    # 检查 API 密钥
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        print("❌ 错误：请在 .env 文件中设置您的 ZHIPUAI_API_KEY")
        print("📝 获取 API 密钥：https://open.bigmodel.cn/")
        return

    try:
        # 运行各种演示
        # explain_middleware()
        # demo_pii_protection()
        # demo_summary()
        # demo_human_approval()
        demo_all_middleware()

        print("\n" + "=" * 70)
        print("🎉 所有 Middleware 演示完成！")
        print("=" * 70)
        print("\n📚 更多信息请参考：")
        print("- LangChain Agent Middleware: https://python.langchain.com/docs/how_to/agents_middleware/")
        print("- PII 检测: https://python.langchain.com/docs/how_to/pii/")
        print("- 人工审核: https://python.langchain.com/docs/how_to/human_approval/")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
