#!/usr/bin/env python3
"""
LangChain v1.0 自定义中间件 - 无API版本演示

此版本不需要实际的API调用，仅用于演示中间件的工作原理和结构。
重点展示自定义中间件的编写方式和用法。
"""

import os
from typing import Callable, Literal

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse

# 从项目根目录加载 .env
dotenv_path = os.path.join(os.path.dirname(__file__), "../..", ".env")
if os.path.exists(dotenv_path):
    with open(dotenv_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value


# ========== 示例工具 ==========

@tool
def simple_search(query: str) -> str:
    """简单的搜索工具（适合初学者）"""
    return f"【简单搜索】{query} - 基础答案"


@tool
def advanced_search(query: str) -> str:
    """高级搜索工具（适合专家）"""
    return f"【高级搜索】{query} - 专业深度分析"


@tool
def basic_calculator(expression: str) -> str:
    """基础计算器"""
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def advanced_analysis(data: str) -> str:
    """高级数据分析工具"""
    return f"【高级分析】数据{data} - 深度统计分析报告"


# ========== 自定义中间件类 ==========

class MockMiddleware(AgentMiddleware):
    """模拟中间件 - 展示基本结构

    这个版本不会真正修改请求，只是演示中间件的调用流程
    """

    def __init__(self, middleware_name: str = "MockMiddleware"):
        self.middleware_name = middleware_name
        self.call_count = 0

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        self.call_count += 1
        print(f"\n🔧 [{self.middleware_name}] 第 {self.call_count} 次调用")
        print(f"   - 消息数量: {len(request.messages)}")
        print(f"   - 模型类型: {type(request.model).__name__}")
        print(f"   - 工具数量: {len(request.tools)}")

        # 模拟中间件处理逻辑
        print(f"   - 执行中间件逻辑...")

        # 实际调用handler
        response = handler(request)

        print(f"   - 处理完成")
        return response


class ConfigurableMiddleware(AgentMiddleware):
    """可配置中间件 - 根据配置调整行为

    展示如何在中间件中存储和使用配置参数
    """

    def __init__(
        self,
        user_level: Literal["beginner", "intermediate", "expert"] = "beginner",
        language: str = "zh",
        max_tokens: int = 2000,
    ):
        self.user_level = user_level
        self.language = language
        self.max_tokens = max_tokens
        self.call_count = 0

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        self.call_count += 1
        print(f"\n⚙️  [ConfigurableMiddleware] 配置:")
        print(f"   - 用户级别: {self.user_level}")
        print(f"   - 语言偏好: {self.language}")
        print(f"   - 最大token: {self.max_tokens}")

        # 尝试修改请求参数
        try:
            # 修改模型参数
            if hasattr(request.model, 'temperature'):
                if self.user_level == "expert":
                    request.model.temperature = 0.2
                elif self.user_level == "beginner":
                    request.model.temperature = 0.7
                print(f"   - 设置温度: {request.model.temperature}")

            if hasattr(request.model, 'max_tokens'):
                request.model.max_tokens = self.max_tokens
                print(f"   - 设置max_tokens: {self.max_tokens}")

        except Exception as e:
            print(f"   - ⚠️ 修改模型参数失败: {e}")

        # 尝试修改工具列表
        try:
            if self.user_level == "expert":
                print(f"   - 专家模式：使用高级工具")
            elif self.user_level == "beginner":
                print(f"   - 初学者模式：使用基础工具")
            else:
                print(f"   - 中级模式：使用标准工具")
        except Exception as e:
            print(f"   - ⚠️ 修改工具列表失败: {e}")

        # 调用handler
        response = handler(request)
        return response


# ========== 演示函数 ==========

def demo_basic_middleware():
    """演示基本中间件调用"""
    print("=" * 70)
    print("🔧 基本中间件演示")
    print("=" * 70)
    print("""
此演示展示中间件的基本结构和调用流程
""")

    # 创建中间件
    middleware = MockMiddleware(middleware_name="TestMiddleware")

    # 创建agent（使用模拟模型，避免API调用）
    try:
        agent = create_agent(
            model=ChatZhipuAI(
                model="glm-4.6",
                temperature=0.5,
                api_key=os.getenv("ZHIPUAI_API_KEY", "demo-key")
            ),
            tools=[simple_search, basic_calculator],
            middleware=[middleware],
        )
        print(f"✅ Agent 创建成功")
        print(f"✅ 中间件已附加到Agent")

        # 模拟调用（不会真正调用API）
        print(f"\n📤 模拟请求:")
        print(f"   消息: '你好'")

        # 注意：这里不会真正发送请求，只是展示结构
        print(f"\n💡 在实际使用中，这里会调用:")
        print(f"   result = agent.invoke({{'messages': [...]}})")
        print(f"   中间件会在调用前后执行自定义逻辑")

    except Exception as e:
        print(f"❌ 创建失败: {e}")
        import traceback
        traceback.print_exc()


def demo_configurable_middleware():
    """演示可配置中间件"""
    print("\n" + "=" * 70)
    print("⚙️ 可配置中间件演示")
    print("=" * 70)
    print("""
展示如何通过中间件构造函数传递参数
""")

    # 不同配置的用户
    users = [
        {"name": "专家", "config": {"user_level": "expert", "language": "zh", "max_tokens": 4000}},
        {"name": "中级", "config": {"user_level": "intermediate", "language": "zh", "max_tokens": 2000}},
        {"name": "初学者", "config": {"user_level": "beginner", "language": "zh", "max_tokens": 1000}},
    ]

    for user in users:
        print(f"\n👤 用户: {user['name']}")

        # 创建中间件实例
        middleware = ConfigurableMiddleware(**user['config'])

        try:
            agent = create_agent(
                model=ChatZhipuAI(
                    model="glm-4.6",
                    temperature=0.5,
                    api_key=os.getenv("ZHIPUAI_API_KEY", "demo-key")
                ),
                tools=[simple_search, advanced_search, basic_calculator, advanced_analysis],
                middleware=[middleware],
            )
            print(f"   ✅ Agent 创建成功")

            # 模拟调用
            print(f"   📝 配置已应用到中间件")

        except Exception as e:
            print(f"   ❌ 创建失败: {e}")


def demo_middleware_combination():
    """演示中间件组合"""
    print("\n" + "=" * 70)
    print("🔗 中间件组合演示")
    print("=" * 70)
    print("""
展示如何将多个中间件组合使用
""")

    # 创建多个中间件
    middlewares = [
        MockMiddleware(middleware_name="Middleware-1"),
        MockMiddleware(middleware_name="Middleware-2"),
        ConfigurableMiddleware(user_level="expert", language="zh"),
    ]

    print(f"\n📦 创建了 {len(middlewares)} 个中间件:")
    for i, mw in enumerate(middlewares, 1):
        print(f"   {i}. {mw.__class__.__name__}")

    try:
        agent = create_agent(
            model=ChatZhipuAI(
                model="glm-4.6",
                temperature=0.5,
                api_key=os.getenv("ZHIPUAI_API_KEY", "demo-key")
            ),
            tools=[simple_search, basic_calculator],
            middleware=middlewares,  # 传递中间件列表
        )
        print(f"\n✅ 所有中间件已组合到Agent中")

    except Exception as e:
        print(f"❌ 创建失败: {e}")


def explain_middleware_architecture():
    """解释中间件架构"""
    print("\n" + "=" * 70)
    print("📚 中间件架构详解")
    print("=" * 70)

    print("""
🔧 中间件的核心组件：

1. 基础类：AgentMiddleware
   - 所有自定义中间件必须继承此类
   - 提供 wrap_model_call 方法

2. 实现方法：wrap_model_call(request, handler) -> response
   - request: ModelRequest 对象，包含所有请求信息
   - handler: 实际处理请求的函数
   - 返回: ModelResponse 对象

3. 工作流程：
   用户请求 → 中间件预处理 → handler执行 → 中间件后处理 → 返回响应

4. 可修改的内容：
   - request.model: 模型参数（temperature, max_tokens等）
   - request.tools: 工具列表
   - request.messages: 消息历史
   - request.config: 配置参数

5. 实际应用场景：
   - 成本控制：限制token使用
   - 性能监控：记录响应时间
   - 安全策略：过滤敏感内容
   - 动态路由：根据用户级别选择模型
   - A/B测试：流量分配
""")

    print("\n💻 代码模板:")
    print("""
class MyMiddleware(AgentMiddleware):
    def __init__(self, param1: str, param2: int):
        self.param1 = param1
        self.param2 = param2

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        # 1. 预处理
        print(f"中间件开始处理: {self.param1}")

        # 2. 修改请求（可选）
        if hasattr(request.model, 'temperature'):
            request.model.temperature = self.param2

        # 3. 调用handler执行请求
        response = handler(request)

        # 4. 后处理（可选）
        print(f"中间件处理完成")

        # 5. 返回响应
        return response

# 使用方式
middleware = MyMiddleware(param1="value", param2=42)
agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[middleware],
)
""")


def demo_api_comparison():
    """对比不同的中间件实现方式"""
    print("\n" + "=" * 70)
    print("⚖️ 不同实现方式对比")
    print("=" * 70)

    print("""
方式1: 通过构造函数传递参数（推荐）
✅ 优点：简单直接，易于理解和测试
✅ 优点：状态清晰，便于调试
✅ 优点：符合Python最佳实践

class MyMiddleware(AgentMiddleware):
    def __init__(self, user_level: str):
        self.user_level = user_level

    def wrap_model_call(self, request, handler):
        if self.user_level == "expert":
            # 使用高级模型
        ...

方式2: 通过运行时上下文传递
⚠️ 缺点：复杂，需要正确的上下文传递机制
⚠️ 缺点：调试困难
⚠️ 缺点：可能与框架版本不兼容

class MyMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        user_level = request.runtime.context.user_level
        ...

方式3: 全局变量（不推荐）
❌ 缺点：不是线程安全的
❌ 缺点：难以测试
❌ 缺点：违反单一职责原则
""")


def main():
    """主函数：运行所有演示"""
    print("🚀 LangChain v1.0 自定义中间件 - 无API演示版")
    print("=" * 80)
    print("""
此版本专注于展示中间件的工作原理和结构
不需要实际的API调用，适合学习和理解概念
    """)

    try:
        explain_middleware_architecture()
        demo_api_comparison()
        demo_basic_middleware()
        demo_configurable_middleware()
        demo_middleware_combination()

        print("\n" + "=" * 70)
        print("✅ 所有演示完成！")
        print("=" * 70)
        print("\n💡 下一步：")
        print("1. 理解中间件的基本结构和调用流程")
        print("2. 根据实际需求创建自定义中间件")
        print("3. 在实际项目中应用中间件")
        print("4. 参考官方文档了解更多高级功能")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
