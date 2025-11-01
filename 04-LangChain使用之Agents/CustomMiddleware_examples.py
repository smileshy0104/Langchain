#!/usr/bin/env python3
"""
LangChain v1.0 自定义中间件示例

TODO Custom Middleware 自定义中间件
展示如何创建自定义中间件来动态控制Agent行为：
1. 根据用户级别动态选择模型和工具
2. 成本优化中间件
3. 性能监控中间件
4. 基于时间的智能路由
5. 多语言适配中间件

基于 GLM-4.6 模型实现
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
)
from langchain.agents.middleware.types import ModelResponse

# 从项目根目录加载 .env
dotenv.load_dotenv(dotenv_path="../.env")


# ========== 数据类定义 ==========

@dataclass
class UserContext:
    """用户上下文信息"""
    user_id: str
    expertise: Literal["beginner", "intermediate", "expert"] = "beginner" # 专业程度
    language: str = "zh"
    budget_tier: Literal["free", "standard", "premium"] = "standard" # 预算等级
    request_count: int = 0 # 请求次数
    last_request_time: float = field(default_factory=time.time) # 上次请求时间
    avg_response_time: float = 0.0 # 平均响应时间


@dataclass
class RequestMetrics:
    """请求指标"""
    start_time: float = 0.0 
    end_time: float = 0.0
    model_used: str = ""
    tools_used: list[str] = field(default_factory=list)
    tokens_consumed: int = 0
    success: bool = True
    error_message: str = ""


# ========== 示例工具 ==========

@tool
def simple_search(query: str) -> str:
    """简单的搜索工具（适合初学者）

    Args:
        query: 搜索关键词
    """
    return f"【简单搜索】针对 '{query}' 的基础搜索结果：这是基本答案，适合初学者理解。"


@tool
def advanced_search(query: str) -> str:
    """高级搜索工具（适合专家）

    Args:
        query: 搜索关键词
    """
    return f"""【高级搜索】针对 '{query}' 的深度分析：

=== 技术细节 ===
- 算法复杂度：O(n log n)
- 数据来源：学术论文、专利数据库
- 分析维度：技术可行性、市场潜力、创新度

=== 专家结论 ===
基于最新研究，该问题需要综合考虑多个技术领域。
建议深入研读相关文献，并结合实际案例验证。"""


@tool
def basic_calculator(expression: str) -> str:
    """基础计算器（适合初学者）

    Args:
        expression: 数学表达式，例如 '2 + 3 * 4'
    """
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def advanced_analysis(data: str, method: str = "statistical") -> str:
    """高级数据分析工具（适合专家）

    Args:
        data: 待分析的数据
        method: 分析方法：statistical, ML, deep_learning
    """
    analysis_methods = {
        "statistical": "统计分析：均值、方差、相关性分析",
        "ML": "机器学习：聚类、回归、分类模型",
        "deep_learning": "深度学习：神经网络、Transformer架构"
    }

    return f"""【高级数据分析】
方法：{method}
数据：{data}

{analysis_methods.get(method, '未知方法')}

详细报告：
1. 数据清洗和预处理
2. 特征工程和选择
3. 模型训练和验证
4. 结果解释和可视化

建议：使用专业工具进行深入分析。"""


@tool
def get_weather(city: str) -> str:
    """获取天气信息

    Args:
        city: 城市名称
    """
    import random
    conditions = ["晴天", "多云", "小雨"]
    temp = random.randint(15, 25)
    condition = random.choice(conditions)
    return f"{city}今天天气：{condition}，温度 {temp}°C"


# ========== 自定义中间件实现 ==========

class ExpertiseBasedMiddleware(AgentMiddleware):
    """根据用户专业程度动态选择模型和工具的中间件

    功能：
    - beginner: 使用简单模型 + 基础工具
    - intermediate: 使用标准模型 + 常规工具
    - expert: 使用高级模型 + 专业工具
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        # 获取用户专业程度
        user_level = request.runtime.context.expertise

        # 根据专业程度选择不同的模型和工具
        if user_level == "expert":
            # 专家：使用更强大的模型和工具
            model = ChatZhipuAI(
                model="glm-4.6",
                temperature=0.2,
                api_key=os.getenv("ZHIPUAI_API_KEY"),
                max_tokens=4000
            )
            tools = [advanced_search, advanced_analysis, get_weather]
            print(f"🎓 使用专家模式：高级模型 + 专业工具")

        elif user_level == "intermediate":
            # 中级：使用标准配置
            model = ChatZhipuAI(
                model="glm-4.6",
                temperature=0.5,
                api_key=os.getenv("ZHIPUAI_API_KEY"),
                max_tokens=2000
            )
            tools = [simple_search, basic_calculator, get_weather]
            print(f"📚 使用中级模式：标准模型 + 常规工具")

        else:
            # 初学者：使用简化配置
            model = ChatZhipuAI(
                model="glm-4.6",
                temperature=0.7,
                api_key=os.getenv("ZHIPUAI_API_KEY"),
                max_tokens=1000
            )
            tools = [simple_search, get_weather]
            print(f"🌱 使用初学者模式：简化模型 + 基础工具")

        # 更新请求
        request.model = model
        request.tools = tools

        return handler(request)


class CostOptimizationMiddleware(AgentMiddleware):
    """成本优化中间件

    功能：
    - 根据预算等级限制模型使用
    - 监控token消耗
    - 智能降级策略
    """

    def __init__(self):
        self.total_tokens = 0
        self.request_count = 0

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        # 获取用户预算等级
        budget_tier = request.runtime.context.budget_tier

        if budget_tier == "free":
            # 免费用户：严格限制
            if request.model.max_tokens and request.model.max_tokens > 1000:
                print(f"💰 免费用户：限制max_tokens=1000")
                request.model.max_tokens = 1000

        elif budget_tier == "standard":
            # 标准用户：适度限制
            if request.model.max_tokens and request.model.max_tokens > 2000:
                print(f"💰 标准用户：限制max_tokens=2000")
                request.model.max_tokens = 2000

        # premium用户不受限制
        print(f"💎 高级用户：无限制")

        # 执行请求并统计
        start_time = time.time()
        response = handler(request)
        end_time = time.time()

        # 记录指标
        self.request_count += 1
        self.total_tokens += getattr(response, 'token_usage', 0)

        print(f"📊 请求 #{self.request_count} - 耗时: {end_time - start_time:.2f}s")

        return response


class PerformanceMonitoringMiddleware(AgentMiddleware):
    """性能监控中间件

    功能：
    - 记录响应时间
    - 监控模型性能
    - 统计工具使用频率
    - 提供性能报告
    """

    def __init__(self):
        self.metrics: list[RequestMetrics] = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        metric = RequestMetrics(start_time=time.time())

        try:
            response = handler(request)
            metric.success = True
            metric.model_used = getattr(request.model, 'model', 'unknown')
            metric.tools_used = [t.name for t in request.tools]

        except Exception as e:
            metric.success = False
            metric.error_message = str(e)
            raise

        finally:
            metric.end_time = time.time()
            metric.tokens_consumed = 0  # 实际应从响应中获取
            self.metrics.append(metric)

        return response

    def get_report(self) -> dict[str, Any]:
        """生成性能报告"""
        if not self.metrics:
            return {"message": "暂无数据"}

        total_requests = len(self.metrics)
        successful_requests = sum(1 for m in self.metrics if m.success)
        avg_response_time = sum(
            (m.end_time - m.start_time) for m in self.metrics
        ) / total_requests

        return {
            "总请求数": total_requests,
            "成功请求数": successful_requests,
            "成功率": f"{successful_requests / total_requests * 100:.1f}%",
            "平均响应时间": f"{avg_response_time:.2f}s",
            "最慢响应": f"{max((m.end_time - m.start_time) for m in self.metrics):.2f}s",
            "最快响应": f"{min((m.end_time - m.start_time) for m in self.metrics):.2f}s",
        }


class TimeBasedRoutingMiddleware(AgentMiddleware):
    """基于时间的智能路由中间件

    功能：
    - 根据时间段选择不同模型
    - 峰值时段降级策略
    - 智能负载均衡
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        current_hour = time.localtime().tm_hour

        # 定义时间策略
        if 9 <= current_hour <= 18:  # 工作时间
            print(f"⏰ 工作时间 ({current_hour}:00): 使用标准模型")
            request.model.temperature = 0.5
            request.model.max_tokens = min(
                request.model.max_tokens or 2000, 2000
            )
        elif 22 <= current_hour or current_hour <= 7:  # 夜间
            print(f"🌙 夜间 ({current_hour}:00): 使用高效模型")
            request.model.temperature = 0.3
            request.model.max_tokens = min(
                request.model.max_tokens or 3000, 3000
            )
        else:  # 其他时间
            print(f"☀️ 闲时 ({current_hour}:00): 使用高性能模型")
            request.model.temperature = 0.7
            # 不限制max_tokens

        return handler(request)


class MultilingualMiddleware(AgentMiddleware):
    """多语言适配中间件

    功能：
    - 根据用户语言偏好调整提示词
    - 动态选择支持的语言模型
    - 自动翻译辅助
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelRequest:
        user_language = request.runtime.context.language

        # 根据语言调整系统提示词
        system_prompts = {
            "zh": "你是一个有用的中文AI助手。",
            "en": "You are a helpful AI assistant.",
            "ja": "あなたは有用なAIアシスタントです。",
            "es": "Eres un asistente de IA útil.",
        }

        # 获取当前系统提示词
        current_prompt = "你是一个有用的AI助手。"
        for msg in request.messages:
            if isinstance(msg, SystemMessage):
                current_prompt = msg.content
                break

        # 添加语言特定的指导
        language_guide = system_prompts.get(user_language, system_prompts["en"])
        if current_prompt != language_guide:
            # 更新系统消息
            for i, msg in enumerate(request.messages):
                if isinstance(msg, SystemMessage):
                    request.messages[i] = SystemMessage(content=language_guide)
                    break
            else:
                # 如果没有系统消息，添加一个
                request.messages.insert(0, SystemMessage(content=language_guide))

        print(f"🌐 使用语言：{user_language} - {language_guide}")

        return request


# ========== 演示函数 ==========

def demo_expertise_based():
    """演示基于专业程度的动态路由"""
    print("=" * 70)
    print("🎓 专业程度自适应演示")
    print("=" * 70)

    # 创建不同专业程度的用户上下文
    users = [
        UserContext(user_id="user1", expertise="beginner"),
        UserContext(user_id="user2", expertise="intermediate"),
        UserContext(user_id="user3", expertise="expert"),
    ]

    for user in users:
        print(f"\n👤 用户：{user.user_id} (专业程度：{user.expertise})")

        agent = create_agent(
            model=ChatZhipuAI(
                model="glm-4.6",
                temperature=0.5,
                api_key=os.getenv("ZHIPUAI_API_KEY")
            ),
            tools=[simple_search, advanced_search, basic_calculator, advanced_analysis],
            middleware=[ExpertiseBasedMiddleware()],
            context_schema=type("UserContext", (), user.__dict__),  # 动态创建schema
        )

        # 模拟请求
        try:
            result = agent.invoke({
                "messages": [HumanMessage(content="请搜索 'AI发展'")],
                "context": user.__dict__
            })
            print(f"✅ 成功处理请求")
        except Exception as e:
            print(f"❌ 处理失败：{e}")


def demo_cost_optimization():
    """演示成本优化"""
    print("\n" + "=" * 70)
    print("💰 成本优化演示")
    print("=" * 70)

    # 创建不同预算等级的用户
    budgets = ["free", "standard", "premium"]

    for budget in budgets:
        user = UserContext(user_id=f"user_{budget}", budget_tier=budget)

        middleware = CostOptimizationMiddleware()

        agent = create_agent(
            model=ChatZhipuAI(
                model="glm-4.6",
                temperature=0.5,
                api_key=os.getenv("ZHIPUAI_API_KEY"),
                max_tokens=4000  # 初始设置，后续会被中间件调整
            ),
            tools=[simple_search, basic_calculator],
            middleware=[middleware],
            context_schema=type("UserContext", (), user.__dict__),
        )

        try:
            result = agent.invoke({
                "messages": [HumanMessage(content="计算 100 + 200")],
                "context": user.__dict__
            })
            print(f"✅ {budget} 用户请求处理完成")
        except Exception as e:
            print(f"❌ {budget} 用户请求失败：{e}")


def demo_performance_monitoring():
    """演示性能监控"""
    print("\n" + "=" * 70)
    print("📊 性能监控演示")
    print("=" * 70)

    user = UserContext(user_id="test_user")
    middleware = PerformanceMonitoringMiddleware()

    agent = create_agent(
        model=ChatZhipuAI(
            model="glm-4.6",
            temperature=0.5,
            api_key=os.getenv("ZHIPUAI_API_KEY")
        ),
        tools=[simple_search, basic_calculator],
        middleware=[middleware],
        context_schema=type("UserContext", (), user.__dict__),
    )

    # 发送多个请求
    for i in range(3):
        print(f"\n📤 发送请求 #{i+1}")
        try:
            result = agent.invoke({
                "messages": [HumanMessage(content=f"查询天气 (请求 #{i+1})")],
                "context": user.__dict__
            })
            print(f"✅ 请求完成")
        except Exception as e:
            print(f"❌ 请求失败：{e}")

    # 显示性能报告
    print("\n📈 性能报告:")
    report = middleware.get_report()
    for key, value in report.items():
        print(f"  {key}：{value}")


def demo_time_based_routing():
    """演示基于时间的智能路由"""
    print("\n" + "=" * 70)
    print("⏰ 基于时间的智能路由演示")
    print("=" * 70)

    current_hour = time.localtime().tm_hour
    print(f"🕐 当前时间：{current_hour}:00")

    user = UserContext(user_id="time_user")
    middleware = TimeBasedRoutingMiddleware()

    agent = create_agent(
        model=ChatZhipuAI(
            model="glm-4.6",
            temperature=0.5,
            api_key=os.getenv("ZHIPUAI_API_KEY"),
            max_tokens=2000
        ),
        tools=[simple_search],
        middleware=[middleware],
        context_schema=type("UserContext", (), user.__dict__),
    )

    try:
        result = agent.invoke({
            "messages": [HumanMessage(content="你好")],
            "context": user.__dict__
        })
        print(f"✅ 请求处理完成")
    except Exception as e:
        print(f"❌ 请求失败：{e}")


def demo_multilingual():
    """演示多语言适配"""
    print("\n" + "=" * 70)
    print("🌐 多语言适配演示")
    print("=" * 70)

    languages = ["zh", "en", "ja"]

    for lang in languages:
        user = UserContext(user_id=f"user_{lang}", language=lang)
        middleware = MultilingualMiddleware()

        agent = create_agent(
            model=ChatZhipuAI(
                model="glm-4.6",
                temperature=0.5,
                api_key=os.getenv("ZHIPUAI_API_KEY")
            ),
            tools=[simple_search],
            middleware=[middleware],
            context_schema=type("UserContext", (), user.__dict__),
        )

        try:
            result = agent.invoke({
                "messages": [HumanMessage(content="你好")],
                "context": user.__dict__
            })
            print(f"✅ {lang} 语言请求处理完成")
        except Exception as e:
            print(f"❌ {lang} 语言请求失败：{e}")


def explain_custom_middleware():
    """解释自定义中间件机制"""
    print("\n" + "=" * 70)
    print("📚 自定义中间件机制详解")
    print("=" * 70)

    print("""
🔧 自定义中间件的核心：

1. 继承 AgentMiddleware 基类
2. 实现 wrap_model_call 方法
3. 在方法中：
   - 获取请求和运行时上下文
   - 根据需要修改请求（模型、工具、提示词等）
   - 调用 handler(request) 执行请求
   - 处理响应并返回

4. 使用场景：
   ✅ 动态模型选择 - 根据用户级别调整
   ✅ 成本控制 - 限制资源使用
   ✅ 性能监控 - 追踪系统指标
   ✅ 安全策略 - 实施访问控制
   ✅ 智能路由 - 基于时间/位置的优化
   ✅ 多语言支持 - 动态调整语言偏好

5. 关键方法：
   wrap_model_call(request, handler) → response
   - request: ModelRequest 对象，包含所有请求信息
   - handler: 实际处理请求的函数
   - 返回: ModelResponse 对象

6. 访问上下文：
   request.runtime.context - 获取用户上下文
   request.model - 访问当前模型
   request.tools - 访问工具列表
   request.messages - 访问消息历史

7. 动态修改：
   - request.model = new_model
   - request.tools = new_tools
   - request.max_tokens = new_limit
   - request.messages = modified_messages
    """)


def main():
    """主函数：运行所有演示"""
    print("🚀 LangChain v1.0 自定义中间件完整示例")
    print("=" * 80)
    print("""
✨ 演示内容：
1. 🎓 ExpertiseBasedMiddleware - 专业程度自适应
2. 💰 CostOptimizationMiddleware - 成本优化
3. 📊 PerformanceMonitoringMiddleware - 性能监控
4. ⏰ TimeBasedRoutingMiddleware - 时间智能路由
5. 🌐 MultilingualMiddleware - 多语言适配

基于 GLM-4.6 模型
    """)

    # 检查 API 密钥
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        print("❌ 错误：请在 .env 文件中设置您的 ZHIPUAI_API_KEY")
        print("📝 获取 API 密钥：https://open.bigmodel.cn/")
        return

    try:
        # explain_custom_middleware()
        demo_expertise_based()
        # demo_cost_optimization()
        # demo_performance_monitoring()
        # demo_time_based_routing()
        # demo_multilingual()

        print("\n" + "=" * 70)
        print("🎉 所有自定义中间件演示完成！")
        print("=" * 70)
        print("\n💡 提示：实际生产环境中，请根据具体需求调整中间件逻辑")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
