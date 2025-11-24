"""
LangChain Agents 结构化输出示例
演示如何让 Agent 返回符合预定义 schema 的结构化数据
使用 GLM 模型

⚠️ 重要提示:
ChatZhipuAI 目前只支持 'auto' tool choice 模式,因此不能直接使用
create_agent 的 response_format 参数(该参数会使用非auto的tool choice)。

替代方案:
1. 使用 model.with_structured_output() - 直接在模型层面实现结构化输出
2. 等待 ChatZhipuAI 支持更多 tool choice 模式
3. 使用支持完整 tool choice 的模型(如 OpenAI GPT-4)

本文件展示了两种方式:
- create_agent (需要支持完整 tool choice 的模型)
- with_structured_output (适用于 ChatZhipuAI)
"""

# 注释掉 create_agent 相关导入,因为它与 ChatZhipuAI 不兼容
# from langchain.agents import create_agent
# from langchain.agents.structured_output import ToolStrategy

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal, Union, List
from dataclasses import dataclass
import os

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 基础结构化输出 ====================

class WeatherResponse(BaseModel):
    """天气响应结构"""
    location: str = Field(description="位置")
    temperature: float = Field(description="温度(摄氏度)")
    condition: str = Field(description="天气状况")
    humidity: int = Field(description="湿度百分比", ge=0, le=100)


@tool
def get_weather(location: str) -> str:
    """获取天气信息"""
    return f"{location}: 晴朗,温度 22°C,湿度 55%"


def basic_structured_output():
    """基础结构化输出示例 - 使用 with_structured_output"""
    print("=" * 50)
    print("基础结构化输出示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    # 使用 with_structured_output (兼容 ChatZhipuAI)
    structured_model = model.with_structured_output(WeatherResponse)

    # 直接调用模型,提供足够的信息让它生成结构化输出
    weather_data = structured_model.invoke([
        HumanMessage(content="北京天气如何？温度22度，晴朗，湿度55%")
    ])

    print(f"\n问题: 北京天气如何？")
    print(f"\n结构化输出:")
    print(f"  位置: {weather_data.location}")
    print(f"  温度: {weather_data.temperature}°C")
    print(f"  状况: {weather_data.condition}")
    print(f"  湿度: {weather_data.humidity}%")


# ==================== 2. 产品评论分析 ====================

class ProductReview(BaseModel):
    """产品评论分析"""
    rating: int = Field(description="产品评分", ge=1, le=5)
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="情感倾向")
    key_points: List[str] = Field(description="关键点,每个 1-3 词")
    summary: str = Field(description="简短总结")


def product_review_analysis():
    """产品评论分析示例"""
    print("\n" + "=" * 50)
    print("产品评论分析示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.3)
    structured_model = model.with_structured_output(ProductReview)

    review_text = "很棒的产品,5星推荐!发货速度很快,质量不错,但是价格有点贵。"

    review = structured_model.invoke([
        HumanMessage(content=f"分析这条评论: '{review_text}'")
    ])

    print(f"\n评论: {review_text}")
    print(f"\n分析结果:")
    print(f"  评分: {review.rating} 星")
    print(f"  情感: {review.sentiment}")
    print(f"  关键点: {', '.join(review.key_points)}")
    print(f"  总结: {review.summary}")


# ==================== 3. 联系人信息提取 ====================

class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱地址")
    phone: str = Field(description="电话号码")
    company: str = Field(default="", description="公司名称(可选)")


def contact_extraction():
    """联系人信息提取示例"""
    print("\n" + "=" * 50)
    print("联系人信息提取示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.1)
    structured_model = model.with_structured_output(ContactInfo)

    text = "我是张三,邮箱是 zhangsan@example.com,电话是 13812345678,在ABC科技公司工作。"

    contact = structured_model.invoke([
        HumanMessage(content=f"提取联系人信息: {text}")
    ])

    print(f"\n文本: {text}")
    print(f"\n提取的信息:")
    print(f"  姓名: {contact.name}")
    print(f"  邮箱: {contact.email}")
    print(f"  电话: {contact.phone}")
    print(f"  公司: {contact.company}")


# ==================== 4. 事件提取 ====================

class Event(BaseModel):
    """事件信息"""
    title: str = Field(description="事件标题")
    date: str = Field(description="日期")
    time: str = Field(description="时间")
    location: str = Field(description="地点")
    participants: List[str] = Field(description="参与者列表")


def event_extraction():
    """事件提取示例"""
    print("\n" + "=" * 50)
    print("事件提取示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.2)
    structured_model = model.with_structured_output(Event)

    text = "明天下午3点在会议室A召开项目评审会,参加人员有张三、李四、王五。"

    event = structured_model.invoke([
        HumanMessage(content=f"提取会议信息: {text}")
    ])

    print(f"\n文本: {text}")
    print(f"\n提取的会议信息:")
    print(f"  标题: {event.title}")
    print(f"  日期: {event.date}")
    print(f"  时间: {event.time}")
    print(f"  地点: {event.location}")
    print(f"  参与者: {', '.join(event.participants)}")


# ==================== 5. Union 类型 - 多种可能的输出 ====================

class EmailAction(BaseModel):
    """邮件操作"""
    type: Literal["email"] = "email"
    to: str = Field(description="收件人")
    subject: str = Field(description="主题")
    body: str = Field(description="正文")


class SlackAction(BaseModel):
    """Slack消息操作"""
    type: Literal["slack"] = "slack"
    channel: str = Field(description="频道")
    message: str = Field(description="消息内容")


class TodoAction(BaseModel):
    """待办事项操作"""
    type: Literal["todo"] = "todo"
    task: str = Field(description="任务")
    priority: Literal["high", "medium", "low"] = Field(description="优先级")


# Union 类型: 模型会根据上下文选择合适的 schema
ActionType = Union[EmailAction, SlackAction, TodoAction]


def union_type_example():
    """Union 类型示例"""
    print("\n" + "=" * 50)
    print("Union 类型示例 - 多种输出格式")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.3)
    structured_model = model.with_structured_output(ActionType)

    # 测试不同的请求
    requests = [
        "发邮件给 boss@company.com,主题是'周报',内容是'本周工作总结'",
        "在 #general 频道发送消息: '会议开始了'",
        "创建一个高优先级任务: '完成项目文档'"
    ]

    for req in requests:
        action = structured_model.invoke([
            HumanMessage(content=req)
        ])

        print(f"\n请求: {req}")
        print(f"操作类型: {action.type}")

        if isinstance(action, EmailAction):
            print(f"  收件人: {action.to}")
            print(f"  主题: {action.subject}")
        elif isinstance(action, SlackAction):
            print(f"  频道: {action.channel}")
            print(f"  消息: {action.message}")
        elif isinstance(action, TodoAction):
            print(f"  任务: {action.task}")
            print(f"  优先级: {action.priority}")


# ==================== 6. 数据类(Dataclass)作为 Schema ====================

@dataclass
class ArticleSummary:
    """文章摘要"""
    title: str
    main_points: List[str]
    word_count: int
    category: str


def dataclass_schema_example():
    """使用 Dataclass 作为 Schema 的示例"""
    print("\n" + "=" * 50)
    print("Dataclass Schema 示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.3)
    # with_structured_output 也支持 Dataclass
    structured_model = model.with_structured_output(ArticleSummary)

    article = """
    人工智能的发展趋势

    近年来,人工智能技术突飞猛进。深度学习、大语言模型、多模态AI等技术不断涌现。
    AI正在改变各个行业,从医疗健康到金融服务,从教育到制造业。
    未来,AI将更加智能化、普及化,成为人类生活不可或缺的一部分。
    """

    summary = structured_model.invoke([
        HumanMessage(content=f"分析这篇文章:\n{article}")
    ])

    print(f"\n文章摘要:")
    print(f"  标题: {summary.title}")
    print(f"  分类: {summary.category}")
    print(f"  字数: {summary.word_count}")
    print(f"  要点:")
    for point in summary.main_points:
        print(f"    - {point}")


# ==================== 7. 嵌套结构 ====================

class Address(BaseModel):
    """地址"""
    city: str = Field(description="城市")
    street: str = Field(description="街道")
    postal_code: str = Field(description="邮编")


class Person(BaseModel):
    """人员信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    address: Address = Field(description="地址")
    hobbies: List[str] = Field(description="爱好列表")


def nested_schema_example():
    """嵌套结构示例"""
    print("\n" + "=" * 50)
    print("嵌套结构 Schema 示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.2)
    structured_model = model.with_structured_output(Person)

    text = "张三今年28岁,住在北京市朝阳区建国路88号,邮编100025,喜欢打篮球和阅读。"

    person = structured_model.invoke([
        HumanMessage(content=f"提取人员信息: {text}")
    ])

    print(f"\n文本: {text}")
    print(f"\n提取的信息:")
    print(f"  姓名: {person.name}")
    print(f"  年龄: {person.age}")
    print(f"  地址:")
    print(f"    城市: {person.address.city}")
    print(f"    街道: {person.address.street}")
    print(f"    邮编: {person.address.postal_code}")
    print(f"  爱好: {', '.join(person.hobbies)}")


if __name__ == "__main__":
    try:
        basic_structured_output()
        # product_review_analysis()
        # contact_extraction()
        # event_extraction()
        # union_type_example()
        # dataclass_schema_example()
        # nested_schema_example()
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()
