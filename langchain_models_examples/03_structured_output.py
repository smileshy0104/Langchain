"""
LangChain Models - 结构化输出示例
演示基本结构化输出、复杂嵌套结构、列表输出、验证器等
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import os

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 基本结构化输出 ====================

class Person(BaseModel):
    """一个人的信息。"""
    name: str = Field(description="人的姓名")
    age: int = Field(description="人的年龄")
    email: str = Field(description="电子邮件地址")
    occupation: str = Field(description="职业")


def basic_structured_output():
    """基本结构化输出示例"""
    print("=" * 50)
    print("基本结构化输出示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")
    structured_model = model.with_structured_output(Person)

    # 调用模型
    response = structured_model.invoke([
        HumanMessage(content="张伟是一位 35 岁的软件工程师，邮箱是 zhang@example.com")
    ])

    # response 是 Person 实例
    print(f"\n姓名: {response.name}")
    print(f"年龄: {response.age}")
    print(f"邮箱: {response.email}")
    print(f"职业: {response.occupation}")


# ==================== 2. 复杂嵌套结构 ====================

class Address(BaseModel):
    """地址信息。"""
    street: str = Field(description="街道地址")
    city: str = Field(description="城市")
    country: str = Field(description="国家")
    postal_code: Optional[str] = Field(description="邮政编码")


class Company(BaseModel):
    """公司信息。"""
    name: str = Field(description="公司名称")
    industry: str = Field(description="行业")
    employees: int = Field(description="员工数量")


class Employee(BaseModel):
    """员工完整信息。"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    position: str = Field(description="职位")
    address: Address = Field(description="住址")
    company: Company = Field(description="所在公司")
    skills: List[str] = Field(description="技能列表")


def nested_structure_output():
    """复杂嵌套结构示例"""
    print("\n" + "=" * 50)
    print("复杂嵌套结构示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")
    structured_model = model.with_structured_output(Employee)

    response = structured_model.invoke([
        HumanMessage(content="""
        李明，32岁，在北京海淀区中关村大街1号的阿里巴巴工作，
        担任高级软件工程师。公司有5000名员工，主要从事电子商务。
        他擅长Python、机器学习和云计算。邮编100080。
        """)
    ])

    print(f"\n员工: {response.name}, {response.position}")
    print(f"公司: {response.company.name}, {response.company.employees}人")
    print(f"地址: {response.address.city}, {response.address.street}")
    print(f"技能: {', '.join(response.skills)}")


# ==================== 3. 列表类型输出 ====================

class Product(BaseModel):
    """产品信息。"""
    name: str = Field(description="产品名称")
    price: float = Field(description="价格")
    category: str = Field(description="类别")


class ProductList(BaseModel):
    """产品列表。"""
    products: List[Product] = Field(description="产品列表")


def list_output_example():
    """列表类型输出示例"""
    print("\n" + "=" * 50)
    print("列表类型输出示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")
    structured_model = model.with_structured_output(ProductList)

    response = structured_model.invoke([
        HumanMessage(content="""
        我们有以下产品:
        1. iPhone 15 Pro - 8999元 - 电子产品
        2. MacBook Air - 9499元 - 电子产品
        3. AirPods Pro - 1999元 - 配件
        """)
    ])

    print("\n产品列表:")
    for i, product in enumerate(response.products, 1):
        print(f"{i}. {product.name}: ¥{product.price} ({product.category})")


# ==================== 4. 使用验证器 ====================

class OrderInfo(BaseModel):
    """订单信息。"""
    order_id: str = Field(description="订单号")
    amount: float = Field(description="金额", gt=0)
    status: str = Field(description="状态")

    @validator('status')
    def validate_status(cls, v):
        """验证状态必须是允许的值之一。"""
        allowed = ['pending', 'paid', 'shipped', 'delivered', 'cancelled']
        if v.lower() not in allowed:
            raise ValueError(f'状态必须是: {allowed}之一')
        return v.lower()

    @validator('order_id')
    def validate_order_id(cls, v):
        """验证订单号格式。"""
        if not v.startswith('ORD-'):
            raise ValueError('订单号必须以 ORD- 开头')
        return v


def validator_example():
    """验证器示例"""
    print("\n" + "=" * 50)
    print("验证器示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")
    structured_model = model.with_structured_output(OrderInfo)

    response = structured_model.invoke([
        HumanMessage(content="订单 ORD-12345，金额 299.99，状态已支付")
    ])

    print(f"\n订单号: {response.order_id}")
    print(f"金额: ¥{response.amount}")
    print(f"状态: {response.status}")


# ==================== 5. 数据提取示例 ====================

class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="姓名")
    phone: str = Field(description="电话号码")
    email: str = Field(description="电子邮件")
    company: Optional[str] = Field(description="公司名称")


def data_extraction_example():
    """数据提取示例"""
    print("\n" + "=" * 50)
    print("数据提取示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0)
    structured_model = model.with_structured_output(ContactInfo)

    text = "我是张三，手机号是13812345678，邮箱zhang@company.com，在ABC科技公司工作。"

    response = structured_model.invoke([
        HumanMessage(content=f"从以下文本中提取联系信息: {text}")
    ])

    print(f"\n原文: {text}")
    print(f"\n提取的信息:")
    print(f"  姓名: {response.name}")
    print(f"  电话: {response.phone}")
    print(f"  邮箱: {response.email}")
    print(f"  公司: {response.company}")


# ==================== 6. 情感分析示例 ====================

class SentimentAnalysis(BaseModel):
    """情感分析结果"""
    text: str = Field(description="原始文本")
    sentiment: str = Field(description="情感: positive/negative/neutral")
    confidence: float = Field(description="置信度 0-1", ge=0, le=1)
    keywords: List[str] = Field(description="关键词列表")


def sentiment_analysis_example():
    """情感分析示例"""
    print("\n" + "=" * 50)
    print("情感分析示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0)
    structured_model = model.with_structured_output(SentimentAnalysis)

    texts = [
        "这个产品非常好用，我很满意！",
        "质量太差了，浪费钱。",
        "还可以，没有特别的感觉。"
    ]

    print("\n情感分析结果:")
    for text in texts:
        response = structured_model.invoke([
            HumanMessage(content=f"分析以下文本的情感: {text}")
        ])

        print(f"\n文本: {response.text}")
        print(f"  情感: {response.sentiment}")
        print(f"  置信度: {response.confidence:.2f}")
        print(f"  关键词: {', '.join(response.keywords)}")


# ==================== 7. 事件提取示例 ====================

class MeetingInfo(BaseModel):
    """会议信息"""
    title: str = Field(description="会议主题")
    date: str = Field(description="日期")
    time: str = Field(description="时间")
    location: str = Field(description="地点")
    participants: List[str] = Field(description="参与者")
    agenda: List[str] = Field(description="议程")


def event_extraction_example():
    """事件提取示例"""
    print("\n" + "=" * 50)
    print("事件提取示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0)
    structured_model = model.with_structured_output(MeetingInfo)

    text = """
    明天下午3点在A会议室召开产品评审会，
    参会人员有张三、李四、王五。
    主要讨论新产品功能设计和上线计划。
    """

    response = structured_model.invoke([
        HumanMessage(content=f"提取会议信息: {text}")
    ])

    print(f"\n会议主题: {response.title}")
    print(f"时间: {response.date} {response.time}")
    print(f"地点: {response.location}")
    print(f"参与者: {', '.join(response.participants)}")
    print(f"议程: {', '.join(response.agenda)}")


if __name__ == "__main__":
    try:
        basic_structured_output()
        nested_structure_output()
        list_output_example()
        validator_example()
        data_extraction_example()
        sentiment_analysis_example()
        event_extraction_example()

        print("\n" + "=" * 50)
        print("所有结构化输出示例完成!")
        print("=" * 50)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()
