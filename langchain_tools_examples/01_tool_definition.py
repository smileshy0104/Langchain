"""
LangChain Tools - 工具定义方式示例
演示 @tool 装饰器、args_schema、继承 BaseTool 等方式
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator
from typing import Type, List, Dict, Optional
import os

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. @tool 装饰器 - 最简单方式 ====================

@tool
def search_database(query: str, limit: int = 10) -> str:
    """搜索客户数据库以查找匹配查询的记录。

    Args:
        query: 要查找的搜索词
        limit: 返回的最大结果数
    """
    return f"找到 {limit} 条关于 '{query}' 的结果"


def basic_tool_decorator_example():
    """基础 @tool 装饰器示例"""
    print("=" * 50)
    print("基础 @tool 装饰器示例")
    print("=" * 50)

    # 1. 直接调用工具
    result = search_database.invoke({"query": "张三", "limit": 5})
    print(f"\n直接调用: {result}")

    # 2. 查看工具的 Schema
    print(f"\n工具名称: {search_database.name}")
    print(f"工具描述: {search_database.description}")
    print(f"\nSchema: {search_database.args_schema.schema()}")


# ==================== 2. 详细的文档字符串 ====================

@tool
def send_email(to: str, subject: str, body: str, cc: List[str] = None) -> str:
    """
    发送电子邮件给指定收件人。

    此工具会发送电子邮件并返回发送状态。

    使用场景:
    - 向客户发送通知
    - 发送报告和摘要
    - 转发重要信息

    Args:
        to: 收件人邮箱地址,必须是有效的邮箱格式
        subject: 邮件主题,应简洁明了
        body: 邮件正文,支持 HTML 格式
        cc: 抄送列表,可选参数

    Returns:
        发送状态消息

    Examples:
        >>> send_email("user@example.com", "会议通知", "明天下午 2 点会议")
        "邮件已发送至 user@example.com"

    注意:
        - 请确保收件人地址正确
        - 敏感信息应加密发送
    """
    if cc is None:
        cc = []

    # 模拟邮件发送
    print(f"  发送邮件至: {to}")
    print(f"  主题: {subject}")
    print(f"  抄送: {', '.join(cc) if cc else '无'}")
    return f"邮件已发送至 {to}"


# 详情文档字符串示例
def detailed_docstring_example():
    """详细文档字符串示例"""
    print("\n" + "=" * 50)
    print("详细文档字符串示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")
    model_with_tools = model.bind_tools([send_email])

    response = model_with_tools.invoke([
        HumanMessage(content="发送会议通知邮件给 1259581033@qq.com")
    ])

    print(f"\n工具描述:\n{send_email.description[:200]}...")

    if response.tool_calls:
        print(f"\n模型生成的工具调用:")
        for tool_call in response.tool_calls:
            print(f"  工具: {tool_call['name']}")
            print(f"  参数: {tool_call['args']}")


# ==================== 3. 使用 args_schema 参数 ====================

class DatabaseQueryInput(BaseModel):
    """数据库查询输入参数"""

    query: str = Field(
        description="SQL 查询语句",
        min_length=5,
        max_length=1000
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="最大返回行数"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="跳过的行数"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """验证查询安全性"""
        # 禁止危险操作
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE"]
        v_upper = v.upper()

        if any(word in v_upper for word in forbidden):
            raise ValueError("只允许 SELECT 查询")

        # 必须以 SELECT 开头
        if not v_upper.strip().startswith("SELECT"):
            raise ValueError("查询必须以 SELECT 开始")

        return v

# 使用 args_schema 参数
@tool(args_schema=DatabaseQueryInput)
def query_database(query: str, limit: int = 100, offset: int = 0) -> str:
    """
    在数据库中执行只读查询。

    安全限制:
    - 仅允许 SELECT 语句
    - 最多返回 1000 行
    - 自动超时保护
    """
    # 模拟数据库查询
    return f"查询结果: 返回 {limit} 行数据 (跳过 {offset} 行)"


def args_schema_example():
    """args_schema 参数示例"""
    print("\n" + "=" * 50)
    print("args_schema 参数示例")
    print("=" * 50)

    # 正确的查询
    try:
        result = query_database.invoke({
            "query": "SELECT * FROM users",
            "limit": 50
        })
        print(f"\n✓ 正确查询: {result}")
    except Exception as e:
        print(f"✗ 错误: {e}")

    # 错误的查询 (包含 DELETE)
    try:
        result = query_database.invoke({
            "query": "DELETE FROM users",
            "limit": 50
        })
        print(f"结果: {result}")
    except Exception as e:
        print(f"\n✗ 错误查询被拒绝: {e}")


# ==================== 4. 继承 BaseTool 类 ====================

class SearchInput(BaseModel):
    """搜索工具输入"""
    query: str = Field(description="搜索查询")
    filters: Dict = Field(default_factory=dict, description="过滤条件")


class AdvancedSearchTool(BaseTool):
    """高级搜索工具"""

    name: str = "advanced_search"
    description: str = "在知识库中进行高级搜索,支持复杂过滤"
    args_schema: Type[BaseModel] = SearchInput

    # 可选: 工具特定的配置
    api_key: str = Field(default="", description="API 密钥")
    max_results: int = Field(default=10, description="最大结果数")

    def _run(self, query: str, filters: Dict = None) -> str:
        """同步执行搜索"""
        if filters is None:
            filters = {}

        # 模拟搜索
        print(f"  搜索查询: {query}")
        print(f"  过滤条件: {filters}")
        print(f"  最大结果: {self.max_results}")

        results = [
            f"结果 {i+1}: 关于 '{query}' 的文档"
            for i in range(min(3, self.max_results))
        ]
        return "\n".join(results)

    async def _arun(self, query: str, filters: Dict = None) -> str:
        """异步执行搜索"""
        # 在实际应用中实现异步搜索
        return self._run(query, filters)


def basetool_inheritance_example():
    """BaseTool 继承示例"""
    print("\n" + "=" * 50)
    print("BaseTool 继承示例")
    print("=" * 50)

    # 创建工具实例
    search_tool = AdvancedSearchTool(api_key="test-key", max_results=5)

    # 直接调用
    result = search_tool.invoke({
        "query": "Python 教程",
        "filters": {"level": "beginner", "language": "zh"}
    })

    print(f"\n搜索结果:\n{result}")


# ==================== 5. 自定义工具名称和描述 ====================

@tool("custom_calculator123", description="执行数学计算,支持基本运算和函数")
def custom_calculator(expression: str) -> float:
    """执行数学计算,支持基本运算和函数

    Args:
        expression: 要计算的数学表达式字符串

    Returns:
        计算结果或错误信息
    """
    try:
        return eval(expression)
    except Exception as e:
        return f"计算错误: {e}"


def custom_name_description_example():
    """自定义工具名称和描述示例"""
    print("\n" + "=" * 50)
    print("自定义工具名称和描述示例")
    print("=" * 50)

    print(f"\n工具名称: {custom_calculator.name}")
    print(f"工具描述: {custom_calculator.description}")

    # 使用工具
    result = custom_calculator.invoke({"expression": "10 * 5 + 3"})
    print(f"\n计算结果: {result}")


# ==================== 6. 复杂类型参数 ====================

class Address(BaseModel):
    """地址信息"""
    street: str
    city: str
    country: str
    postal_code: str


class Person(BaseModel):
    """人员信息"""
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=150)
    email: str
    address: Address  # 嵌套模型


class CreateUserInput(BaseModel):
    """创建用户输入"""
    person: Person
    role: str = Field(default="user", description="用户角色")
    active: bool = True


@tool(args_schema=CreateUserInput)
def create_user(person: Person, role: str = "user", active: bool = True) -> str:
    """在系统中创建新用户"""
    return f"用户 {person.name} 已创建 (角色: {role}, 城市: {person.address.city})"


def complex_types_example():
    """复杂类型参数示例"""
    print("\n" + "=" * 50)
    print("复杂类型参数示例")
    print("=" * 50)

    # 创建用户
    user_data = {
        "person": {
            "name": "张三",
            "age": 30,
            "email": "zhang@example.com",
            "address": {
                "street": "中关村大街1号",
                "city": "北京",
                "country": "中国",
                "postal_code": "100000"
            }
        },
        "role": "admin",
        "active": True
    }

    result = create_user.invoke(user_data)
    print(f"\n{result}")


# ==================== 7. 多个工具组合使用 ====================

@tool
def get_user(user_id: str) -> dict:
    """获取用户信息"""
    return {"id": user_id, "name": f"User {user_id}", "active": True}


@tool
def update_user(user_id: str, data: dict) -> str:
    """更新用户信息"""
    return f"用户 {user_id} 已更新: {data}"


@tool
def delete_user(user_id: str) -> str:
    """删除用户"""
    return f"用户 {user_id} 已删除"


def multiple_tools_example():
    """多个工具组合使用示例"""
    print("\n" + "=" * 50)
    print("多个工具组合使用示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")
    model_with_tools = model.bind_tools([get_user, update_user, delete_user])

    response = model_with_tools.invoke([
        HumanMessage(content="获取用户 123 的信息")
    ])

    print("\n可用工具:")
    for tool in [get_user, update_user, delete_user]:
        print(f"  - {tool.name}: {tool.description}")

    if response.tool_calls:
        print(f"\n模型选择的工具: {response.tool_calls[0]['name']}")


if __name__ == "__main__":
    try:
        # basic_tool_decorator_example()
        # detailed_docstring_example()
        # args_schema_example()
        # basetool_inheritance_example()
        # custom_name_description_example()
        # complex_types_example()
        multiple_tools_example()

        print("\n" + "=" * 50)
        print("所有工具定义示例完成!")
        print("=" * 50)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()
