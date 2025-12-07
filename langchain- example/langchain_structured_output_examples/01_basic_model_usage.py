"""
示例1：Model 中使用 Structured Output - 基础用法
演示如何在 LangChain 模型中使用结构化输出
"""

import os
from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import List, Optional
from langchain_community.chat_models import ChatZhipuAI

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 示例 1.1: 基础 Pydantic Model ====================

class Movie(BaseModel):
    """电影详情"""
    title: str = Field(..., description="电影标题")
    year: int = Field(..., description="上映年份")
    director: str = Field(..., description="导演")
    rating: float = Field(..., description="评分（满分10）", ge=0, le=10)


def example_01_basic_pydantic():
    """示例 1.1: 使用 Pydantic Model"""
    print("\n" + "=" * 60)
    print("示例 1.1: 基础 Pydantic Model")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    model_with_structure = model.with_structured_output(Movie)

    print("\n👤 用户: 提供《盗梦空间》的详细信息")
    response = model_with_structure.invoke("提供《盗梦空间》的详细信息")

    print(f"\n🤖 结构化响应:")
    print(f"   类型: {type(response)}")
    print(f"   数据: {response}")
    print(f"\n📊 字段访问:")
    print(f"   标题: {response.title}")
    print(f"   年份: {response.year}")
    print(f"   导演: {response.director}")
    print(f"   评分: {response.rating}")

    # 可以直接使用
    print(f"\n💾 转换为字典:")
    print(f"   {response.dict()}")

    print(f"\n📄 转换为 JSON:")
    print(f"   {response.json()}")


# ==================== 示例 1.2: 嵌套结构 ====================

class Actor(BaseModel):
    """演员信息"""
    name: str = Field(description="演员姓名")
    role: str = Field(description="角色名称")


class MovieDetails(BaseModel):
    """详细电影信息（包含嵌套结构）"""
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    cast: List[Actor] = Field(default_factory=list, description="主要演员阵容，至少列出2-3位主演")
    genres: List[str] = Field(default_factory=list, description="电影类型/风格")
    budget: Optional[float] = Field(None, description="预算（百万美元）")


def example_02_nested_structure():
    """示例 1.2: 嵌套结构"""
    print("\n" + "=" * 60)
    print("示例 1.2: 嵌套结构")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    model_with_structure = model.with_structured_output(MovieDetails)

    print("\n👤 用户: 提供《盗梦空间》的完整信息，包括主要演员")
    response = model_with_structure.invoke("提供《盗梦空间》的完整信息，包括主要演员阵容")

    print(f"\n🤖 结构化响应:")
    print(f"   标题: {response.title}")
    print(f"   年份: {response.year}")
    print(f"   类型: {', '.join(response.genres)}")
    print(f"   预算: ${response.budget}M" if response.budget else "   预算: 未知")

    print(f"\n🎭 主要演员:")
    for actor in response.cast:
        print(f"   - {actor.name} 饰演 {actor.role}")


# ==================== 示例 1.3: 使用验证器 ====================

class ContactInfo(BaseModel):
    """联系信息（带验证）"""
    name: str = Field(..., min_length=1, max_length=100, description="姓名")
    email: EmailStr = Field(..., description="电子邮箱")
    phone: str = Field(..., pattern=r'^\+?1?\d{9,15}$', description="电话号码")
    company: Optional[str] = Field(None, description="公司名称")

    @field_validator('name')
    @classmethod
    def name_must_be_capitalized(cls, v):
        """姓名首字母必须大写"""
        if not v[0].isupper():
            raise ValueError('姓名首字母必须大写')
        return v

    @field_validator('phone')
    @classmethod
    def standardize_phone(cls, v):
        """标准化电话号码"""
        # 移除所有非数字字符
        digits = ''.join(c for c in v if c.isdigit())
        return digits


def example_03_with_validators():
    """示例 1.3: 使用验证器"""
    print("\n" + "=" * 60)
    print("示例 1.3: 使用验证器")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    model_with_structure = model.with_structured_output(ContactInfo)

    # 校验失败
    text = """
    联系人信息:
    姓名: John Doe
    邮箱: john.doe@techcorp.com
    电话: +1-555-123-4567
    公司: Tech Corp Inc.
    """

    # text = """
    # 联系人信息:
    # 姓名: John Doe
    # 邮箱: john.doe@techcorp.com
    # 电话: 18569364569
    # 公司: Tech Corp Inc.
    # """

    print(f"\n📝 输入文本: {text}")
    print("\n👤 用户: 从文本中提取联系信息")

    response = model_with_structure.invoke(f"从以下文本中提取联系信息：\n{text}")

    print(f"\n🤖 提取结果:")
    print(f"   姓名: {response.name}")
    print(f"   邮箱: {response.email}")
    print(f"   电话: {response.phone}")
    print(f"   公司: {response.company if response.company else '未提供'}")

    print(f"\n✅ 验证通过！所有字段符合要求")


# ==================== 示例 1.4: 获取原始响应 ====================

def example_04_include_raw():
    """示例 1.4: 获取原始响应（包含 token 使用信息）"""
    print("\n" + "=" * 60)
    print("示例 1.4: 获取原始响应")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    model_with_structure = model.with_structured_output(
        Movie,
        include_raw=True  # 包含原始响应
    )

    print("\n👤 用户: 提供《星际穿越》的详细信息")
    response = model_with_structure.invoke("提供《星际穿越》的详细信息")

    print(f"\n🤖 响应结构:")
    print(f"   类型: {type(response)}")
    print(f"   键: {list(response.keys())}")

    # 访问解析后的数据
    movie = response['parsed']
    print(f"\n📊 解析后的数据:")
    print(f"   {movie}")

    # 访问原始响应
    raw = response['raw']
    print(f"\n📄 原始响应信息:")
    print(f"   消息类型: {type(raw)}")
    print(f"   内容预览: {str(raw.content)[:100]}...")

    # 访问 token 使用情况（如果可用）
    if hasattr(raw, 'usage_metadata') and raw.usage_metadata:
        print(f"\n💰 Token 使用:")
        print(f"   {raw.usage_metadata}")


# ==================== 示例 1.5: 多个实例提取 ====================

class Person(BaseModel):
    """人员信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄", ge=0, le=150)
    occupation: str = Field(description="职业")


class People(BaseModel):
    """多个人员信息"""
    persons: List[Person] = Field(description="人员列表")


def example_05_multiple_instances():
    """示例 1.5: 提取多个实例"""
    print("\n" + "=" * 60)
    print("示例 1.5: 提取多个实例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    model_with_structure = model.with_structured_output(People)

    text = """
    团队成员:
    1. Alice Wang, 28岁, 软件工程师
    2. Bob Chen, 35岁, 项目经理
    3. Carol Li, 42岁, 产品设计师
    """

    print(f"\n📝 输入文本: {text}")
    print("\n👤 用户: 提取所有团队成员信息")

    response = model_with_structure.invoke(f"从以下文本提取所有人员信息：\n{text}")

    print(f"\n🤖 提取的人员列表:")
    for i, person in enumerate(response.persons, 1):
        print(f"   {i}. {person.name}, {person.age}岁, {person.occupation}")

    print(f"\n📊 统计:")
    print(f"   总人数: {len(response.persons)}")
    print(f"   平均年龄: {sum(p.age for p in response.persons) / len(response.persons):.1f}岁")


# ==================== 主函数 ====================

def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("LangChain Structured Output - Model 基础用法")
    print("=" * 60)

    examples = [
        # ("基础 Pydantic Model", example_01_basic_pydantic),
        # ("嵌套结构", example_02_nested_structure),
        # ("使用验证器", example_03_with_validators),
        # ("获取原始响应", example_04_include_raw),
        ("提取多个实例", example_05_multiple_instances),
    ]

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"运行示例 {i}/{len(examples)}: {name}")
        print(f"{'='*60}")
        try:
            func()
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已终止")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
