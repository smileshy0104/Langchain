"""
示例3：实际应用场景
演示 Structured Output 在真实场景中的应用
"""

import os
from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional
from enum import Enum
from datetime import date
from langchain_community.chat_models import ChatZhipuAI

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 场景 1: 数据提取 ====================

class ExtractedContact(BaseModel):
    """提取的联系信息"""
    name: str = Field(description="全名")
    email: EmailStr = Field(description="电子邮箱")
    phone: Optional[str] = Field(None, description="电话号码")
    company: Optional[str] = Field(None, description="公司名称")
    position: Optional[str] = Field(None, description="职位")


def scenario_01_data_extraction():
    """场景 1: 从邮件签名提取联系信息"""
    print("\n" + "=" * 60)
    print("场景 1: 数据提取 - 邮件签名")
    print("=" * 60)

    email_signature = """
    Best regards,
    张伟
    高级软件工程师
    创新科技有限公司
    zhangwei@innovatech.com
    +86 138-1234-5678
    """

    model = ChatZhipuAI(model="glm-4.6", temperature=0.1)
    extractor = model.with_structured_output(ExtractedContact)

    print(f"\n📧 邮件签名:\n{email_signature}")
    print("\n🔍 提取联系信息...")

    result = extractor.invoke(f"从以下邮件签名中提取联系信息：\n{email_signature}")

    print(f"\n✅ 提取结果:")
    print(f"   姓名: {result.name}")
    print(f"   邮箱: {result.email}")
    print(f"   电话: {result.phone}")
    print(f"   公司: {result.company}")
    print(f"   职位: {result.position}")

    # 可以直接存入数据库
    print(f"\n💾 准备存入数据库:")
    print(f"   {result.dict()}")


# ==================== 场景 2: 内容分类 ====================

class Category(str, Enum):
    """文章分类"""
    TECH = "科技"
    BUSINESS = "商业"
    SPORTS = "体育"
    ENTERTAINMENT = "娱乐"
    POLITICS = "政治"


class ArticleClassification(BaseModel):
    """文章分类结果"""
    title: str = Field(description="文章标题")
    primary_category: Category = Field(description="主要分类")
    secondary_categories: List[Category] = Field(default_factory=list, description="次要分类")
    keywords: List[str] = Field(description="关键词")
    summary: str = Field(max_length=200, description="文章摘要")
    sentiment: str = Field(description="情感倾向：正面/中性/负面")


def scenario_02_classification():
    """场景 2: 新闻文章分类"""
    print("\n" + "=" * 60)
    print("场景 2: 内容分类 - 新闻文章")
    print("=" * 60)

    article = """
    标题: 苹果发布全新AI芯片，性能提升300%

    苹果公司今天在其年度开发者大会上发布了全新的M4芯片，这款芯片集成了先进的AI加速器，
    专门针对机器学习任务进行优化。据苹果工程副总裁介绍，新芯片的AI性能相比上一代提升了
    300%，能耗却降低了40%。这一突破性进展将大大提升Mac电脑在图像处理、视频编辑和AI应用
    方面的性能。业界分析师预计，这将进一步巩固苹果在高端计算市场的领先地位。
    """

    model = ChatZhipuAI(model="glm-4.6", temperature=0.3)
    classifier = model.with_structured_output(ArticleClassification)

    print(f"\n📰 文章内容:\n{article.strip()}")
    print("\n🏷️  分类中...")

    result = classifier.invoke(f"对以下文章进行分类和分析：\n{article}")

    print(f"\n✅ 分类结果:")
    print(f"   标题: {result.title}")
    print(f"   主要分类: {result.primary_category.value}")
    print(f"   次要分类: {', '.join([c.value for c in result.secondary_categories])}")
    print(f"   关键词: {', '.join(result.keywords)}")
    print(f"   摘要: {result.summary}")
    print(f"   情感: {result.sentiment}")


# ==================== 场景 3: 表单填充 ====================

class JobApplication(BaseModel):
    """职位申请表单"""
    first_name: str = Field(description="名字", min_length=1)
    last_name: str = Field(description="姓氏", min_length=1)
    email: EmailStr = Field(description="电子邮箱")
    phone: str = Field(description="电话号码", pattern=r'^\+?86?\d{11}$')
    position: str = Field(description="申请职位")
    years_experience: int = Field(description="工作年限", ge=0, le=50)
    skills: List[str] = Field(description="技能列表")
    education: str = Field(description="学历")
    cover_letter: str = Field(description="求职信", min_length=50, max_length=500)

    @validator('phone')
    def standardize_phone(cls, v):
        """标准化电话号码"""
        return ''.join(c for c in v if c.isdigit())


def scenario_03_form_filling():
    """场景 3: 自动填充求职申请表"""
    print("\n" + "=" * 60)
    print("场景 3: 表单填充 - 求职申请")
    print("=" * 60)

    user_input = """
    我叫李明，姓李名明。我的邮箱是 liming@email.com，电话是 13812345678。
    我想申请Python开发工程师的职位。我有5年的软件开发经验，擅长Python、Django、
    React和Docker。我是计算机科学硕士学位。

    关于求职信：
    我对贵公司的Python开发工程师职位非常感兴趣。我在过去5年中积累了丰富的Web开发经验，
    特别擅长使用Python和Django构建高性能的后端系统。我相信我的技能和经验能够为贵公司
    创造价值。
    """

    model = ChatZhipuAI(model="glm-4.6", temperature=0.1)
    form_filler = model.with_structured_output(JobApplication)

    print(f"\n📝 用户输入:\n{user_input.strip()}")
    print("\n✍️  填充表单...")

    result = form_filler.invoke(f"根据以下信息填写职位申请表：\n{user_input}")

    print(f"\n✅ 填充结果:")
    print(f"   姓名: {result.last_name}{result.first_name}")
    print(f"   邮箱: {result.email}")
    print(f"   电话: {result.phone}")
    print(f"   职位: {result.position}")
    print(f"   经验: {result.years_experience}年")
    print(f"   技能: {', '.join(result.skills)}")
    print(f"   学历: {result.education}")
    print(f"   求职信: {result.cover_letter[:100]}...")


# ==================== 场景 4: 评分系统 ====================

class EssayGrade(BaseModel):
    """作文评分"""
    content_score: int = Field(description="内容分数", ge=0, le=100)
    grammar_score: int = Field(description="语法分数", ge=0, le=100)
    structure_score: int = Field(description="结构分数", ge=0, le=100)
    creativity_score: int = Field(description="创意分数", ge=0, le=100)
    overall_score: int = Field(description="总分", ge=0, le=100)
    strengths: List[str] = Field(description="优点列表")
    weaknesses: List[str] = Field(description="缺点列表")
    feedback: str = Field(description="改进建议")


def scenario_04_grading():
    """场景 4: 自动作文评分"""
    print("\n" + "=" * 60)
    print("场景 4: 评分系统 - 作文评分")
    print("=" * 60)

    essay = """
    标题：我的梦想

    每个人都有自己的梦想，我的梦想是成为一名科学家。从小我就对科学充满了好奇，
    喜欢探索未知的世界。我经常阅读科学书籍，做各种小实验。

    为了实现这个梦想，我努力学习数学和物理，参加科学竞赛，争取获得好成绩。
    我相信只要坚持不懈，总有一天能够实现自己的梦想，为人类的进步做出贡献。
    """

    model = ChatZhipuAI(model="glm-4.6", temperature=0.3)
    grader = model.with_structured_output(EssayGrade)

    print(f"\n📄 作文内容:\n{essay.strip()}")
    print("\n📝 评分中...")

    result = grader.invoke(f"请评分以下作文：\n{essay}")

    print(f"\n✅ 评分结果:")
    print(f"   内容: {result.content_score}/100")
    print(f"   语法: {result.grammar_score}/100")
    print(f"   结构: {result.structure_score}/100")
    print(f"   创意: {result.creativity_score}/100")
    print(f"   总分: {result.overall_score}/100")

    print(f"\n💪 优点:")
    for strength in result.strengths:
        print(f"   - {strength}")

    print(f"\n⚠️  缺点:")
    for weakness in result.weaknesses:
        print(f"   - {weakness}")

    print(f"\n💡 反馈: {result.feedback}")


# ==================== 场景 5: 产品信息提取 ====================

class Product(BaseModel):
    """产品信息"""
    name: str = Field(description="产品名称")
    brand: str = Field(description="品牌")
    price: float = Field(description="价格", gt=0)
    currency: str = Field(default="CNY", description="货币单位")
    specifications: dict = Field(description="规格参数")
    features: List[str] = Field(description="主要特性")
    category: str = Field(description="产品类别")


def scenario_05_product_extraction():
    """场景 5: 电商产品信息提取"""
    print("\n" + "=" * 60)
    print("场景 5: 产品信息提取 - 电商描述")
    print("=" * 60)

    product_description = """
    【小米14 Pro 5G手机】

    价格：¥4999

    小米14 Pro，搭载高通骁龙8 Gen3处理器，配备6.73英寸2K AMOLED屏幕，
    支持120Hz自适应刷新率。后置徕卡三摄系统，主摄5000万像素，支持OIS光学防抖。
    内置5000mAh大电池，支持120W有线快充和50W无线快充。

    主要特性：
    - 徕卡专业影像
    - 2K超清屏幕
    - 骁龙8 Gen3旗舰芯片
    - 120W闪充
    - IP68防尘防水
    """

    model = ChatZhipuAI(model="glm-4.6", temperature=0.1)
    extractor = model.with_structured_output(Product)

    print(f"\n🛍️  产品描述:\n{product_description.strip()}")
    print("\n📦 提取产品信息...")

    result = extractor.invoke(f"从以下描述中提取产品信息：\n{product_description}")

    print(f"\n✅ 提取结果:")
    print(f"   产品: {result.name}")
    print(f"   品牌: {result.brand}")
    print(f"   价格: ¥{result.price}")
    print(f"   类别: {result.category}")

    print(f"\n📊 规格:")
    for key, value in result.specifications.items():
        print(f"   {key}: {value}")

    print(f"\n✨ 特性:")
    for feature in result.features:
        print(f"   - {feature}")


# ==================== 主函数 ====================

def main():
    """运行所有场景"""
    print("\n" + "=" * 60)
    print("LangChain Structured Output - 实际应用场景")
    print("=" * 60)

    scenarios = [
        ("数据提取 - 邮件签名", scenario_01_data_extraction),
        ("内容分类 - 新闻文章", scenario_02_classification),
        ("表单填充 - 求职申请", scenario_03_form_filling),
        ("评分系统 - 作文评分", scenario_04_grading),
        ("产品信息提取 - 电商描述", scenario_05_product_extraction),
    ]

    for i, (name, func) in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"场景 {i}/{len(scenarios)}: {name}")
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
