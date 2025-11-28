"""
示例: 使用 ChatZhipuAI 和结构化输出的 Agent
参考 langchain_agents_examples/04_structured_output.py 的最佳实践
"""

import os
import sys
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import setup_llm

# 加载环境变量
load_dotenv()

# ==================== 定义结构化输出 Schema ====================

class AnalysisResult(BaseModel):
    """文本分析结果"""
    summary: str = Field(description="文本摘要，不超过50字")
    sentiment: str = Field(description="情感倾向：积极/消极/中性")
    keywords: List[str] = Field(description="3-5个关键短语")
    topics: List[str] = Field(description="涉及的主要话题")

class WeatherReport(BaseModel):
    """天气报告"""
    city: str = Field(description="城市名称")
    temperature: float = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况，如晴朗、多云、雨")
    advice: str = Field(description="给用户的出行建议")

# ==================== 主函数 ====================

def main():
    print("🚀 ChatZhipuAI 结构化输出示例")
    print("=" * 60)

    # 1. 初始化 LLM
    try:
        # 使用 GLM-4-Flash 模型，因为它更快且更便宜，适合测试
        # 429 错误通常是因为请求过于频繁或使用了限制较严格的模型
        llm = setup_llm(model="glm-4-flash", temperature=0.1)
    except Exception as e:
        print(f"❌ LLM 初始化失败: {e}")
        return

    # 2. 文本分析示例
    print("\n📝 示例 A: 文本分析 (结构化输出)")
    print("-" * 40)

    # 使用 with_structured_output 绑定 Schema
    analyzer_llm = llm.with_structured_output(AnalysisResult)
    
    text = """
    LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它使得应用程序能够：
    1. 具有上下文感知能力：将语言模型连接到上下文来源（提示指令、少量的示例、内容等）。
    2. 具有推理能力：依靠语言模型进行推理（根据提供的上下文如何回答，采取什么行动等）。
    这个框架非常强大，虽然学习曲线有点陡峭，但一旦掌握就能构建出惊人的应用。
    社区非常活跃，每天都有新的工具和集成出现。
    """
    
    print(f"待分析文本:\n{text.strip()[:100]}...")
    
    try:
        result = analyzer_llm.invoke([
            SystemMessage(content="你是一个专业的文本分析专家。"),
            HumanMessage(content=f"请分析以下文本：\n{text}")
        ])
        
        print(f"\n✅ 分析结果:")
        print(f"  摘要: {result.summary}")
        print(f"  情感: {result.sentiment}")
        print(f"  关键词: {', '.join(result.keywords)}")
        print(f"  话题: {', '.join(result.topics)}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

    # 3. 数据提取示例
    import time
    time.sleep(2) # 避免速率限制
    
    print("\n📝 示例 B: 从非结构化文本提取天气信息")
    print("-" * 40)
    
    # 注意：ChatZhipuAI 对某些类型的结构化输出支持可能有限
    # 我们可以尝试给更明确的 Prompt 指令
    weather_llm = llm.with_structured_output(WeatherReport)
    
    user_input = "哎呀，今天上海热死了，都35度了，太阳大得不得了，一点云都没有。这种天气真的不适合出门，除非去游泳。"
    
    print(f"用户输入: {user_input}")
    
    try:
        report = weather_llm.invoke([
            SystemMessage(content="你是一个数据提取助手。请从用户输入中提取天气信息。城市是上海，温度是35度，天气状况是晴朗。"),
            HumanMessage(content=f"文本：{user_input}")
        ])
        
        if report:
            print(f"\n✅ 天气报告:")
            print(f"  城市: {report.city}")
            print(f"  温度: {report.temperature}°C")
            print(f"  状况: {report.condition}")
            print(f"  建议: {report.advice}")
        else:
            print(f"\n❌ 提取失败: 模型未返回有效数据")
        
    except Exception as e:
        print(f"❌ 提取失败: {e}")

if __name__ == "__main__":
    main()