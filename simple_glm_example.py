#!/usr/bin/env python3
"""
简化版GLM-4.6调用示例
直接使用HTTP API调用，无需复杂依赖
"""

import os
import json
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class GLMClient:
    """GLM API客户端"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages, model="glm-4", temperature=0.7, max_tokens=2000):
        """发送聊天请求"""
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"API调用失败: {e}"
        except KeyError as e:
            return f"响应解析失败: {e}"
        except Exception as e:
            return f"未知错误: {e}"

def basic_example():
    """基础示例"""
    print("=" * 60)
    print("📝 基础示例：GLM-4.6 简单对话")
    print("=" * 60)

    # 检查API密钥
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key == "your-zhipu-api-key-here":
        print("❌ 错误：请在.env文件中设置您的ZHIPUAI_API_KEY")
        return

    # 创建客户端
    client = GLMClient()

    # 简单对话
    messages = [
        {"role": "user", "content": "你好，请简单介绍一下GLM-4模型的特点"}
    ]

    print("🤖 用户：你好，请简单介绍一下GLM-4模型的特点")
    print("⏳ 正在思考...")

    response = client.chat(messages)
    print(f"🧠 GLM-4 回答：\n{response}\n")

def system_prompt_example():
    """系统提示示例"""
    print("=" * 60)
    print("🤖 系统提示示例：角色扮演")
    print("=" * 60)

    client = GLMClient()

    messages = [
        {
            "role": "system",
            "content": "你是一个专业的Python编程老师，擅长用生动有趣的方式讲解编程概念。"
        },
        {
            "role": "user",
            "content": "请用一个形象的比喻解释什么是Python的装饰器。"
        }
    ]

    print("👨‍🏫 编程老师：请用比喻解释装饰器")
    print("⏳ 正在准备生动有趣的解释...")

    response = client.chat(messages, temperature=0.8)
    print(f"📚 老师的回答：\n{response}\n")

def multi_turn_example():
    """多轮对话示例"""
    print("=" * 60)
    print("💬 多轮对话示例：上下文理解")
    print("=" * 60)

    client = GLMClient()

    messages = [
        {"role": "user", "content": "我想学习机器学习，应该从哪里开始？"},
        {"role": "assistant", "content": "学习机器学习建议从以下几个方面开始：\n1. 数学基础：线性代数、概率论、微积分\n2. 编程基础：Python语言\n3. 机器学习理论：监督学习、无监督学习、强化学习\n4. 实践项目：从简单的分类问题开始\n\n你想先从哪个方面入手呢？"},
        {"role": "user", "content": "我想先从Python编程开始，有什么推荐的学习资源吗？"}
    ]

    print("📚 对话历史：")
    print("  用户：我想学习机器学习，应该从哪里开始？")
    print("  AI：建议从数学基础、编程基础、理论和实践开始...")
    print("  用户：我想先从Python编程开始，有什么推荐的学习资源吗？")
    print("⏳ 正在分析学习需求...")

    response = client.chat(messages, temperature=0.7)
    print(f"🎯 学习建议：\n{response}\n")

def creative_example():
    """创意写作示例"""
    print("=" * 60)
    print("✍️ 创意写作示例：AI创作")
    print("=" * 60)

    client = GLMClient()

    messages = [
        {
            "role": "user",
            "content": """请写一个关于AI与人类合作的短诗，要求：
1. 体现AI和人类的互补关系
2. 语言优美，富有想象力
3. 字数控制在150字以内
4. 传递积极向上的情感"""
        }
    ]

    print("🎨 创作要求：AI与人类合作的短诗")
    print("⏳ 正在激发创意灵感...")

    response = client.chat(messages, temperature=0.9)
    print(f"🖋️ AI创作的诗歌：\n{response}\n")

def code_example():
    """代码生成示例"""
    print("=" * 60)
    print("💻 代码生成示例：实用工具")
    print("=" * 60)

    client = GLMClient()

    messages = [
        {
            "role": "user",
            "content": """请用Python写一个简单的待办事项管理器，要求：
1. 使用类实现
2. 包含添加、删除、查看待办事项的功能
3. 有简单的用户交互界面
4. 代码注释清晰
5. 适合初学者理解"""
        }
    ]

    print("⚙️ 代码需求：待办事项管理器")
    print("⏳ 正在编写代码...")

    response = client.chat(messages, temperature=0.3)  # 代码生成用较低温度
    print(f"🐍 生成的Python代码：\n{response}\n")

def translation_example():
    """翻译示例"""
    print("=" * 60)
    print("🌍 翻译示例：多语言能力")
    print("=" * 60)

    client = GLMClient()

    messages = [
        {
            "role": "user",
            "content": """请将以下英文段落翻译成中文，并保持原文的专业性和准确性：

'Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. Machine learning, a subset of AI, enables systems to learn and improve from experience without being explicitly programmed.'"""
        }
    ]

    print("📝 翻译任务：AI相关英文段落")
    print("⏳ 正在进行专业翻译...")

    response = client.chat(messages, temperature=0.2)  # 翻译用很低温度保证准确性
    print(f"🈯️ 翻译结果：\n{response}\n")

def analyze_example():
    """分析示例"""
    print("=" * 60)
    print("🔍 分析示例：文本分析")
    print("=" * 60)

    client = GLMClient()

    text_to_analyze = """
    人工智能技术的发展正在深刻改变着我们的生活方式。从智能手机到自动驾驶汽车，
    从医疗诊断到金融分析，AI的应用无处不在。然而，随着技术的发展，我们也需要
    思考如何确保AI的安全性和伦理问题，让技术真正为人类服务。
    """

    messages = [
        {
            "role": "user",
            "content": f"""请分析以下文本的主要观点和情感倾向：

{text_to_analyze}

请从以下几个方面进行分析：
1. 主要论点
2. 情感倾向（积极/消极/中性）
3. 关键词提取
4. 文本类型判断"""
        }
    ]

    print("📊 分析任务：文本内容分析")
    print("⏳ 正在进行深度分析...")

    response = client.chat(messages, temperature=0.5)
    print(f"📈 分析结果：\n{response}\n")

def main():
    """主函数"""
    print("🚀 GLM-4.6 简化版调用示例")
    print("📋 功能：直接HTTP API调用，无需复杂依赖\n")

    # 检查API密钥
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key == "your-zhipu-api-key-here":
        print("❌ 错误：请在.env文件中设置您的ZHIPUAI_API_KEY")
        print("📝 获取API密钥：https://open.bigmodel.cn/")
        return

    try:
        # 运行各种示例
        basic_example()
        system_prompt_example()
        multi_turn_example()
        creative_example()
        code_example()
        translation_example()
        analyze_example()

        print("🎉 所有示例运行完成！")
        print("\n💡 提示：")
        print("- 修改temperature参数控制输出随机性（0-1）")
        print("- 调整max_tokens参数控制输出长度")
        print("- 可以根据需要修改messages数组来实现不同的对话模式")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")

if __name__ == "__main__":
    main()