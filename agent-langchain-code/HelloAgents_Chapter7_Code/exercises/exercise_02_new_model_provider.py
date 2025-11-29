"""
习题 2: 多模型支持
实践添加新的模型供应商

本文件展示如何为框架添加新的 LLM 提供商支持:
1. Anthropic Claude (使用 langchain-anthropic)
2. Moonshot AI (使用智谱 API 格式)
3. 本地 Ollama 模型
"""

import os
from typing import Optional, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models import ChatZhipuAI

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


class MultiModelLLM:
    """
    多模型 LLM 管理器
    支持多个模型提供商的统一接口
    """

    def __init__(
        self,
        provider: str = "zhipuai",
        model: Optional[str] = None,
        **kwargs
    ):
        """
        初始化多模型 LLM

        Args:
            provider: 提供商名称 (zhipuai/anthropic/moonshot/ollama)
            model: 模型名称
            **kwargs: 其他参数
        """
        self.provider = provider.lower()
        self.model = model
        self.kwargs = kwargs
        self.llm = self._create_llm()

    def _create_llm(self) -> BaseChatModel:
        """根据提供商创建对应的 LLM 实例"""

        if self.provider == "zhipuai":
            return self._create_zhipuai_llm()
        elif self.provider == "anthropic":
            return self._create_anthropic_llm()
        elif self.provider == "moonshot":
            return self._create_moonshot_llm()
        elif self.provider == "ollama":
            return self._create_ollama_llm()
        else:
            raise ValueError(
                f"不支持的提供商: {self.provider}. "
                f"支持的提供商: zhipuai, anthropic, moonshot, ollama"
            )

    def _create_zhipuai_llm(self) -> ChatZhipuAI:
        """创建智谱 AI LLM"""
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("未设置 ZHIPUAI_API_KEY 环境变量")

        model = self.model or "glm-4-plus"
        temperature = self.kwargs.get("temperature", 0.7)

        print(f"✅ 创建智谱AI LLM: {model}")
        return ChatZhipuAI(
            model=model,
            temperature=temperature,
            zhipuai_api_key=api_key,
            **{k: v for k, v in self.kwargs.items() if k != "temperature"}
        )

    def _create_anthropic_llm(self) -> BaseChatModel:
        """创建 Anthropic Claude LLM"""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "请安装 langchain-anthropic: pip install langchain-anthropic"
            )

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("未设置 ANTHROPIC_API_KEY 环境变量")

        model = self.model or "claude-3-5-sonnet-20241022"
        temperature = self.kwargs.get("temperature", 0.7)

        print(f"✅ 创建 Anthropic LLM: {model}")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=api_key,
            **{k: v for k, v in self.kwargs.items() if k != "temperature"}
        )

    def _create_moonshot_llm(self) -> BaseChatModel:
        """
        创建 Moonshot AI LLM
        Moonshot 使用 OpenAI 兼容格式
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "请安装 langchain-openai: pip install langchain-openai"
            )

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("未设置 MOONSHOT_API_KEY 环境变量")

        model = self.model or "moonshot-v1-8k"
        temperature = self.kwargs.get("temperature", 0.7)

        print(f"✅ 创建 Moonshot LLM: {model}")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base="https://api.moonshot.cn/v1",
            **{k: v for k, v in self.kwargs.items() if k != "temperature"}
        )

    def _create_ollama_llm(self) -> BaseChatModel:
        """创建本地 Ollama LLM"""
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            raise ImportError(
                "请安装 langchain-community: pip install langchain-community"
            )

        model = self.model or "llama2"
        temperature = self.kwargs.get("temperature", 0.7)
        base_url = self.kwargs.get("base_url", "http://localhost:11434")

        print(f"✅ 创建 Ollama LLM: {model}")
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            **{k: v for k, v in self.kwargs.items()
               if k not in ["temperature", "base_url"]}
        )

    def invoke(self, messages):
        """调用 LLM"""
        return self.llm.invoke(messages)

    def __repr__(self):
        return f"MultiModelLLM(provider={self.provider}, model={self.model})"


def test_zhipuai():
    """测试智谱 AI"""
    print("\n" + "=" * 60)
    print("测试 1: 智谱 AI (GLM-4)")
    print("=" * 60)

    try:
        llm = MultiModelLLM(provider="zhipuai", model="glm-4-flash")
        response = llm.invoke([{"role": "user", "content": "你好,请介绍你自己"}])
        print(f"\n💬 模型: {llm.model}")
        print(f"📝 响应: {response.content}\n")
        print("✅ 智谱 AI 测试通过")
    except Exception as e:
        print(f"❌ 智谱 AI 测试失败: {e}")


def test_anthropic():
    """测试 Anthropic Claude"""
    print("\n" + "=" * 60)
    print("测试 2: Anthropic Claude")
    print("=" * 60)

    try:
        llm = MultiModelLLM(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022"
        )
        response = llm.invoke([{"role": "user", "content": "Hello, introduce yourself"}])
        print(f"\n💬 模型: {llm.model}")
        print(f"📝 响应: {response.content}\n")
        print("✅ Anthropic 测试通过")
    except ValueError as e:
        print(f"⚠️  Anthropic 测试跳过: {e}")
    except ImportError as e:
        print(f"⚠️  Anthropic 测试跳过: {e}")
    except Exception as e:
        print(f"❌ Anthropic 测试失败: {e}")


def test_moonshot():
    """测试 Moonshot AI"""
    print("\n" + "=" * 60)
    print("测试 3: Moonshot AI")
    print("=" * 60)

    try:
        llm = MultiModelLLM(provider="moonshot", model="moonshot-v1-8k")
        response = llm.invoke([{"role": "user", "content": "你好,请介绍你自己"}])
        print(f"\n💬 模型: {llm.model}")
        print(f"📝 响应: {response.content}\n")
        print("✅ Moonshot 测试通过")
    except ValueError as e:
        print(f"⚠️  Moonshot 测试跳过: {e}")
    except ImportError as e:
        print(f"⚠️  Moonshot 测试跳过: {e}")
    except Exception as e:
        print(f"❌ Moonshot 测试失败: {e}")


def test_ollama():
    """测试 Ollama"""
    print("\n" + "=" * 60)
    print("测试 4: Ollama (本地模型)")
    print("=" * 60)

    try:
        llm = MultiModelLLM(
            provider="ollama",
            model="llama2",
            base_url="http://localhost:11434"
        )
        response = llm.invoke([{"role": "user", "content": "Hello, who are you?"}])
        print(f"\n💬 模型: {llm.model}")
        print(f"📝 响应: {response.content}\n")
        print("✅ Ollama 测试通过")
    except ImportError as e:
        print(f"⚠️  Ollama 测试跳过: {e}")
    except Exception as e:
        print(f"⚠️  Ollama 测试跳过: {e}")
        print("提示: 请确保 Ollama 服务正在运行并已安装 llama2 模型")


def demo_agent_with_multiple_models():
    """演示: 使用不同模型创建 Agent"""
    print("\n" + "=" * 60)
    print("演示: 同一个 Agent,切换不同模型")
    print("=" * 60)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    try:
        from agents.simple_agent_langchain import SimpleAgent

        # 测试问题
        question = "什么是人工智能?请用一句话回答"

        # 使用智谱 AI
        print("\n🤖 Agent 1: 使用智谱 GLM-4-Flash")
        llm1 = MultiModelLLM(provider="zhipuai", model="glm-4-flash")
        agent1 = SimpleAgent(name="智谱助手", llm=llm1.llm)
        response1 = agent1.run(question)
        print(f"回答: {response1}")

        # 如果配置了其他模型,也可以测试
        if os.getenv("ANTHROPIC_API_KEY"):
            print("\n🤖 Agent 2: 使用 Claude")
            llm2 = MultiModelLLM(provider="anthropic")
            agent2 = SimpleAgent(name="Claude助手", llm=llm2.llm)
            response2 = agent2.run(question)
            print(f"回答: {response2}")

        print("\n✅ 多模型 Agent 演示完成")

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


def show_configuration_guide():
    """显示配置指南"""
    print("\n" + "=" * 60)
    print("📋 配置指南")
    print("=" * 60)

    print("""
要测试不同的模型提供商,请在 .env 文件中配置相应的 API Key:

1. 智谱 AI (必需)
   ZHIPUAI_API_KEY=your-key
   获取地址: https://open.bigmodel.cn/

2. Anthropic Claude (可选)
   ANTHROPIC_API_KEY=your-key
   获取地址: https://console.anthropic.com/

3. Moonshot AI (可选)
   MOONSHOT_API_KEY=your-key
   获取地址: https://platform.moonshot.cn/

4. Ollama (可选 - 本地)
   无需 API Key,需要安装 Ollama
   安装地址: https://ollama.ai/
   启动命令: ollama serve
   下载模型: ollama pull llama2

支持的模型列表:
- 智谱: glm-4-plus, glm-4-flash, glm-4, glm-3-turbo
- Claude: claude-3-5-sonnet-20241022, claude-3-opus-20240229
- Moonshot: moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k
- Ollama: llama2, mistral, codellama, qwen 等
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("  习题 2: 多模型支持测试")
    print("=" * 60)

    # 显示配置指南
    show_configuration_guide()

    # 测试各个提供商
    test_zhipuai()
    test_anthropic()
    test_moonshot()
    test_ollama()

    # 演示多模型 Agent
    demo_agent_with_multiple_models()

    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60)
    print("""
💡 扩展思考:
1. 如何自动选择最便宜的模型?
2. 如何实现模型的热切换(不重启程序)?
3. 如何实现模型的负载均衡?
4. 如何为不同任务选择最合适的模型?
    """)
