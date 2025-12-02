"""
统一 Embeddings 接口

支持多种 Embedding 服务提供商:
- VolcEngine (火山引擎豆包)
- OpenAI
- DashScope (通义千问)
- Zhipu AI (智谱AI)
"""

from typing import List, Optional
from langchain_core.embeddings import Embeddings
from config.config_loader import AIConfig


class VolcEngineEmbeddings(Embeddings):
    """VolcEngine (火山引擎) Embeddings

    使用火山引擎的豆包 Embedding 模型。

    Attributes:
        api_key: API 密钥
        base_url: API 基础 URL
        model: 模型名称 (如 doubao-embedding-text-240715)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "doubao-embedding-text-240715",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        batch_size: int = 100
    ):
        """初始化 VolcEngine Embeddings

        Args:
            api_key: API 密钥
            model: 模型名称
            base_url: API 基础 URL
            batch_size: 批处理大小 (默认 100，智谱 AI 建议使用 64)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size

        # 使用 OpenAI 兼容的客户端
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            print(f"✅ VolcEngine Embeddings 初始化成功")
            print(f"   - 模型: {model}")
            print(f"   - Base URL: {base_url}")
            print(f"   - 批处理大小: {batch_size}")
        except ImportError:
            raise ImportError(
                "VolcEngine Embeddings 需要安装 openai 包:\n"
                "  pip install openai"
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        embeddings = []

        # 批处理 (使用配置的批处理大小)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )

                # 提取嵌入向量
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                raise Exception(f"VolcEngine Embeddings 调用失败: {e}")

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本

        Args:
            text: 查询文本

        Returns:
            List[float]: 嵌入向量
        """
        return self.embed_documents([text])[0]


def get_embeddings(config: AIConfig) -> Embeddings:
    """根据配置获取 Embeddings 实例

    Args:
        config: AI 配置

    Returns:
        Embeddings: Embeddings 实例

    Raises:
        ValueError: 不支持的 provider

    Example:
        >>> from config.config_loader import load_config
        >>> config = load_config()
        >>> embeddings = get_embeddings(config.ai)
    """
    provider = config.provider.lower()
    embedding_model = config.models.get("embedding", "")

    if provider == "volcengine":
        return VolcEngineEmbeddings(
            api_key=config.api_key,
            model=embedding_model,
            base_url=config.base_url or "https://ark.cn-beijing.volces.com/api/v3"
        )

    elif provider == "dashscope":
        # 使用通义千问 Embeddings
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
            return DashScopeEmbeddings(
                model=embedding_model or "text-embedding-v2",
                dashscope_api_key=config.api_key
            )
        except ImportError:
            raise ImportError(
                "DashScope Embeddings 需要安装 dashscope:\n"
                "  pip install dashscope"
            )

    elif provider == "openai":
        # 使用 OpenAI Embeddings
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=embedding_model or "text-embedding-3-small",
                openai_api_key=config.api_key,
                openai_api_base=config.base_url
            )
        except ImportError:
            raise ImportError(
                "OpenAI Embeddings 需要安装 langchain-openai:\n"
                "  pip install langchain-openai"
            )

    elif provider == "zhipu":
        # 使用智谱 AI Embeddings (OpenAI 兼容)
        # 智谱 AI 限制每次最多 64 条
        return VolcEngineEmbeddings(
            api_key=config.api_key,
            model=embedding_model or "embedding-3",
            base_url=config.base_url or "https://open.bigmodel.cn/api/paas/v4",
            batch_size=64
        )

    else:
        raise ValueError(
            f"不支持的 Embedding provider: {provider}\n"
            f"支持的 providers: volcengine, dashscope, openai, zhipu"
        )


# 示例用法
if __name__ == "__main__":
    print("=" * 70)
    print("Embeddings 测试")
    print("=" * 70)

    from config.config_loader import load_config

    try:
        # 加载配置
        config = load_config("config.yaml")

        print(f"\nAI Provider: {config.ai.provider}")
        print(f"Embedding Model: {config.ai.models.get('embedding')}")

        # 获取 Embeddings 实例
        embeddings = get_embeddings(config.ai)

        # 测试嵌入
        print("\n测试文本嵌入:")
        print("-" * 70)

        test_texts = [
            "这是第一个测试文本",
            "这是第二个测试文本"
        ]

        print(f"测试文本:")
        for i, text in enumerate(test_texts, 1):
            print(f"  {i}. {text}")

        print(f"\n生成嵌入向量... (需要 API 调用)")
        print("(跳过实际调用以避免 API 消耗)")

        # 实际使用时的代码:
        # embeddings_result = embeddings.embed_documents(test_texts)
        # print(f"\n✅ 嵌入成功!")
        # print(f"   - 向量数量: {len(embeddings_result)}")
        # print(f"   - 向量维度: {len(embeddings_result[0])}")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
