"""
LLM 客户端模块

提供统一的 LLM 客户端接口,支持通义千问等多种大语言模型。

核心功能:
- 通义千问 ChatTongyi 客户端初始化
- 模型参数配置
- 流式输出支持
- 统一的调用接口
"""

from typing import Optional, List, Dict, Any, Iterator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatTongyi
from langchain_core.callbacks import StreamingStdOutCallbackHandler


class LLMClient:
    """LLM 客户端基类

    提供统一的 LLM 调用接口,支持多种模型后端。

    Attributes:
        model_name: 模型名称
        temperature: 温度参数,控制随机性 (0-1)
        top_p: Top-p 采样参数 (0-1)
        streaming: 是否启用流式输出
        max_tokens: 最大生成 token 数
        llm: 底层 LLM 实例
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        top_p: float = 0.8,
        streaming: bool = False,
        max_tokens: Optional[int] = None
    ):
        """初始化 LLM 客户端

        Args:
            model_name: 模型名称
            temperature: 温度参数 (默认 0.3)
            top_p: Top-p 采样参数 (默认 0.8)
            streaming: 是否启用流式输出 (默认 False)
            max_tokens: 最大生成 token 数 (可选)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.streaming = streaming
        self.max_tokens = max_tokens
        self.llm = None

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """调用 LLM 生成响应

        Args:
            messages: 消息列表

        Returns:
            AIMessage: AI 响应消息

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现 invoke 方法")

    def stream(self, messages: List[BaseMessage]) -> Iterator[str]:
        """流式调用 LLM 生成响应

        Args:
            messages: 消息列表

        Yields:
            str: 生成的文本片段

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现 stream 方法")

    def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息

        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "max_tokens": self.max_tokens
        }


class TongyiLLMClient(LLMClient):
    """通义千问 LLM 客户端

    基于 ChatTongyi 实现的通义千问大语言模型客户端。
    支持多种通义千问模型,包括 qwen-plus、qwen-max、qwen-turbo 等。

    支持的模型:
    - qwen-plus: 通义千问增强版 (推荐,性价比高)
    - qwen-max: 通义千问旗舰版 (最强性能)
    - qwen-turbo: 通义千问快速版 (速度快)
    - qwen-7b-chat: 通义千问 7B 对话版
    - qwen-14b-chat: 通义千问 14B 对话版

    Example:
        >>> client = TongyiLLMClient(
        ...     api_key="your-api-key",
        ...     model="qwen-plus",
        ...     temperature=0.3
        ... )
        >>> messages = [HumanMessage(content="你好")]
        >>> response = client.invoke(messages)
        >>> print(response.content)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        temperature: float = 0.3,
        top_p: float = 0.8,
        streaming: bool = False,
        max_tokens: Optional[int] = None,
        verbose: bool = False
    ):
        """初始化通义千问客户端

        Args:
            api_key: DashScope API 密钥
            model: 模型名称 (默认 qwen-plus)
            temperature: 温度参数,控制随机性 (默认 0.3)
            top_p: Top-p 采样参数 (默认 0.8)
            streaming: 是否启用流式输出 (默认 False)
            max_tokens: 最大生成 token 数 (可选)
            verbose: 是否输出详细日志 (默认 False)

        Raises:
            ValueError: 如果 API 密钥为空
        """
        if not api_key or not api_key.strip():
            raise ValueError("API 密钥不能为空")

        super().__init__(
            model_name=model,
            temperature=temperature,
            top_p=top_p,
            streaming=streaming,
            max_tokens=max_tokens
        )

        self.api_key = api_key
        self.verbose = verbose

        # 初始化 ChatTongyi
        self._init_llm()

        if verbose:
            print(f"✅ 通义千问客户端初始化成功")
            print(f"   - 模型: {model}")
            print(f"   - 温度: {temperature}")
            print(f"   - Top-P: {top_p}")
            print(f"   - 流式输出: {streaming}")
            if max_tokens:
                print(f"   - 最大 Token: {max_tokens}")

    def _init_llm(self):
        """初始化底层 ChatTongyi LLM"""
        # 准备参数
        kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "dashscope_api_key": self.api_key,
            "streaming": self.streaming
        }

        # 添加可选参数
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        # 添加流式回调
        if self.streaming:
            kwargs["callbacks"] = [StreamingStdOutCallbackHandler()]

        # 创建 LLM 实例
        self.llm = ChatTongyi(**kwargs)

    def invoke(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> AIMessage:
        """调用通义千问生成响应

        Args:
            messages: 消息列表,支持 SystemMessage、HumanMessage、AIMessage
            **kwargs: 额外参数传递给 LLM

        Returns:
            AIMessage: AI 响应消息

        Example:
            >>> messages = [
            ...     SystemMessage(content="你是一个技术专家"),
            ...     HumanMessage(content="如何使用 Python?")
            ... ]
            >>> response = client.invoke(messages)
            >>> print(response.content)
        """
        if not messages:
            raise ValueError("消息列表不能为空")

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"调用通义千问 LLM")
            print(f"{'='*70}")
            print(f"消息数: {len(messages)}")
            for i, msg in enumerate(messages):
                print(f"  {i+1}. {type(msg).__name__}: {msg.content[:50]}...")
            print(f"{'='*70}\n")

        # 调用 LLM
        response = self.llm.invoke(messages, **kwargs)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ 响应生成完成")
            print(f"{'='*70}")
            print(f"响应长度: {len(response.content)} 字符")
            print(f"响应预览: {response.content[:100]}...")
            print(f"{'='*70}\n")

        return response

    def stream(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> Iterator[str]:
        """流式调用通义千问生成响应

        Args:
            messages: 消息列表
            **kwargs: 额外参数传递给 LLM

        Yields:
            str: 生成的文本片段

        Example:
            >>> messages = [HumanMessage(content="写一首诗")]
            >>> for chunk in client.stream(messages):
            ...     print(chunk, end="", flush=True)
        """
        if not messages:
            raise ValueError("消息列表不能为空")

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"流式调用通义千问 LLM")
            print(f"{'='*70}\n")

        # 流式调用
        for chunk in self.llm.stream(messages, **kwargs):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)

    def batch_invoke(
        self,
        message_lists: List[List[BaseMessage]],
        **kwargs
    ) -> List[AIMessage]:
        """批量调用通义千问生成响应

        Args:
            message_lists: 消息列表的列表
            **kwargs: 额外参数传递给 LLM

        Returns:
            List[AIMessage]: AI 响应列表

        Example:
            >>> message_lists = [
            ...     [HumanMessage(content="问题1")],
            ...     [HumanMessage(content="问题2")]
            ... ]
            >>> responses = client.batch_invoke(message_lists)
        """
        if self.verbose:
            print(f"批量调用 LLM (批次大小: {len(message_lists)})")

        responses = self.llm.batch(message_lists, **kwargs)

        if self.verbose:
            print(f"✅ 批量调用完成 ({len(responses)} 个响应)")

        return responses

    def get_num_tokens(self, text: str) -> int:
        """估算文本的 Token 数

        Args:
            text: 输入文本

        Returns:
            int: 估算的 Token 数

        Note:
            这是一个粗略估算,实际 Token 数可能有差异
        """
        # 简单估算: 中文约 1.5 字符/token, 英文约 4 字符/token
        # 这里使用平均值 2 字符/token
        return len(text) // 2

    def update_config(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """更新模型配置

        Args:
            temperature: 新的温度参数
            top_p: 新的 Top-p 参数
            max_tokens: 新的最大 Token 数
        """
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if max_tokens is not None:
            self.max_tokens = max_tokens

        # 重新初始化 LLM
        self._init_llm()

        if self.verbose:
            print(f"✅ 配置已更新")
            print(f"   - 温度: {self.temperature}")
            print(f"   - Top-P: {self.top_p}")
            print(f"   - 最大 Token: {self.max_tokens}")


# 便捷函数

def create_tongyi_client(
    api_key: str,
    model: str = "qwen-plus",
    temperature: float = 0.3,
    top_p: float = 0.8,
    streaming: bool = False,
    **kwargs
) -> TongyiLLMClient:
    """创建通义千问客户端的便捷函数

    Args:
        api_key: DashScope API 密钥
        model: 模型名称 (默认 qwen-plus)
        temperature: 温度参数 (默认 0.3)
        top_p: Top-p 参数 (默认 0.8)
        streaming: 是否启用流式输出 (默认 False)
        **kwargs: 其他参数传递给 TongyiLLMClient

    Returns:
        TongyiLLMClient: 通义千问客户端实例

    Example:
        >>> client = create_tongyi_client(api_key="your-key")
        >>> response = client.invoke([HumanMessage(content="你好")])
    """
    return TongyiLLMClient(
        api_key=api_key,
        model=model,
        temperature=temperature,
        top_p=top_p,
        streaming=streaming,
        **kwargs
    )


# 示例用法
if __name__ == "__main__":
    import os

    print("=" * 70)
    print("LLM 客户端模块示例")
    print("=" * 70)

    # 示例 1: 基本用法
    print("\n示例 1: 创建客户端")
    print("-" * 70)

    # 从环境变量获取 API 密钥
    api_key = os.getenv("DASHSCOPE_API_KEY", "test-key-for-demo")

    try:
        client = create_tongyi_client(
            api_key=api_key,
            model="qwen-plus",
            temperature=0.3,
            top_p=0.8,
            verbose=True
        )
        print(f"✅ 客户端创建成功: {client.model_name}")
    except Exception as e:
        print(f"⚠️  客户端创建失败: {e}")

    # 示例 2: 配置信息
    print("\n示例 2: 获取配置信息")
    print("-" * 70)
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 示例 3: Token 估算
    print("\n示例 3: Token 估算")
    print("-" * 70)
    text = "这是一个测试文本,用于估算 Token 数量。"
    num_tokens = client.get_num_tokens(text)
    print(f"  文本: {text}")
    print(f"  估算 Token 数: {num_tokens}")

    print("\n" + "=" * 70)
    print("✅ 所有示例执行完成")
    print("=" * 70)
