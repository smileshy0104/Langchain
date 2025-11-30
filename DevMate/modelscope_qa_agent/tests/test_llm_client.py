"""
测试 LLM 客户端模块

测试 TongyiLLMClient 的功能:
- 客户端初始化
- 参数配置
- 消息调用
- 流式输出
- 批量调用
- Token 估算
- 配置更新
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from core.llm_client import (
    LLMClient,
    TongyiLLMClient,
    create_tongyi_client
)


class TestLLMClientBase:
    """测试 LLMClient 基类"""

    def test_base_class_init(self):
        """测试基类初始化"""
        client = LLMClient(
            model_name="test-model",
            temperature=0.5,
            top_p=0.9,
            streaming=True,
            max_tokens=1000
        )

        assert client.model_name == "test-model"
        assert client.temperature == 0.5
        assert client.top_p == 0.9
        assert client.streaming is True
        assert client.max_tokens == 1000

        print("✅ 基类初始化测试通过")

    def test_invoke_not_implemented(self):
        """测试 invoke 未实现"""
        client = LLMClient("test-model")

        with pytest.raises(NotImplementedError):
            client.invoke([HumanMessage(content="test")])

        print("✅ invoke 未实现验证通过")

    def test_stream_not_implemented(self):
        """测试 stream 未实现"""
        client = LLMClient("test-model")

        with pytest.raises(NotImplementedError):
            list(client.stream([HumanMessage(content="test")]))

        print("✅ stream 未实现验证通过")

    def test_get_stats(self):
        """测试获取统计信息"""
        client = LLMClient(
            model_name="test-model",
            temperature=0.3,
            top_p=0.8,
            max_tokens=2000
        )

        stats = client.get_stats()

        assert stats["model_name"] == "test-model"
        assert stats["temperature"] == 0.3
        assert stats["top_p"] == 0.8
        assert stats["max_tokens"] == 2000

        print("✅ 统计信息测试通过")


class TestTongyiLLMClientInit:
    """测试 TongyiLLMClient 初始化"""

    def test_init_success(self):
        """测试成功初始化"""
        client = TongyiLLMClient(
            api_key="test-api-key",
            model="qwen-plus"
        )

        assert client.model_name == "qwen-plus"
        assert client.api_key == "test-api-key"
        assert client.temperature == 0.3  # 默认值
        assert client.top_p == 0.8  # 默认值
        assert client.streaming is False  # 默认值
        assert client.llm is not None

        print("✅ 成功初始化测试通过")

    def test_init_with_custom_params(self):
        """测试自定义参数初始化"""
        client = TongyiLLMClient(
            api_key="test-key",
            model="qwen-max",
            temperature=0.5,
            top_p=0.9,
            streaming=True,
            max_tokens=2000,
            verbose=True
        )

        assert client.model_name == "qwen-max"
        assert client.temperature == 0.5
        assert client.top_p == 0.9
        assert client.streaming is True
        assert client.max_tokens == 2000
        assert client.verbose is True

        print("✅ 自定义参数初始化测试通过")

    def test_init_empty_api_key(self):
        """测试空 API 密钥"""
        with pytest.raises(ValueError, match="API 密钥不能为空"):
            TongyiLLMClient(api_key="")

        with pytest.raises(ValueError, match="API 密钥不能为空"):
            TongyiLLMClient(api_key="   ")

        print("✅ 空 API 密钥验证通过")

    def test_init_default_model(self):
        """测试默认模型"""
        client = TongyiLLMClient(api_key="test-key")

        assert client.model_name == "qwen-plus"
        print("✅ 默认模型测试通过")


class TestTongyiLLMClientInvoke:
    """测试 TongyiLLMClient 调用"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TongyiLLMClient(api_key="test-key", verbose=False)

    def test_invoke_empty_messages(self, client):
        """测试空消息列表"""
        with pytest.raises(ValueError, match="消息列表不能为空"):
            client.invoke([])

        print("✅ 空消息验证通过")

    @patch('core.llm_client.ChatTongyi')
    def test_invoke_single_message(self, mock_chat_tongyi, client):
        """测试单条消息调用"""
        # Mock LLM 响应
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="测试响应")
        mock_chat_tongyi.return_value = mock_llm

        # 重新初始化客户端以使用 mock
        client._init_llm = lambda: setattr(client, 'llm', mock_llm)
        client._init_llm()

        # 调用
        messages = [HumanMessage(content="你好")]
        response = client.invoke(messages)

        assert isinstance(response, AIMessage)
        assert response.content == "测试响应"
        mock_llm.invoke.assert_called_once()

        print("✅ 单条消息调用测试通过")

    @patch('core.llm_client.ChatTongyi')
    def test_invoke_multiple_messages(self, mock_chat_tongyi, client):
        """测试多条消息调用"""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="多消息响应")
        client.llm = mock_llm

        # 调用
        messages = [
            SystemMessage(content="系统提示"),
            HumanMessage(content="用户问题"),
            AIMessage(content="AI 回答"),
            HumanMessage(content="追问")
        ]
        response = client.invoke(messages)

        assert response.content == "多消息响应"
        # 验证传入的消息数量
        call_args = mock_llm.invoke.call_args
        assert len(call_args[0][0]) == 4

        print("✅ 多条消息调用测试通过")


class TestTongyiLLMClientStream:
    """测试 TongyiLLMClient 流式输出"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TongyiLLMClient(api_key="test-key", streaming=True, verbose=False)

    def test_stream_empty_messages(self, client):
        """测试空消息列表"""
        with pytest.raises(ValueError, match="消息列表不能为空"):
            list(client.stream([]))

        print("✅ 流式空消息验证通过")

    @patch('core.llm_client.ChatTongyi')
    def test_stream_output(self, mock_chat_tongyi, client):
        """测试流式输出"""
        # Mock 流式响应
        mock_llm = MagicMock()
        mock_chunks = [
            Mock(content="这"),
            Mock(content="是"),
            Mock(content="流"),
            Mock(content="式"),
            Mock(content="输"),
            Mock(content="出")
        ]
        mock_llm.stream.return_value = iter(mock_chunks)
        client.llm = mock_llm

        # 调用流式输出
        messages = [HumanMessage(content="测试")]
        chunks = list(client.stream(messages))

        assert len(chunks) == 6
        assert "".join(chunks) == "这是流式输出"

        print("✅ 流式输出测试通过")


class TestTongyiLLMClientBatch:
    """测试 TongyiLLMClient 批量调用"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TongyiLLMClient(api_key="test-key", verbose=False)

    @patch('core.llm_client.ChatTongyi')
    def test_batch_invoke(self, mock_chat_tongyi, client):
        """测试批量调用"""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.batch.return_value = [
            AIMessage(content="响应1"),
            AIMessage(content="响应2"),
            AIMessage(content="响应3")
        ]
        client.llm = mock_llm

        # 批量调用
        message_lists = [
            [HumanMessage(content="问题1")],
            [HumanMessage(content="问题2")],
            [HumanMessage(content="问题3")]
        ]
        responses = client.batch_invoke(message_lists)

        assert len(responses) == 3
        assert responses[0].content == "响应1"
        assert responses[1].content == "响应2"
        assert responses[2].content == "响应3"

        print("✅ 批量调用测试通过")


class TestTongyiLLMClientUtils:
    """测试 TongyiLLMClient 工具方法"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TongyiLLMClient(api_key="test-key")

    def test_get_num_tokens_chinese(self, client):
        """测试中文 Token 估算"""
        text = "这是一个测试文本"
        num_tokens = client.get_num_tokens(text)

        # 8 个字符 / 2 = 4 tokens
        assert num_tokens == 4
        print("✅ 中文 Token 估算测试通过")

    def test_get_num_tokens_english(self, client):
        """测试英文 Token 估算"""
        text = "This is a test"
        num_tokens = client.get_num_tokens(text)

        # 14 个字符 / 2 = 7 tokens
        assert num_tokens == 7
        print("✅ 英文 Token 估算测试通过")

    def test_get_num_tokens_mixed(self, client):
        """测试中英文混合 Token 估算"""
        text = "这是 test 文本"
        num_tokens = client.get_num_tokens(text)

        assert num_tokens > 0
        print(f"✅ 混合文本 Token 估算测试通过 (估算: {num_tokens} tokens)")

    def test_get_num_tokens_empty(self, client):
        """测试空文本 Token 估算"""
        num_tokens = client.get_num_tokens("")

        assert num_tokens == 0
        print("✅ 空文本 Token 估算测试通过")


class TestTongyiLLMClientConfig:
    """测试 TongyiLLMClient 配置更新"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return TongyiLLMClient(api_key="test-key", verbose=False)

    def test_update_temperature(self, client):
        """测试更新温度参数"""
        original_temp = client.temperature

        client.update_config(temperature=0.7)

        assert client.temperature == 0.7
        assert client.temperature != original_temp

        print("✅ 温度参数更新测试通过")

    def test_update_top_p(self, client):
        """测试更新 Top-P 参数"""
        original_top_p = client.top_p

        client.update_config(top_p=0.95)

        assert client.top_p == 0.95
        assert client.top_p != original_top_p

        print("✅ Top-P 参数更新测试通过")

    def test_update_max_tokens(self, client):
        """测试更新最大 Token 数"""
        client.update_config(max_tokens=3000)

        assert client.max_tokens == 3000
        print("✅ 最大 Token 数更新测试通过")

    def test_update_multiple_params(self, client):
        """测试同时更新多个参数"""
        client.update_config(
            temperature=0.6,
            top_p=0.85,
            max_tokens=2500
        )

        assert client.temperature == 0.6
        assert client.top_p == 0.85
        assert client.max_tokens == 2500

        print("✅ 多参数更新测试通过")

    def test_update_none_params(self, client):
        """测试不更新任何参数"""
        original_temp = client.temperature
        original_top_p = client.top_p

        client.update_config()

        assert client.temperature == original_temp
        assert client.top_p == original_top_p

        print("✅ 空更新测试通过")


class TestCreateTongyiClient:
    """测试便捷函数"""

    def test_create_client_basic(self):
        """测试基本创建"""
        client = create_tongyi_client(api_key="test-key")

        assert isinstance(client, TongyiLLMClient)
        assert client.model_name == "qwen-plus"
        assert client.temperature == 0.3
        assert client.top_p == 0.8

        print("✅ 基本创建测试通过")

    def test_create_client_with_params(self):
        """测试带参数创建"""
        client = create_tongyi_client(
            api_key="test-key",
            model="qwen-max",
            temperature=0.5,
            top_p=0.9,
            streaming=True
        )

        assert client.model_name == "qwen-max"
        assert client.temperature == 0.5
        assert client.top_p == 0.9
        assert client.streaming is True

        print("✅ 带参数创建测试通过")

    def test_create_client_with_kwargs(self):
        """测试带额外参数创建"""
        client = create_tongyi_client(
            api_key="test-key",
            max_tokens=2000,
            verbose=True
        )

        assert client.max_tokens == 2000
        assert client.verbose is True

        print("✅ 额外参数创建测试通过")


class TestTongyiLLMClientGetStats:
    """测试统计信息"""

    def test_get_stats_basic(self):
        """测试基本统计信息"""
        client = TongyiLLMClient(
            api_key="test-key",
            model="qwen-plus",
            temperature=0.3,
            top_p=0.8
        )

        stats = client.get_stats()

        assert stats["model_name"] == "qwen-plus"
        assert stats["temperature"] == 0.3
        assert stats["top_p"] == 0.8
        assert stats["streaming"] is False
        assert stats["max_tokens"] is None

        print("✅ 基本统计信息测试通过")

    def test_get_stats_with_all_params(self):
        """测试完整统计信息"""
        client = TongyiLLMClient(
            api_key="test-key",
            model="qwen-max",
            temperature=0.5,
            top_p=0.9,
            streaming=True,
            max_tokens=3000
        )

        stats = client.get_stats()

        assert stats["model_name"] == "qwen-max"
        assert stats["temperature"] == 0.5
        assert stats["top_p"] == 0.9
        assert stats["streaming"] is True
        assert stats["max_tokens"] == 3000

        print("✅ 完整统计信息测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
