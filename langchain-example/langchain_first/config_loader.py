"""
环境变量配置加载器
用于从 .env 文件加载 API 配置
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, env_path: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            env_path: .env 文件路径，默认为当前目录下的 .env
        """
        if env_path is None:
            # 默认使用当前文件所在目录的 .env
            current_dir = Path(__file__).parent
            env_path = current_dir / ".env"
        
        self.env_path = Path(env_path)
        self._load_env()
    
    def _load_env(self):
        """加载 .env 文件"""
        if not self.env_path.exists():
            print(f"⚠️  .env 文件不存在: {self.env_path}")
            print(f"📝 请复制 .env.example 为 .env 并填写 API Key")
            print(f"\n命令: cp {self.env_path.parent}/.env.example {self.env_path}")
            raise FileNotFoundError(f".env 文件不存在: {self.env_path}")
        
        # 加载环境变量
        load_dotenv(self.env_path, override=True)
        print(f"✅ 已加载配置文件: {self.env_path}")
    
    def get_api_key(self, provider: str) -> str:
        """
        获取指定提供商的 API Key
        
        Args:
            provider: 提供商名称，如 'zhipuai', 'openai', 'anthropic', 'custom'
        
        Returns:
            API Key 字符串
        
        Raises:
            ValueError: 如果 API Key 未配置或为默认值
        """
        env_var_map = {
            'zhipuai': 'ZHIPUAI_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'custom': 'CUSTOM_MODEL_API_KEY',
        }
        
        if provider not in env_var_map:
            raise ValueError(f"未知的提供商: {provider}")
        
        env_var = env_var_map[provider]
        api_key = os.getenv(env_var, '')
        
        # 检查是否为空或默认值
        if not api_key or 'your-' in api_key.lower():
            raise ValueError(
                f"❌ 请在 {self.env_path} 中配置 {env_var}\n"
                f"当前值: {api_key or '(未设置)'}"
            )
        
        return api_key
    
    def get_model_config(self, provider: str) -> Dict[str, Any]:
        """
        获取指定提供商的完整模型配置
        
        Args:
            provider: 提供商名称
        
        Returns:
            包含 api_key, model, temperature 等配置的字典
        """
        config = {
            'api_key': self.get_api_key(provider)
        }
        
        # 根据不同提供商获取配置
        if provider == 'zhipuai':
            config['model'] = os.getenv('ZHIPUAI_MODEL', 'glm-4-flash')
            config['temperature'] = float(os.getenv('ZHIPUAI_TEMPERATURE', '0.7'))
        
        elif provider == 'openai':
            config['model'] = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            config['temperature'] = float(os.getenv('OPENAI_TEMPERATURE', '0.5'))
            config['max_tokens'] = int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
        
        elif provider == 'anthropic':
            config['model'] = os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')
            config['temperature'] = float(os.getenv('ANTHROPIC_TEMPERATURE', '0.7'))
            config['max_tokens'] = int(os.getenv('ANTHROPIC_MAX_TOKENS', '1024'))

        elif provider == 'custom':
            config['model'] = os.getenv('CUSTOM_MODEL_NAME', 'gpt-4o-mini')
            config['base_url'] = os.getenv('CUSTOM_MODEL_BASE_URL', '').strip()
            config['temperature'] = float(os.getenv('CUSTOM_MODEL_TEMPERATURE', '0.7'))

            max_tokens = os.getenv('CUSTOM_MODEL_MAX_TOKENS', '').strip()
            if max_tokens:
                config['max_tokens'] = int(max_tokens)

            if not config['base_url'] or 'your-' in config['base_url'].lower():
                raise ValueError(
                    f"❌ 请在 {self.env_path} 中配置 CUSTOM_MODEL_BASE_URL\n"
                    f"当前值: {config['base_url'] or '(未设置)'}"
                )
        
        return config
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        获取通用设置
        
        Args:
            key: 设置键名（大写，如 'DEFAULT_PROVIDER'）
            default: 默认值
        
        Returns:
            设置值
        """
        return os.getenv(key, default)


# ==================== 便捷函数 ====================

def load_config(env_path: Optional[str] = None) -> ConfigLoader:
    """
    加载配置文件（便捷函数）
    
    Args:
        env_path: .env 文件路径
    
    Returns:
        ConfigLoader 实例
    """
    return ConfigLoader(env_path)


def get_zhipuai_config() -> Dict[str, Any]:
    """获取智谱 AI 配置"""
    loader = load_config()
    return loader.get_model_config('zhipuai')


def get_openai_config() -> Dict[str, Any]:
    """获取 OpenAI 配置"""
    loader = load_config()
    return loader.get_model_config('openai')


def get_anthropic_config() -> Dict[str, Any]:
    """获取 Anthropic 配置"""
    loader = load_config()
    return loader.get_model_config('anthropic')


def get_custom_model_config() -> Dict[str, Any]:
    """获取自定义 OpenAI-compatible 模型配置"""
    loader = load_config()
    return loader.get_model_config('custom')


def create_chat_model(provider: Optional[str] = None):
    """
    根据配置创建聊天模型。

    - zhipuai: 使用 langchain_community.chat_models.ChatZhipuAI
    - custom: 使用 langchain_openai.ChatOpenAI，支持自定义 base_url 和 api_key
    """
    loader = load_config()
    provider = provider or loader.get_setting('DEFAULT_PROVIDER', 'zhipuai')
    provider = provider.lower()

    if provider == 'zhipuai':
        from langchain_community.chat_models import ChatZhipuAI

        return ChatZhipuAI(**loader.get_model_config('zhipuai'))

    if provider == 'custom':
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**loader.get_model_config('custom'))

    raise ValueError(
        f"暂不支持 provider={provider!r}。"
        "当前 create_chat_model 支持: zhipuai, custom"
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("配置加载器示例")
    print("=" * 60)
    
    try:
        # 加载配置
        config = load_config()
        
        # 获取智谱 AI 配置
        print("\n📦 智谱 AI 配置:")
        zhipuai_config = config.get_model_config('zhipuai')
        print(f"  ├─ 模型: {zhipuai_config['model']}")
        print(f"  ├─ 温度: {zhipuai_config['temperature']}")
        print(f"  └─ API Key: {zhipuai_config['api_key'][:10]}...{zhipuai_config['api_key'][-4:]}")
        
        # 获取通用设置
        print("\n⚙️  通用设置:")
        default_provider = config.get_setting('DEFAULT_PROVIDER', 'zhipuai')
        timeout = config.get_setting('REQUEST_TIMEOUT', '30')
        print(f"  ├─ 默认提供商: {default_provider}")
        print(f"  └─ 请求超时: {timeout}秒")
        
        print("\n✅ 配置加载成功！")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        
    except ValueError as e:
        print(f"\n{e}")
        print("\n📝 配置步骤:")
        print("1. 复制 .env.example 为 .env")
        print("   命令: cp .env.example .env")
        print("2. 编辑 .env 文件，填写您的 API Key")
        print("3. 保存文件后重新运行")
        
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
