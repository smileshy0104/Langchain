"""
YAML 配置加载器

从 YAML 文件加载配置,提供类型安全的配置访问。
支持多种 AI 服务提供商 (VolcEngine, OpenAI, DashScope) 和存储后端 (MinIO, Local, OSS, S3)。
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class AIConfig(BaseModel):
    """AI 服务配置"""
    provider: str = Field(description="AI 服务提供商: volcengine, openai, dashscope")
    api_key: str = Field(description="API 密钥")
    base_url: Optional[str] = Field(default=None, description="API 基础 URL")
    models: Dict[str, str] = Field(description="模型配置 (embedding, chat)")
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 4000
        },
        description="模型参数"
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """验证 provider"""
        allowed = ["volcengine", "openai", "dashscope"]
        if v not in allowed:
            raise ValueError(f"provider 必须是 {allowed} 之一, 收到: {v}")
        return v


class LocalStorageConfig(BaseModel):
    """本地存储配置"""
    upload_path: str = Field(default="./uploads", description="上传文件存储路径")


class MinIOStorageConfig(BaseModel):
    """MinIO 存储配置"""
    endpoint: str = Field(description="MinIO 服务地址")
    access_key: str = Field(description="访问密钥")
    secret_key: str = Field(description="私钥")
    bucket_name: str = Field(description="存储桶名称")
    use_ssl: bool = Field(default=False, description="是否使用 SSL")


class StorageConfig(BaseModel):
    """存储配置"""
    type: str = Field(description="存储类型: local, minio, oss, s3")
    local: Optional[LocalStorageConfig] = Field(default=None)
    minio: Optional[MinIOStorageConfig] = Field(default=None)
    max_file_size: str = Field(default="100MB", description="最大文件大小")
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".md", ".json", ".xml", ".html", ".rtf", ".xlsx", ".pptx"],
        description="允许的文件扩展名"
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """验证 storage type"""
        allowed = ["local", "minio", "oss", "s3"]
        if v not in allowed:
            raise ValueError(f"storage.type 必须是 {allowed} 之一, 收到: {v}")
        return v

    def parse_max_file_size(self) -> int:
        """解析最大文件大小为字节数

        Returns:
            int: 文件大小 (字节)
        """
        size_str = self.max_file_size.upper()

        # 提取数字和单位
        import re
        match = re.match(r"(\d+(?:\.\d+)?)\s*([KMGT]?B)", size_str)
        if not match:
            raise ValueError(f"无法解析文件大小: {self.max_file_size}")

        value = float(match.group(1))
        unit = match.group(2)

        # 转换为字节
        multipliers = {
            "B": 1,
            "KB": 1024,
            "MB": 1024 ** 2,
            "GB": 1024 ** 3,
            "TB": 1024 ** 4
        }

        return int(value * multipliers[unit])


class MilvusConfig(BaseModel):
    """Milvus 向量数据库配置"""
    host: str = Field(default="localhost")
    port: int = Field(default=19530)
    collection_name: str = Field(default="modelscope_docs")
    vector_dim: int = Field(default=1024, description="向量维度")


class RedisConfig(BaseModel):
    """Redis 缓存配置"""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)


class RetrievalConfig(BaseModel):
    """检索配置"""
    top_k: int = Field(default=3, ge=1, le=20)
    min_confidence_score: float = Field(default=0.7, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)


class ClarificationConfig(BaseModel):
    """主动澄清配置"""
    enabled: bool = Field(default=True, description="是否启用主动澄清")
    threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="触发澄清的置信度阈值")
    max_attempts: int = Field(default=2, ge=1, description="最大澄清尝试次数")


class AgentConfig(BaseModel):
    """Agent 配置"""
    max_conversation_turns: int = Field(default=10, ge=1)
    context_window_size: int = Field(default=4000, ge=100)
    progress_threshold: int = Field(default=5, ge=1, description="进度评估触发轮数")
    clarification: ClarificationConfig = Field(default_factory=ClarificationConfig)


class AppConfig(BaseModel):
    """应用配置"""
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别"""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level 必须是 {allowed} 之一")
        return v_upper


class SessionConfig(BaseModel):
    """会话配置"""
    ttl: int = Field(default=3600, ge=60, description="会话过期时间(秒)")
    max_sessions_per_user: int = Field(default=5, ge=1, description="每个用户最大会话数")
    cleanup_interval: int = Field(default=300, ge=60, description="清理过期会话的间隔(秒)")


class LangSmithConfig(BaseModel):
    """LangSmith 配置"""
    enabled: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None)
    project: str = Field(default="modelscope-qa-agent")


class Config(BaseModel):
    """完整配置"""
    ai: AIConfig
    storage: StorageConfig
    milvus: MilvusConfig
    redis: RedisConfig
    retrieval: RetrievalConfig
    agent: AgentConfig
    session: SessionConfig
    app: AppConfig
    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Config":
        """从 YAML 文件加载配置

        Args:
            config_path: 配置文件路径 (相对于项目根目录)

        Returns:
            Config: 配置实例

        Example:
            >>> config = Config.from_yaml("config.yaml")
            >>> print(config.ai.provider)
            volcengine
        """
        # 获取项目根目录
        project_root = Path(__file__).parent.parent
        full_path = project_root / config_path

        if not full_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {full_path}")

        # 读取 YAML 文件
        with open(full_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        # 创建配置实例
        return cls(**yaml_data)

    def validate_required_fields(self) -> List[str]:
        """验证必需字段是否已配置

        Returns:
            List[str]: 缺失的配置项列表
        """
        missing = []

        # 检查 AI API Key
        if not self.ai.api_key or self.ai.api_key.startswith("your-"):
            missing.append("ai.api_key")

        # 检查存储配置
        if self.storage.type == "minio" and not self.storage.minio:
            missing.append("storage.minio")
        elif self.storage.type == "local" and not self.storage.local:
            missing.append("storage.local")

        # 检查 LangSmith (如果启用)
        if self.langsmith.enabled and not self.langsmith.api_key:
            missing.append("langsmith.api_key")

        return missing

    def get_embedding_model_name(self) -> str:
        """获取 Embedding 模型名称"""
        return self.ai.models.get("embedding", "")

    def get_chat_model_name(self) -> str:
        """获取 Chat 模型名称"""
        return self.ai.models.get("chat", "")

    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.app.environment.lower() == "production"

    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.app.environment.lower() == "development"


# 全局配置实例
_config: Optional[Config] = None


def load_config(config_path: str = "config.yaml") -> Config:
    """加载配置 (单例模式)

    Args:
        config_path: 配置文件路径

    Returns:
        Config: 配置实例
    """
    global _config

    if _config is None:
        _config = Config.from_yaml(config_path)

        # 验证必需字段
        missing = _config.validate_required_fields()
        if missing:
            print(f"⚠️  警告: 以下配置项缺失或无效: {', '.join(missing)}")

    return _config


def get_config() -> Config:
    """获取全局配置实例

    Returns:
        Config: 配置实例

    Raises:
        RuntimeError: 配置未加载
    """
    if _config is None:
        raise RuntimeError("配置未加载,请先调用 load_config()")
    return _config


# 便捷函数

def reload_config(config_path: str = "config.yaml") -> Config:
    """重新加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        Config: 新的配置实例
    """
    global _config
    _config = None
    return load_config(config_path)


# 示例用法
if __name__ == "__main__":
    print("=" * 70)
    print("配置加载测试")
    print("=" * 70)

    try:
        # 加载配置
        config = load_config("config.yaml")

        print("\n✅ 配置加载成功!\n")

        # 打印 AI 配置
        print("AI 服务配置:")
        print(f"  提供商: {config.ai.provider}")
        print(f"  Base URL: {config.ai.base_url}")
        print(f"  Embedding 模型: {config.get_embedding_model_name()}")
        print(f"  Chat 模型: {config.get_chat_model_name()}")
        print(f"  Temperature: {config.ai.parameters['temperature']}")

        # 打印存储配置
        print(f"\n存储配置:")
        print(f"  类型: {config.storage.type}")
        print(f"  最大文件大小: {config.storage.max_file_size} ({config.storage.parse_max_file_size()} 字节)")
        print(f"  允许的扩展名: {', '.join(config.storage.allowed_extensions)}")

        if config.storage.type == "minio" and config.storage.minio:
            print(f"  MinIO 端点: {config.storage.minio.endpoint}")
            print(f"  存储桶: {config.storage.minio.bucket_name}")

        # 打印 Milvus 配置
        print(f"\nMilvus 配置:")
        print(f"  地址: {config.milvus.host}:{config.milvus.port}")
        print(f"  Collection: {config.milvus.collection_name}")
        print(f"  向量维度: {config.milvus.vector_dim}")

        # 打印检索配置
        print(f"\n检索配置:")
        print(f"  Top-K: {config.retrieval.top_k}")
        print(f"  最小置信度: {config.retrieval.min_confidence_score}")

        # 打印应用配置
        print(f"\n应用配置:")
        print(f"  环境: {config.app.environment}")
        print(f"  调试模式: {config.app.debug}")
        print(f"  日志级别: {config.app.log_level}")

        # 验证必需字段
        missing = config.validate_required_fields()
        if missing:
            print(f"\n⚠️  缺失的配置项: {', '.join(missing)}")
        else:
            print("\n✅ 所有必需的配置项都已设置")

    except Exception as e:
        print(f"\n❌ 配置加载失败: {e}")
        raise

    print("\n" + "=" * 70)
