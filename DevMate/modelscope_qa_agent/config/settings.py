"""
配置管理模块

从环境变量加载配置,提供类型安全的配置访问。
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """应用配置类"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # 通义千问 API 配置
    dashscope_api_key: str = Field(default="", description="通义千问 API Key")

    # Milvus 配置
    milvus_host: str = Field(default="localhost", description="Milvus 服务器地址")
    milvus_port: int = Field(default=19530, description="Milvus 服务器端口")

    # Redis 配置
    redis_host: str = Field(default="localhost", description="Redis 服务器地址")
    redis_port: int = Field(default=6379, description="Redis 服务器端口")
    redis_db: int = Field(default=0, description="Redis 数据库编号")
    redis_password: Optional[str] = Field(default=None, description="Redis 密码")

    # MySQL 配置 (可选)
    mysql_host: str = Field(default="localhost", description="MySQL 服务器地址")
    mysql_port: int = Field(default=3309, description="MySQL 服务器端口")
    mysql_user: str = Field(default="root", description="MySQL 用户名")
    mysql_password: Optional[str] = Field(default=None, description="MySQL 密码")
    mysql_database: str = Field(default="modelscope_qa", description="MySQL 数据库名")

    # LangSmith 配置 (可选)
    langchain_tracing_v2: bool = Field(default=False, description="启用 LangSmith 追踪")
    langchain_api_key: Optional[str] = Field(default=None, description="LangSmith API Key")
    langchain_project: str = Field(default="modelscope-qa-agent", description="LangSmith 项目名")

    # 应用配置
    app_env: str = Field(default="development", description="应用环境")
    log_level: str = Field(default="INFO", description="日志级别")
    debug: bool = Field(default=False, description="调试模式")

    # Agent 配置
    max_conversation_turns: int = Field(default=10, description="最大对话轮数")
    context_window_size: int = Field(default=4000, description="上下文窗口大小")
    default_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="默认温度参数")
    default_top_p: float = Field(default=0.8, ge=0.0, le=1.0, description="默认 Top-P 参数")

    # 检索配置
    retrieval_top_k: int = Field(default=3, ge=1, le=20, description="检索返回文档数")
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="向量检索权重")
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="BM25 检索权重")
    min_confidence_score: float = Field(default=0.7, ge=0.0, le=1.0, description="最小置信度阈值")

    # 缓存配置
    enable_cache: bool = Field(default=True, description="启用缓存")
    cache_ttl: int = Field(default=3600, description="缓存过期时间(秒)")

    @property
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.app_env.lower() == "production"

    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.app_env.lower() == "development"

    def validate_api_keys(self) -> list[str]:
        """验证必需的 API 密钥是否已配置

        Returns:
            list[str]: 缺失的配置项列表
        """
        missing = []

        if not self.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY")

        if self.langchain_tracing_v2 and not self.langchain_api_key:
            missing.append("LANGCHAIN_API_KEY")

        return missing


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例

    Returns:
        Settings: 配置实例
    """
    return settings


# 验证配置
if __name__ == "__main__":
    print("=" * 70)
    print("配置验证")
    print("=" * 70)

    missing_keys = settings.validate_api_keys()
    if missing_keys:
        print(f"⚠️  缺失的配置项: {', '.join(missing_keys)}")
    else:
        print("✅ 所有必需的 API 密钥已配置")

    print(f"\n环境: {settings.app_env}")
    print(f"调试模式: {settings.debug}")
    print(f"日志级别: {settings.log_level}")
    print(f"\nMilvus: {settings.milvus_host}:{settings.milvus_port}")
    print(f"Redis: {settings.redis_host}:{settings.redis_port}")
    print(f"LangSmith 追踪: {settings.langchain_tracing_v2}")
    print("=" * 70)
