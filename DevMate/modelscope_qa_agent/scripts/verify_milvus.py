"""
验证 Milvus 向量数据库连接

测试与 Milvus 服务器的连接是否正常。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymilvus import connections, utility
from config.settings import settings


def verify_milvus_connection() -> bool:
    """验证 Milvus 连接

    Returns:
        bool: 连接是否成功
    """
    try:
        print("=" * 70)
        print("Milvus 连接验证")
        print("=" * 70)

        # 尝试连接
        print(f"正在连接到 Milvus: {settings.milvus_host}:{settings.milvus_port}")
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port
        )

        # 检查连接状态
        if not connections.has_connection("default"):
            print("❌ 连接失败: 无法建立连接")
            return False

        # 获取服务器版本
        version = utility.get_server_version()
        print(f"✅ 连接成功!")
        print(f"   Milvus 版本: {version}")

        # 列出所有集合
        collections = utility.list_collections()
        print(f"   当前集合数: {len(collections)}")
        if collections:
            print(f"   集合列表: {', '.join(collections)}")

        # 断开连接
        connections.disconnect("default")
        print("\n✅ Milvus 连接验证通过")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        print(f"\n请检查:")
        print(f"   1. Milvus 服务是否已启动")
        print(f"   2. 连接地址是否正确: {settings.milvus_host}:{settings.milvus_port}")
        print(f"   3. 防火墙是否允许连接")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = verify_milvus_connection()
    sys.exit(0 if success else 1)
