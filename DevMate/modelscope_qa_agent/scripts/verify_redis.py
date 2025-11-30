"""
验证 Redis 缓存服务连接

测试与 Redis 服务器的连接是否正常。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import redis
from config.settings import settings


def verify_redis_connection() -> bool:
    """验证 Redis 连接

    Returns:
        bool: 连接是否成功
    """
    try:
        print("=" * 70)
        print("Redis 连接验证")
        print("=" * 70)

        # 构建连接参数
        print(f"正在连接到 Redis: {settings.redis_host}:{settings.redis_port}")
        print(f"数据库: {settings.redis_db}")

        # 创建 Redis 客户端
        redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password if settings.redis_password else None,
            decode_responses=True,
            socket_connect_timeout=5
        )

        # 测试连接
        response = redis_client.ping()
        if not response:
            print("❌ 连接失败: PING 命令无响应")
            return False

        print(f"✅ 连接成功!")
        print(f"   PING 响应: {response}")

        # 获取服务器信息
        info = redis_client.info("server")
        print(f"   Redis 版本: {info.get('redis_version', 'Unknown')}")
        print(f"   运行模式: {info.get('redis_mode', 'Unknown')}")
        print(f"   操作系统: {info.get('os', 'Unknown')}")

        # 测试基本操作
        test_key = "modelscope_qa:test:connection"
        redis_client.set(test_key, "test_value", ex=10)
        value = redis_client.get(test_key)

        if value == "test_value":
            print(f"   读写测试: ✅ 成功")
            redis_client.delete(test_key)
        else:
            print(f"   读写测试: ❌ 失败")

        # 关闭连接
        redis_client.close()

        print("\n✅ Redis 连接验证通过")
        print("=" * 70)
        return True

    except redis.ConnectionError as e:
        print(f"❌ 连接失败: {e}")
        print(f"\n请检查:")
        print(f"   1. Redis 服务是否已启动")
        print(f"   2. 连接地址是否正确: {settings.redis_host}:{settings.redis_port}")
        print(f"   3. 密码配置是否正确")
        print(f"   4. 防火墙是否允许连接")
        print("=" * 70)
        return False
    except redis.AuthenticationError as e:
        print(f"❌ 认证失败: {e}")
        print(f"\n请检查 Redis 密码配置")
        print("=" * 70)
        return False
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = verify_redis_connection()
    sys.exit(0 if success else 1)
