"""
基础设施健康检查脚本

检查所有必需和可选的基础设施服务是否正常运行。
包括: Milvus, Redis, MySQL (可选)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from enum import Enum

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymilvus import connections, utility
import redis
import pymysql
from config.settings import settings


class ServiceStatus(Enum):
    """服务状态枚举"""
    HEALTHY = "✅ 正常"
    UNHEALTHY = "❌ 异常"
    SKIPPED = "⏭️  跳过"
    WARNING = "⚠️  警告"


class InfrastructureChecker:
    """基础设施健康检查器"""

    def __init__(self):
        self.results: Dict[str, Tuple[ServiceStatus, str]] = {}

    def check_milvus(self) -> Tuple[ServiceStatus, str]:
        """检查 Milvus 向量数据库"""
        try:
            connections.connect(
                alias="milvus_health_check",
                host=settings.milvus_host,
                port=settings.milvus_port,
                timeout=5
            )

            if not connections.has_connection("milvus_health_check"):
                return ServiceStatus.UNHEALTHY, "无法建立连接"

            version = utility.get_server_version()
            connections.disconnect("milvus_health_check")

            return ServiceStatus.HEALTHY, f"版本 {version}"

        except Exception as e:
            error_msg = str(e)
            # 检测特定错误类型
            if "connection refused" in error_msg.lower():
                return ServiceStatus.UNHEALTHY, "连接被拒绝,服务可能未启动"
            elif "timeout" in error_msg.lower():
                return ServiceStatus.UNHEALTHY, "连接超时"
            else:
                return ServiceStatus.UNHEALTHY, f"{type(e).__name__}: {str(e)[:50]}"

    def check_redis(self) -> Tuple[ServiceStatus, str]:
        """检查 Redis 缓存服务"""
        try:
            redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password if settings.redis_password else None,
                socket_connect_timeout=5,
                decode_responses=True
            )

            if not redis_client.ping():
                return ServiceStatus.UNHEALTHY, "PING 失败"

            info = redis_client.info("server")
            version = info.get('redis_version', 'Unknown')
            redis_client.close()

            return ServiceStatus.HEALTHY, f"版本 {version}"

        except redis.ConnectionError:
            return ServiceStatus.UNHEALTHY, "连接失败,服务可能未启动"
        except redis.AuthenticationError:
            return ServiceStatus.UNHEALTHY, "认证失败,密码错误"
        except Exception as e:
            return ServiceStatus.UNHEALTHY, f"{type(e).__name__}: {str(e)[:50]}"

    def check_mysql(self) -> Tuple[ServiceStatus, str]:
        """检查 MySQL 数据库 (可选)"""
        # 如果没有配置密码,跳过检查
        if not settings.mysql_password:
            return ServiceStatus.SKIPPED, "未配置密码 (可选服务)"

        try:
            connection = pymysql.connect(
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                connect_timeout=5,
                charset='utf8mb4'
            )

            with connection.cursor() as cursor:
                cursor.execute("SELECT VERSION()")
                result = cursor.fetchone()
                version = result[0] if result else 'Unknown'

                # 检查数据库是否存在
                if settings.mysql_database:
                    cursor.execute(
                        "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = %s",
                        (settings.mysql_database,)
                    )
                    db_exists = cursor.fetchone()

                    if not db_exists:
                        connection.close()
                        return ServiceStatus.WARNING, f"版本 {version}, 数据库 '{settings.mysql_database}' 不存在"

            connection.close()
            return ServiceStatus.HEALTHY, f"版本 {version}"

        except pymysql.err.OperationalError as e:
            error_code, error_msg = e.args
            if error_code == 1045:
                return ServiceStatus.UNHEALTHY, "认证失败"
            elif error_code == 2003:
                return ServiceStatus.UNHEALTHY, "连接失败,服务可能未启动"
            else:
                return ServiceStatus.UNHEALTHY, f"错误 {error_code}: {error_msg[:50]}"
        except Exception as e:
            return ServiceStatus.UNHEALTHY, f"{type(e).__name__}: {str(e)[:50]}"

    def check_api_keys(self) -> Tuple[ServiceStatus, str]:
        """检查必需的 API 密钥配置"""
        missing = settings.validate_api_keys()

        if missing:
            return ServiceStatus.WARNING, f"缺失: {', '.join(missing)}"
        return ServiceStatus.HEALTHY, "所有必需密钥已配置"

    def run_all_checks(self) -> Dict[str, Tuple[ServiceStatus, str]]:
        """运行所有健康检查"""
        print("=" * 70)
        print("基础设施健康检查")
        print("=" * 70)
        print()

        checks = [
            ("API 密钥", self.check_api_keys, True),
            ("Milvus 向量数据库", self.check_milvus, True),
            ("Redis 缓存", self.check_redis, True),
            ("MySQL 数据库 (可选)", self.check_mysql, False),
        ]

        for service_name, check_func, is_required in checks:
            print(f"检查 {service_name}...", end=" ")
            sys.stdout.flush()

            status, message = check_func()
            self.results[service_name] = (status, message)

            print(f"{status.value}")
            print(f"   {message}")
            print()

        return self.results

    def print_summary(self):
        """打印检查摘要"""
        print("=" * 70)
        print("检查摘要")
        print("=" * 70)

        healthy_count = sum(
            1 for status, _ in self.results.values()
            if status == ServiceStatus.HEALTHY
        )
        total_count = len([k for k in self.results.keys() if "可选" not in k])

        # 统计各状态数量
        status_counts = {
            ServiceStatus.HEALTHY: 0,
            ServiceStatus.UNHEALTHY: 0,
            ServiceStatus.WARNING: 0,
            ServiceStatus.SKIPPED: 0,
        }

        for status, _ in self.results.values():
            status_counts[status] += 1

        print(f"健康: {status_counts[ServiceStatus.HEALTHY]}")
        print(f"异常: {status_counts[ServiceStatus.UNHEALTHY]}")
        print(f"警告: {status_counts[ServiceStatus.WARNING]}")
        print(f"跳过: {status_counts[ServiceStatus.SKIPPED]}")
        print()

        # 检查是否所有必需服务都正常
        required_services = ["Milvus 向量数据库", "Redis 缓存"]
        all_required_healthy = all(
            self.results.get(svc, (ServiceStatus.UNHEALTHY, ""))[0] == ServiceStatus.HEALTHY
            for svc in required_services
        )

        api_key_status = self.results.get("API 密钥", (ServiceStatus.UNHEALTHY, ""))[0]
        has_api_key_warning = api_key_status == ServiceStatus.WARNING

        if all_required_healthy and not has_api_key_warning:
            print("✅ 所有必需服务运行正常,系统可以启动")
            return 0
        elif all_required_healthy and has_api_key_warning:
            print("⚠️  所有必需服务运行正常,但 API 密钥未完全配置")
            print("   请在 .env 文件中配置缺失的 API 密钥后再使用")
            return 1
        else:
            print("❌ 部分必需服务异常,请修复后再启动系统")
            return 1


def main():
    """主函数"""
    checker = InfrastructureChecker()
    checker.run_all_checks()
    exit_code = checker.print_summary()
    print("=" * 70)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
