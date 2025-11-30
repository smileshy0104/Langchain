"""
验证 MySQL 数据库连接 (可选)

测试与 MySQL 服务器的连接是否正常。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pymysql
from config.settings import settings


def verify_mysql_connection() -> bool:
    """验证 MySQL 连接

    Returns:
        bool: 连接是否成功
    """
    try:
        print("=" * 70)
        print("MySQL 连接验证 (可选)")
        print("=" * 70)

        # 构建连接参数
        print(f"正在连接到 MySQL: {settings.mysql_host}:{settings.mysql_port}")
        print(f"用户: {settings.mysql_user}")
        print(f"数据库: {settings.mysql_database}")

        # 创建连接
        connection = pymysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password if settings.mysql_password else "",
            database=settings.mysql_database if settings.mysql_database else None,
            connect_timeout=5,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )

        print(f"✅ 连接成功!")

        # 获取服务器信息
        with connection.cursor() as cursor:
            # MySQL 版本
            cursor.execute("SELECT VERSION()")
            version_result = cursor.fetchone()
            mysql_version = version_result['VERSION()'] if version_result else 'Unknown'
            print(f"   MySQL 版本: {mysql_version}")

            # 当前数据库
            cursor.execute("SELECT DATABASE()")
            db_result = cursor.fetchone()
            current_db = db_result['DATABASE()'] if db_result else 'None'
            print(f"   当前数据库: {current_db}")

            # 测试数据库是否存在
            if settings.mysql_database:
                cursor.execute(
                    "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = %s",
                    (settings.mysql_database,)
                )
                db_exists = cursor.fetchone()

                if not db_exists:
                    print(f"   ⚠️  数据库 '{settings.mysql_database}' 不存在,需要创建")
                else:
                    print(f"   ✅ 数据库 '{settings.mysql_database}' 已存在")

                    # 显示表数量
                    cursor.execute(
                        "SELECT COUNT(*) as count FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = %s",
                        (settings.mysql_database,)
                    )
                    table_count_result = cursor.fetchone()
                    table_count = table_count_result['count'] if table_count_result else 0
                    print(f"   表数量: {table_count}")

        # 关闭连接
        connection.close()

        print("\n✅ MySQL 连接验证通过")
        print("=" * 70)
        return True

    except pymysql.err.OperationalError as e:
        error_code, error_msg = e.args
        print(f"❌ 连接失败 (错误代码: {error_code}): {error_msg}")
        print(f"\n请检查:")
        print(f"   1. MySQL 服务是否已启动")
        print(f"   2. 连接地址是否正确: {settings.mysql_host}:{settings.mysql_port}")
        print(f"   3. 用户名密码是否正确")
        print(f"   4. 用户是否有访问数据库的权限")
        print(f"   5. 防火墙是否允许连接")
        print("=" * 70)
        return False
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = verify_mysql_connection()
    sys.exit(0 if success else 1)
