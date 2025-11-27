"""
习题 4: 工具开发
实现实用的自定义工具

本文件实现了几个实用工具:
1. 文件操作工具 (读写文件)
2. HTTP API 调用工具
3. JSON 处理工具
4. 日期时间工具
5. 文本处理工具
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
from langchain_core.tools import BaseTool, tool


# ============================================================================
# 1. 文件操作工具
# ============================================================================

class FileReadTool(BaseTool):
    """读取文件内容的工具"""

    name: str = "read_file"
    description: str = """
    读取指定文件的内容。
    输入格式: 文件路径 (如: /path/to/file.txt)
    返回: 文件内容或错误信息
    """

    def _run(self, file_path: str) -> str:
        """读取文件"""
        try:
            path = Path(file_path).expanduser()

            if not path.exists():
                return f"❌ 文件不存在: {file_path}"

            if not path.is_file():
                return f"❌ 不是文件: {file_path}"

            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            return f"✅ 成功读取文件 ({len(content)} 字符):\n{content[:500]}"

        except Exception as e:
            return f"❌ 读取文件失败: {str(e)}"

    async def _arun(self, file_path: str) -> str:
        return self._run(file_path)


class FileWriteTool(BaseTool):
    """写入文件内容的工具"""

    name: str = "write_file"
    description: str = """
    将内容写入指定文件。
    输入格式: 文件路径::内容 (用::分隔路径和内容)
    示例: /tmp/test.txt::Hello World
    """

    def _run(self, input_str: str) -> str:
        """写入文件"""
        try:
            # 解析输入
            if "::" not in input_str:
                return "❌ 格式错误,请使用: 文件路径::内容"

            file_path, content = input_str.split("::", 1)
            path = Path(file_path.strip()).expanduser()

            # 创建父目录
            path.parent.mkdir(parents=True, exist_ok=True)

            # 写入文件
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content.strip())

            return f"✅ 成功写入文件: {path} ({len(content)} 字符)"

        except Exception as e:
            return f"❌ 写入文件失败: {str(e)}"

    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)


class FileListTool(BaseTool):
    """列出目录内容的工具"""

    name: str = "list_files"
    description: str = """
    列出指定目录下的文件和子目录。
    输入: 目录路径
    返回: 文件和目录列表
    """

    def _run(self, dir_path: str) -> str:
        """列出目录"""
        try:
            path = Path(dir_path).expanduser()

            if not path.exists():
                return f"❌ 目录不存在: {dir_path}"

            if not path.is_dir():
                return f"❌ 不是目录: {dir_path}"

            items = list(path.iterdir())
            files = [f.name for f in items if f.is_file()]
            dirs = [d.name for d in items if d.is_dir()]

            result = f"📁 目录: {dir_path}\n\n"
            if dirs:
                result += f"📂 子目录 ({len(dirs)}):\n"
                result += "\n".join(f"  - {d}" for d in sorted(dirs)[:20])
                if len(dirs) > 20:
                    result += f"\n  ... 还有 {len(dirs) - 20} 个"
                result += "\n\n"

            if files:
                result += f"📄 文件 ({len(files)}):\n"
                result += "\n".join(f"  - {f}" for f in sorted(files)[:20])
                if len(files) > 20:
                    result += f"\n  ... 还有 {len(files) - 20} 个"

            return result if (files or dirs) else "📂 空目录"

        except Exception as e:
            return f"❌ 列出目录失败: {str(e)}"

    async def _arun(self, dir_path: str) -> str:
        return self._run(dir_path)


# ============================================================================
# 2. HTTP API 调用工具
# ============================================================================

class HTTPGetTool(BaseTool):
    """HTTP GET 请求工具"""

    name: str = "http_get"
    description: str = """
    发送 HTTP GET 请求获取数据。
    输入: URL 地址
    返回: API 响应内容
    """

    def _run(self, url: str) -> str:
        """发送 GET 请求"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # 尝试解析 JSON
            try:
                data = response.json()
                return f"✅ 成功获取数据:\n{json.dumps(data, indent=2, ensure_ascii=False)[:1000]}"
            except:
                return f"✅ 成功获取数据:\n{response.text[:1000]}"

        except requests.exceptions.Timeout:
            return "❌ 请求超时"
        except requests.exceptions.RequestException as e:
            return f"❌ 请求失败: {str(e)}"
        except Exception as e:
            return f"❌ 错误: {str(e)}"

    async def _arun(self, url: str) -> str:
        return self._run(url)


@tool
def fetch_github_repo_info(repo: str) -> str:
    """
    获取 GitHub 仓库信息。

    输入仓库名称 (格式: owner/repo)
    返回仓库的基本信息,如星标数、fork数等。

    示例: langchain-ai/langchain
    """
    try:
        url = f"https://api.github.com/repos/{repo}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        result = f"""
📦 仓库: {data['full_name']}
📝 描述: {data.get('description', '无')}
⭐ Stars: {data['stargazers_count']}
🔱 Forks: {data['forks_count']}
👀 Watchers: {data['watchers_count']}
🐛 Issues: {data['open_issues_count']}
📅 创建时间: {data['created_at']}
🔄 最后更新: {data['updated_at']}
🔗 链接: {data['html_url']}
"""
        return result.strip()

    except Exception as e:
        return f"❌ 获取仓库信息失败: {str(e)}"


# ============================================================================
# 3. JSON 处理工具
# ============================================================================

@tool
def parse_json(json_str: str) -> str:
    """
    解析 JSON 字符串并格式化输出。

    输入: JSON 字符串
    返回: 格式化的 JSON 或错误信息
    """
    try:
        data = json.loads(json_str)
        formatted = json.dumps(data, indent=2, ensure_ascii=False)
        return f"✅ JSON 解析成功:\n{formatted}"
    except json.JSONDecodeError as e:
        return f"❌ JSON 解析失败: {str(e)}"
    except Exception as e:
        return f"❌ 错误: {str(e)}"


@tool
def extract_json_field(json_and_field: str) -> str:
    """
    从 JSON 中提取指定字段。

    输入格式: JSON字符串::字段路径
    字段路径支持点号分隔,如: data.user.name
    示例: {"data":{"user":{"name":"Tom"}}}::data.user.name
    """
    try:
        if "::" not in json_and_field:
            return "❌ 格式错误,请使用: JSON::字段路径"

        json_str, field_path = json_and_field.split("::", 1)
        data = json.loads(json_str)

        # 按路径提取
        result = data
        for field in field_path.strip().split('.'):
            result = result[field]

        return f"✅ 提取成功: {json.dumps(result, ensure_ascii=False)}"

    except (KeyError, IndexError) as e:
        return f"❌ 字段不存在: {e}"
    except json.JSONDecodeError as e:
        return f"❌ JSON 解析失败: {e}"
    except Exception as e:
        return f"❌ 错误: {str(e)}"


# ============================================================================
# 4. 日期时间工具
# ============================================================================

@tool
def get_current_datetime() -> str:
    """
    获取当前日期和时间。
    返回当前的日期、时间、星期等信息。
    """
    now = datetime.now()
    weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

    result = f"""
📅 当前日期: {now.strftime('%Y年%m月%d日')}
🕐 当前时间: {now.strftime('%H:%M:%S')}
📆 星期: {weekdays[now.weekday()]}
🌍 时区: {now.astimezone().strftime('%Z')}
⏰ 时间戳: {int(now.timestamp())}
"""
    return result.strip()


@tool
def calculate_date_diff(date_str: str) -> str:
    """
    计算日期差值。

    输入格式: YYYY-MM-DD (计算到今天的天数差)
    示例: 2024-01-01
    返回: 距离今天的天数
    """
    try:
        target_date = datetime.strptime(date_str.strip(), '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        diff = target_date - today
        days = diff.days

        if days > 0:
            return f"📅 {date_str} 是 {days} 天后 ({abs(days // 365)} 年 {abs(days % 365)} 天)"
        elif days < 0:
            return f"📅 {date_str} 是 {abs(days)} 天前 ({abs(days // 365)} 年 {abs(days % 365)} 天)"
        else:
            return f"📅 {date_str} 就是今天!"

    except ValueError:
        return "❌ 日期格式错误,请使用: YYYY-MM-DD"
    except Exception as e:
        return f"❌ 计算失败: {str(e)}"


@tool
def add_days_to_date(date_and_days: str) -> str:
    """
    给日期加上指定天数。

    输入格式: YYYY-MM-DD::天数
    示例: 2024-01-01::30
    返回: 计算后的日期
    """
    try:
        if "::" not in date_and_days:
            return "❌ 格式错误,请使用: YYYY-MM-DD::天数"

        date_str, days_str = date_and_days.split("::")
        date = datetime.strptime(date_str.strip(), '%Y-%m-%d')
        days = int(days_str.strip())

        new_date = date + timedelta(days=days)

        return f"📅 {date_str} + {days} 天 = {new_date.strftime('%Y-%m-%d')} ({new_date.strftime('%A')})"

    except ValueError as e:
        return f"❌ 格式错误: {str(e)}"
    except Exception as e:
        return f"❌ 计算失败: {str(e)}"


# ============================================================================
# 5. 文本处理工具
# ============================================================================

@tool
def count_words(text: str) -> str:
    """
    统计文本的字数、字符数等信息。
    输入: 任意文本
    返回: 统计信息
    """
    words = text.split()
    lines = text.split('\n')
    chars = len(text)
    chars_no_space = len(text.replace(' ', '').replace('\n', ''))

    result = f"""
📊 文本统计:
  - 总字符数: {chars}
  - 不含空格: {chars_no_space}
  - 单词数: {len(words)}
  - 行数: {len(lines)}
  - 段落数: {len([l for l in lines if l.strip()])}
"""
    return result.strip()


@tool
def text_transform(text_and_op: str) -> str:
    """
    文本转换工具。

    输入格式: 文本::操作
    支持的操作: upper(大写), lower(小写), title(标题), reverse(反转)
    示例: Hello World::upper
    """
    try:
        if "::" not in text_and_op:
            return "❌ 格式错误,请使用: 文本::操作"

        text, operation = text_and_op.split("::", 1)
        op = operation.strip().lower()

        if op == "upper":
            result = text.upper()
        elif op == "lower":
            result = text.lower()
        elif op == "title":
            result = text.title()
        elif op == "reverse":
            result = text[::-1]
        else:
            return f"❌ 不支持的操作: {operation}. 支持: upper, lower, title, reverse"

        return f"✅ 转换结果: {result}"

    except Exception as e:
        return f"❌ 转换失败: {str(e)}"


# ============================================================================
# 测试和演示
# ============================================================================

def demo_file_tools():
    """演示文件工具"""
    print("\n" + "=" * 60)
    print("演示 1: 文件操作工具")
    print("=" * 60)

    write_tool = FileWriteTool()
    read_tool = FileReadTool()
    list_tool = FileListTool()

    # 写入文件
    print("\n1. 写入文件:")
    result = write_tool.run("/tmp/test_langchain.txt::这是 LangChain 测试内容\nHello World!")
    print(result)

    # 读取文件
    print("\n2. 读取文件:")
    result = read_tool.run("/tmp/test_langchain.txt")
    print(result)

    # 列出目录
    print("\n3. 列出目录:")
    result = list_tool.run("/tmp")
    print(result[:500])  # 只显示前500字符


def demo_http_tools():
    """演示 HTTP 工具"""
    print("\n" + "=" * 60)
    print("演示 2: HTTP API 工具")
    print("=" * 60)

    # GitHub 仓库信息
    print("\n获取 GitHub 仓库信息:")
    result = fetch_github_repo_info.invoke("langchain-ai/langchain")
    print(result)


def demo_json_tools():
    """演示 JSON 工具"""
    print("\n" + "=" * 60)
    print("演示 3: JSON 处理工具")
    print("=" * 60)

    # 解析 JSON
    print("\n1. 解析 JSON:")
    json_str = '{"name":"LangChain","version":"1.0","features":["agents","chains","tools"]}'
    result = parse_json.invoke(json_str)
    print(result)

    # 提取字段
    print("\n2. 提取 JSON 字段:")
    result = extract_json_field.invoke(f'{json_str}::features')
    print(result)


def demo_datetime_tools():
    """演示日期时间工具"""
    print("\n" + "=" * 60)
    print("演示 4: 日期时间工具")
    print("=" * 60)

    # 当前时间
    print("\n1. 当前日期时间:")
    result = get_current_datetime.invoke("")
    print(result)

    # 日期差值
    print("\n2. 计算日期差值:")
    result = calculate_date_diff.invoke("2025-12-31")
    print(result)

    # 日期加减
    print("\n3. 日期加减:")
    result = add_days_to_date.invoke("2024-01-01::100")
    print(result)


def demo_text_tools():
    """演示文本工具"""
    print("\n" + "=" * 60)
    print("演示 5: 文本处理工具")
    print("=" * 60)

    # 统计字数
    print("\n1. 统计文本信息:")
    text = "LangChain is a framework for developing applications powered by language models.\nIt's awesome!"
    result = count_words.invoke(text)
    print(result)

    # 文本转换
    print("\n2. 文本转换:")
    result = text_transform.invoke("Hello LangChain World::upper")
    print(result)


def demo_agent_with_custom_tools():
    """演示: Agent 使用自定义工具"""
    print("\n" + "=" * 60)
    print("演示 6: Agent 使用自定义工具")
    print("=" * 60)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    try:
        from core.utils import setup_llm
        from agents.react_agent_langchain import ReActAgent

        # 创建工具列表
        tools = [
            get_current_datetime,
            calculate_date_diff,
            count_words,
            fetch_github_repo_info,
        ]

        # 创建 Agent
        llm = setup_llm(model="glm-4-flash")
        agent = ReActAgent(
            name="工具专家",
            llm=llm,
            tools=tools,
            max_iterations=5
        )

        # 测试任务
        print("\n💬 任务: 获取 langchain-ai/langchain 的信息")
        result = agent.run("获取 GitHub 仓库 langchain-ai/langchain 的信息")
        print(f"\n结果:\n{result}")

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("  习题 4: 自定义工具开发与测试")
    print("=" * 60)

    # 运行各个演示
    demo_file_tools()
    demo_http_tools()
    demo_json_tools()
    demo_datetime_tools()
    demo_text_tools()

    # Agent 演示
    demo_agent_with_custom_tools()

    print("\n" + "=" * 60)
    print("  所有演示完成")
    print("=" * 60)
    print("""
💡 扩展思考:
1. 如何实现工具的权限控制?
2. 如何实现工具的错误恢复机制?
3. 如何实现工具的缓存以提升性能?
4. 如何实现工具的链式调用?
5. 如何设计一个工具市场,让用户可以分享和下载工具?
    """)
