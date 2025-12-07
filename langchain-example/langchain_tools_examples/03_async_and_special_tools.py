"""
LangChain Tools - 异步工具和特殊类型工具示例
演示异步工具、流式工具、Retriever工具、数据库工具等
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool, BaseTool, StructuredTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Type, List, Dict, AsyncIterator, Iterator
import asyncio
import os
import time

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 异步工具基础 ====================

@tool
async def async_fetch_data(url: str) -> str:
    """异步获取数据

    Args:
        url: 数据源 URL
    """
    print(f"  开始异步获取: {url}")
    await asyncio.sleep(1)  # 模拟网络请求
    return f"从 {url} 获取的数据"


@tool
def sync_process_data(data: str) -> str:
    """同步处理数据

    Args:
        data: 要处理的数据
    """
    print(f"  处理数据: {data[:50]}...")
    time.sleep(0.5)  # 模拟处理时间
    return f"已处理: {data}"


async def async_tool_example():
    """异步工具示例"""
    print("=" * 50)
    print("异步工具示例")
    print("=" * 50)

    # 并行调用多个异步工具
    urls = [
        "https://api.example.com/data1",
        "https://api.example.com/data2",
        "https://api.example.com/data3"
    ]

    print("\n并行获取多个数据源...")
    tasks = [
        async_fetch_data.ainvoke({"url": url})
        for url in urls
    ]

    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}: {result}")


# ==================== 2. 混合同步异步工具 ====================

class DataProcessor(BaseTool):
    """数据处理工具（支持同步和异步）"""

    name: str = "process_data"
    description: str = "处理数据，支持同步和异步模式"

    def _run(self, data: str) -> str:
        """同步处理"""
        print(f"  同步处理: {data}")
        time.sleep(0.5)
        return f"同步结果: {data.upper()}"

    async def _arun(self, data: str) -> str:
        """异步处理"""
        print(f"  异步处理: {data}")
        await asyncio.sleep(0.5)
        return f"异步结果: {data.upper()}"


async def mixed_sync_async_example():
    """混合同步异步示例"""
    print("\n" + "=" * 50)
    print("混合同步异步工具示例")
    print("=" * 50)

    processor = DataProcessor()

    # 同步调用
    print("\n同步模式:")
    result_sync = processor.invoke({"data": "hello world"})
    print(f"结果: {result_sync}")

    # 异步调用
    print("\n异步模式:")
    result_async = await processor.ainvoke({"data": "hello world"})
    print(f"结果: {result_async}")


# ==================== 3. 流式工具 ====================

class StreamingTool(BaseTool):
    """流式数据生成工具"""

    name: str = "generate_stream"
    description: str = "生成流式数据"

    def _run(self, topic: str, count: int = 5) -> Iterator[str]:
        """同步流式生成"""
        for i in range(count):
            time.sleep(0.2)
            yield f"[{i+1}/{count}] {topic} 的内容片段 {i+1}\n"

    async def _arun(self, topic: str, count: int = 5) -> AsyncIterator[str]:
        """异步流式生成"""
        for i in range(count):
            await asyncio.sleep(0.2)
            yield f"[{i+1}/{count}] {topic} 的内容片段 {i+1}\n"


def streaming_tool_example():
    """流式工具示例"""
    print("\n" + "=" * 50)
    print("流式工具示例")
    print("=" * 50)

    tool = StreamingTool()

    print("\n生成流式内容:")
    for chunk in tool._run(topic="Python 教程", count=3):
        print(chunk, end="", flush=True)


# ==================== 4. 数据库工具 ====================

class DatabaseTool(BaseTool):
    """数据库查询工具"""

    name: str = "query_database"
    description: str = "查询数据库并返回结果"

    # 模拟数据库
    _database: Dict[str, List[Dict]] = {
        "users": [
            {"id": 1, "name": "张三", "email": "zhang@example.com"},
            {"id": 2, "name": "李四", "email": "li@example.com"},
            {"id": 3, "name": "王五", "email": "wang@example.com"}
        ],
        "products": [
            {"id": 101, "name": "笔记本", "price": 5000},
            {"id": 102, "name": "手机", "price": 3000},
            {"id": 103, "name": "平板", "price": 2000}
        ]
    }

    def _run(self, table: str, filter_field: str = None, filter_value: str = None) -> str:
        """查询数据库"""
        if table not in self._database:
            return f"表 '{table}' 不存在"

        data = self._database[table]

        # 应用过滤
        if filter_field and filter_value:
            data = [
                row for row in data
                if str(row.get(filter_field)) == filter_value
            ]

        # 格式化输出
        if not data:
            return "未找到匹配记录"

        result = f"从 {table} 表找到 {len(data)} 条记录:\n"
        for row in data:
            result += f"  {row}\n"

        return result

    async def _arun(self, table: str, filter_field: str = None, filter_value: str = None) -> str:
        """异步查询"""
        await asyncio.sleep(0.1)  # 模拟查询延迟
        return self._run(table, filter_field, filter_value)


def database_tool_example():
    """数据库工具示例"""
    print("\n" + "=" * 50)
    print("数据库工具示例")
    print("=" * 50)

    db_tool = DatabaseTool()

    # 查询所有用户
    print("\n查询 1: 所有用户")
    result = db_tool.invoke({"table": "users"})
    print(result)

    # 按条件过滤
    print("\n查询 2: 查找特定用户")
    result = db_tool.invoke({
        "table": "users",
        "filter_field": "name",
        "filter_value": "张三"
    })
    print(result)

    # 查询产品
    print("\n查询 3: 所有产品")
    result = db_tool.invoke({"table": "products"})
    print(result)


# ==================== 5. API 调用工具 ====================

class APITool(BaseTool):
    """API 调用工具"""

    name: str = "call_api"
    description: str = "调用外部 API 获取数据"

    api_base_url: str = Field(default="https://api.example.com", description="API 基础 URL")

    async def _arun(self, endpoint: str, method: str = "GET", params: Dict = None) -> str:
        """异步 API 调用"""
        if params is None:
            params = {}

        print(f"  调用 API: {method} {self.api_base_url}{endpoint}")
        print(f"  参数: {params}")

        # 模拟 API 调用
        await asyncio.sleep(0.5)

        # 模拟响应
        mock_responses = {
            "/weather": {"temp": 22, "condition": "晴朗"},
            "/users": [{"id": 1, "name": "用户1"}],
            "/products": [{"id": 101, "name": "产品1", "price": 100}]
        }

        response = mock_responses.get(endpoint, {"message": "成功"})
        return f"API 响应: {response}"

    def _run(self, endpoint: str, method: str = "GET", params: Dict = None) -> str:
        """同步调用（使用 asyncio.run）"""
        return asyncio.run(self._arun(endpoint, method, params))


async def api_tool_example():
    """API 工具示例"""
    print("\n" + "=" * 50)
    print("API 调用工具示例")
    print("=" * 50)

    api_tool = APITool()

    # 并行调用多个 API
    endpoints = ["/weather", "/users", "/products"]

    print("\n并行调用多个 API...")
    tasks = [
        api_tool.ainvoke({"endpoint": ep})
        for ep in endpoints
    ]

    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results, 1):
        print(f"\nAPI {i} 结果:")
        print(result)


# ==================== 6. 文件处理工具 ====================

class FileOperationTool(BaseTool):
    """文件操作工具"""

    name: str = "file_operation"
    description: str = "执行文件读取、写入等操作"

    def _run(self, operation: str, file_path: str, content: str = None) -> str:
        """执行文件操作"""
        if operation == "read":
            # 模拟读取
            return f"文件 {file_path} 的内容: [示例内容]"

        elif operation == "write":
            if not content:
                return "写入操作需要提供 content 参数"
            # 模拟写入
            return f"已将内容写入 {file_path}"

        elif operation == "list":
            # 模拟列出文件
            mock_files = ["file1.txt", "file2.txt", "file3.txt"]
            return f"目录 {file_path} 包含: {', '.join(mock_files)}"

        else:
            return f"不支持的操作: {operation}"

    async def _arun(self, operation: str, file_path: str, content: str = None) -> str:
        """异步文件操作"""
        await asyncio.sleep(0.1)
        return self._run(operation, file_path, content)


def file_tool_example():
    """文件工具示例"""
    print("\n" + "=" * 50)
    print("文件操作工具示例")
    print("=" * 50)

    file_tool = FileOperationTool()

    # 读取文件
    print("\n操作 1: 读取文件")
    result = file_tool.invoke({
        "operation": "read",
        "file_path": "/data/report.txt"
    })
    print(result)

    # 写入文件
    print("\n操作 2: 写入文件")
    result = file_tool.invoke({
        "operation": "write",
        "file_path": "/data/output.txt",
        "content": "Hello, World!"
    })
    print(result)

    # 列出文件
    print("\n操作 3: 列出文件")
    result = file_tool.invoke({
        "operation": "list",
        "file_path": "/data"
    })
    print(result)


# ==================== 7. 缓存工具 ====================

class CachedTool(BaseTool):
    """带缓存的工具"""

    name: str = "cached_operation"
    description: str = "执行带缓存的操作"

    _cache: Dict[str, str] = {}

    def _run(self, key: str, compute: bool = False) -> str:
        """执行操作（带缓存）"""
        # 检查缓存
        if key in self._cache and not compute:
            print(f"  ✓ 缓存命中: {key}")
            return f"缓存结果: {self._cache[key]}"

        # 计算结果
        print(f"  计算中: {key}")
        time.sleep(0.5)  # 模拟耗时操作
        result = f"{key.upper()}_COMPUTED"

        # 存入缓存
        self._cache[key] = result
        print(f"  ✓ 已缓存: {key}")

        return f"新计算结果: {result}"

    async def _arun(self, key: str, compute: bool = False) -> str:
        """异步执行"""
        if key in self._cache and not compute:
            print(f"  ✓ 缓存命中: {key}")
            return f"缓存结果: {self._cache[key]}"

        print(f"  计算中: {key}")
        await asyncio.sleep(0.5)
        result = f"{key.upper()}_COMPUTED"

        self._cache[key] = result
        print(f"  ✓ 已缓存: {key}")

        return f"新计算结果: {result}"


def cached_tool_example():
    """缓存工具示例"""
    print("\n" + "=" * 50)
    print("缓存工具示例")
    print("=" * 50)

    cached_tool = CachedTool()

    # 第一次调用 - 需要计算
    print("\n第一次调用 (需要计算):")
    result = cached_tool.invoke({"key": "data_123"})
    print(f"结果: {result}")

    # 第二次调用 - 使用缓存
    print("\n第二次调用 (使用缓存):")
    result = cached_tool.invoke({"key": "data_123"})
    print(f"结果: {result}")

    # 强制重新计算
    print("\n第三次调用 (强制计算):")
    result = cached_tool.invoke({"key": "data_123", "compute": True})
    print(f"结果: {result}")


# ==================== 8. 批处理工具 ====================

class BatchProcessTool(BaseTool):
    """批处理工具"""

    name: str = "batch_process"
    description: str = "批量处理多个项目"

    async def _arun(self, items: List[str], operation: str = "process") -> str:
        """异步批处理"""
        print(f"\n  批处理 {len(items)} 个项目...")

        async def process_item(item: str, index: int) -> str:
            await asyncio.sleep(0.2)
            return f"[{index+1}] {operation}: {item} -> 完成"

        # 并行处理
        tasks = [
            process_item(item, i)
            for i, item in enumerate(items)
        ]

        results = await asyncio.gather(*tasks)
        return "\n  ".join(["批处理结果:"] + results)

    def _run(self, items: List[str], operation: str = "process") -> str:
        """同步批处理"""
        return asyncio.run(self._arun(items, operation))


async def batch_tool_example():
    """批处理工具示例"""
    print("\n" + "=" * 50)
    print("批处理工具示例")
    print("=" * 50)

    batch_tool = BatchProcessTool()

    # 批量处理
    items = ["文档1.pdf", "文档2.pdf", "文档3.pdf", "文档4.pdf"]

    print("\n执行批处理...")
    result = await batch_tool.ainvoke({
        "items": items,
        "operation": "转换"
    })
    print(f"\n{result}")


# ==================== 9. 与模型集成 ====================

async def model_integration_example():
    """与模型集成示例"""
    print("\n" + "=" * 50)
    print("与模型集成 - 异步工具")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus")

    # 创建工具列表
    tools = [
        async_fetch_data,
        DatabaseTool(),
        APITool(),
        FileOperationTool()
    ]

    model_with_tools = model.bind_tools(tools)

    # 测试工具调用
    print("\n场景: 请求获取用户数据")
    response = model_with_tools.invoke([
        HumanMessage(content="查询数据库中的所有用户")
    ])

    if response.tool_calls:
        print("\n模型生成的工具调用:")
        for tool_call in response.tool_calls:
            print(f"  工具: {tool_call['name']}")
            print(f"  参数: {tool_call['args']}")


# ==================== 10. StructuredTool 使用 ====================

def simple_function(text: str, count: int = 1) -> str:
    """简单的处理函数"""
    return f"{text} " * count


def structured_tool_example():
    """StructuredTool 示例"""
    print("\n" + "=" * 50)
    print("StructuredTool 示例")
    print("=" * 50)

    # 从函数创建工具
    tool = StructuredTool.from_function(
        func=simple_function,
        name="repeat_text",
        description="重复文本指定次数"
    )

    print(f"\n工具名称: {tool.name}")
    print(f"工具描述: {tool.description}")

    # 使用工具
    result = tool.invoke({"text": "Hello", "count": 3})
    print(f"\n结果: {result}")


# ==================== 主函数 ====================

async def run_async_examples():
    """运行异步示例"""
    await async_tool_example()
    await mixed_sync_async_example()
    await api_tool_example()
    await batch_tool_example()
    await model_integration_example()


def main():
    """主函数"""
    try:
        # 同步示例
        streaming_tool_example()
        # database_tool_example()
        # file_tool_example()
        # cached_tool_example()
        # structured_tool_example()

        # # 异步示例
        # print("\n" + "=" * 50)
        # print("运行异步示例...")
        # print("=" * 50)
        # asyncio.run(run_async_examples())

        print("\n" + "=" * 50)
        print("所有异步和特殊工具示例完成!")
        print("=" * 50)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
