"""
LangChain Tools - 参数验证和错误处理示例
演示 Pydantic 验证、ToolException、重试逻辑等
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool, BaseTool, ToolException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Type, List, Optional
import os
import time
import re

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 字段级验证 ====================

class EmailInput(BaseModel):
    """邮件发送参数验证"""

    to: str = Field(
        description="收件人邮箱地址",
        min_length=5,
        max_length=100
    )
    subject: str = Field(
        description="邮件主题",
        min_length=1,
        max_length=200
    )
    body: str = Field(
        description="邮件正文",
        min_length=1
    )
    cc: List[str] = Field(
        default_factory=list,
        description="抄送列表"
    )

    @field_validator("to")
    @classmethod
    def validate_email(cls, v):
        """验证邮箱格式"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError(f"无效的邮箱地址: {v}")
        return v.lower()  # 转为小写

    @field_validator("cc")
    @classmethod
    def validate_cc_emails(cls, v):
        """验证抄送邮箱列表"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        for email in v:
            if not re.match(email_pattern, email):
                raise ValueError(f"抄送列表中的邮箱格式错误: {email}")
        return [email.lower() for email in v]

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, v):
        """验证主题不包含敏感词"""
        forbidden_words = ["spam", "广告", "test"]
        v_lower = v.lower()
        if any(word in v_lower for word in forbidden_words):
            raise ValueError(f"主题包含禁止词汇")
        return v


@tool(args_schema=EmailInput)
def send_email(to: str, subject: str, body: str, cc: List[str] = None) -> str:
    """发送电子邮件"""
    if cc is None:
        cc = []

    print(f"  发送至: {to}")
    print(f"  主题: {subject}")
    print(f"  抄送: {', '.join(cc) if cc else '无'}")

    return f"邮件已成功发送至 {to}"

# 字段级验证示例
def field_validation_example():
    """字段级验证示例"""
    print("=" * 50)
    print("字段级验证示例")
    print("=" * 50)

    # 正确的邮件
    try:
        result = send_email.invoke({
            "to": "user@example.com",
            "subject": "会议通知",
            "body": "明天下午2点会议",
            "cc": ["manager@example.com"]
        })
        print(f"\n✓ 正确邮件: {result}")
    except Exception as e:
        print(f"✗ 错误: {e}")

    # 错误的邮箱格式
    try:
        result = send_email.invoke({
            "to": "invalid-email",
            "subject": "测试",
            "body": "内容"
        })
        print(f"结果: {result}")
    except Exception as e:
        print(f"\n✗ 邮箱格式错误被拒绝: {e}")

    # 主题包含禁止词
    try:
        result = send_email.invoke({
            "to": "user@example.com",
            "subject": "这是一个 spam 邮件",
            "body": "内容"
        })
        print(f"结果: {result}")
    except Exception as e:
        print(f"\n✗ 主题验证失败: {e}")


# ==================== 2. 模型级验证 ====================

class TransferInput(BaseModel):
    """转账参数验证"""

    from_account: str = Field(description="源账户")
    to_account: str = Field(description="目标账户")
    amount: float = Field(gt=0, description="转账金额")
    currency: str = Field(default="CNY", description="货币类型")

    @field_validator("from_account", "to_account")
    @classmethod
    def validate_account(cls, v):
        """验证账户格式"""
        if not re.match(r'^\d{10,20}$', v):
            raise ValueError(f"账户格式错误，应为10-20位数字")
        return v

    @model_validator(mode='after')
    def validate_transfer(self):
        """模型级验证：确保不是自己转给自己"""
        if self.from_account == self.to_account:
            raise ValueError("不能转账给自己")

        # 验证金额限制
        if self.amount > 50000:
            raise ValueError("单笔转账金额不能超过 50000")

        return self


@tool(args_schema=TransferInput)
def transfer_money(from_account: str, to_account: str, amount: float, currency: str = "CNY") -> str:
    """转账功能"""
    return f"已从账户 {from_account} 转账 {amount} {currency} 到账户 {to_account}"

# 模型级验证示例
def model_validation_example():
    """模型级验证示例"""
    print("\n" + "=" * 50)
    print("模型级验证示例")
    print("=" * 50)

    # 正确转账
    try:
        result = transfer_money.invoke({
            "from_account": "1234567890",
            "to_account": "9876543210",
            "amount": 1000.0
        })
        print(f"\n✓ 正确转账: {result}")
    except Exception as e:
        print(f"✗ 错误: {e}")

    # 转给自己
    try:
        result = transfer_money.invoke({
            "from_account": "1234567890",
            "to_account": "1234567890",
            "amount": 1000.0
        })
        print(f"结果: {result}")
    except Exception as e:
        print(f"\n✗ 不能转给自己: {e}")

    # 金额超限
    try:
        result = transfer_money.invoke({
            "from_account": "1234567890",
            "to_account": "9876543210",
            "amount": 100000.0
        })
        print(f"结果: {result}")
    except Exception as e:
        print(f"\n✗ 金额超限: {e}")


# ==================== 3. 使用 ToolException ====================

@tool
def delete_file(file_path: str, force: bool = False) -> str:
    """删除文件

    Args:
        file_path: 文件路径
        force: 是否强制删除（跳过安全检查）
    """
    # 安全检查
    dangerous_paths = ["/", "/usr", "/etc", "/System"]

    if not force and any(file_path.startswith(path) for path in dangerous_paths):
        raise ToolException(
            "不能删除系统目录！请使用 force=True 参数（不推荐）"
        )

    # 检查文件是否存在
    if not file_path.endswith(".txt"):
        raise ToolException(
            f"只能删除 .txt 文件，当前文件: {file_path}"
        )

    return f"文件 {file_path} 已删除"


@tool
def database_query(query: str) -> str:
    """执行数据库查询"""
    # SQL 注入检查
    dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT"]
    query_upper = query.upper()

    if any(keyword in query_upper for keyword in dangerous_keywords):
        raise ToolException(
            "检测到危险 SQL 操作，查询被拒绝 [SQL_INJECTION]"
        )

    if not query_upper.strip().startswith("SELECT"):
        raise ToolException(
            "只允许 SELECT 查询 [INVALID_OPERATION]"
        )

    return f"查询成功: 返回 5 条记录"

# ToolException 依赖示例
def tool_exception_example():
    """ToolException 示例"""
    print("\n" + "=" * 50)
    print("ToolException 示例")
    print("=" * 50)

    # 正常删除
    try:
        result = delete_file.invoke({
            "file_path": "/home/user/test.txt",
            "force": False
        })
        print(f"\n✓ {result}")
    except ToolException as e:
        print(f"✗ 工具错误: {e}")

    # 尝试删除系统目录
    try:
        result = delete_file.invoke({
            "file_path": "/etc/passwd",
            "force": False
        })
        print(f"结果: {result}")
    except ToolException as e:
        print(f"\n✗ 安全检查失败: {e}")

    # SQL 查询 - 正常
    try:
        result = database_query.invoke({
            "query": "SELECT * FROM users WHERE id = 1"
        })
        print(f"\n✓ {result}")
    except ToolException as e:
        print(f"✗ 错误: {e}")

    # SQL 注入尝试
    try:
        result = database_query.invoke({
            "query": "DROP TABLE users"
        })
        print(f"结果: {result}")
    except ToolException as e:
        print(f"\n✗ SQL 注入被阻止: {e}")


# ==================== 4. 工具错误处理中间件 ====================

class ErrorHandlingTool(BaseTool):
    """带错误处理的工具基类"""

    max_retries: int = Field(default=3, description="最大重试次数")
    retry_delay: float = Field(default=1.0, description="重试延迟(秒)")

    def _run_with_retry(self, func, *args, **kwargs):
        """带重试的执行"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                    print(f"  第 {attempt + 1} 次尝试失败，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise ToolException(
                        f"操作失败，已重试 {self.max_retries} 次: {str(last_error)}"
                    )


class APIInput(BaseModel):
    """API 调用输入"""
    endpoint: str = Field(description="API 端点")
    method: str = Field(default="GET", description="HTTP 方法")


class ReliableAPITool(ErrorHandlingTool):
    """可靠的 API 调用工具"""

    name: str = "call_api"
    description: str = "调用外部 API，支持自动重试"
    args_schema: Type[BaseModel] = APIInput

    # 模拟失败计数
    _attempt_count: int = 0

    def _run(self, endpoint: str, method: str = "GET") -> str:
        """执行 API 调用"""
        return self._run_with_retry(self._call_api, endpoint, method)

    def _call_api(self, endpoint: str, method: str) -> str:
        """实际的 API 调用（模拟）"""
        self._attempt_count += 1

        # 模拟前两次失败
        if self._attempt_count <= 2:
            raise Exception(f"网络错误: 连接超时")

        return f"API {method} {endpoint} 调用成功"

    async def _arun(self, endpoint: str, method: str = "GET") -> str:
        """异步执行"""
        return self._run(endpoint, method)


def error_handling_middleware_example():
    """错误处理中间件示例"""
    print("\n" + "=" * 50)
    print("错误处理中间件示例")
    print("=" * 50)

    # 创建工具实例
    api_tool = ReliableAPITool(max_retries=3, retry_delay=0.5)

    print("\n调用 API (会自动重试)...")
    try:
        result = api_tool.invoke({
            "endpoint": "/api/users",
            "method": "GET"
        })
        print(f"\n✓ {result}")
    except ToolException as e:
        print(f"✗ 最终失败: {e}")


# ==================== 5. 复杂验证场景 ====================

class DateRange(BaseModel):
    """日期范围验证"""

    start_date: str = Field(description="开始日期 (YYYY-MM-DD)")
    end_date: str = Field(description="结束日期 (YYYY-MM-DD)")

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v):
        """验证日期格式"""
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError(f"日期格式错误，应为 YYYY-MM-DD: {v}")
        return v

    @model_validator(mode='after')
    def validate_date_range(self):
        """验证日期范围"""
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("开始日期不能晚于结束日期")

        return self


class ReportInput(BaseModel):
    """报告生成参数"""

    report_type: str = Field(description="报告类型")
    date_range: DateRange = Field(description="日期范围")
    include_charts: bool = Field(default=True, description="包含图表")
    format: str = Field(default="pdf", description="输出格式")

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, v):
        """验证报告类型"""
        valid_types = ["sales", "inventory", "financial", "customer"]
        if v not in valid_types:
            raise ValueError(f"无效的报告类型，支持: {', '.join(valid_types)}")
        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        """验证输出格式"""
        valid_formats = ["pdf", "xlsx", "csv", "html"]
        if v not in valid_formats:
            raise ValueError(f"无效的格式，支持: {', '.join(valid_formats)}")
        return v


@tool(args_schema=ReportInput)
def generate_report(
    report_type: str,
    date_range: DateRange,
    include_charts: bool = True,
    format: str = "pdf"
) -> str:
    """生成业务报告"""
    return f"已生成 {report_type} 报告 ({date_range.start_date} 至 {date_range.end_date})，格式: {format}"


def complex_validation_example():
    """复杂验证场景示例"""
    print("\n" + "=" * 50)
    print("复杂验证场景示例")
    print("=" * 50)

    # 正确的报告请求
    try:
        result = generate_report.invoke({
            "report_type": "sales",
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31"
            },
            "include_charts": True,
            "format": "pdf"
        })
        print(f"\n✓ {result}")
    except Exception as e:
        print(f"✗ 错误: {e}")

    # 日期范围错误
    try:
        result = generate_report.invoke({
            "report_type": "sales",
            "date_range": {
                "start_date": "2024-12-31",
                "end_date": "2024-01-01"
            }
        })
        print(f"结果: {result}")
    except Exception as e:
        print(f"\n✗ 日期范围错误: {e}")

    # 无效的报告类型
    try:
        result = generate_report.invoke({
            "report_type": "invalid_type",
            "date_range": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31"
            }
        })
        print(f"结果: {result}")
    except Exception as e:
        print(f"\n✗ 报告类型错误: {e}")


# ==================== 6. 与模型集成的错误处理 ====================

def model_integration_example():
    """与模型集成的错误处理示例"""
    print("\n" + "=" * 50)
    print("与模型集成的错误处理")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")

    # 绑定带验证的工具
    model_with_tools = model.bind_tools([
        send_email,
        delete_file,
        database_query
    ])

    # 测试：模型尝试发送邮件
    print("\n场景 1: 发送邮件")
    response = model_with_tools.invoke([
        HumanMessage(content="发送邮件给 1259581033@qq.com，主题是'会议通知'")
    ])

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"\n模型选择工具: {tool_call['name']}")
            print(f"参数: {tool_call['args']}")

            # 执行工具
            try:
                # 这里应该实际执行工具，这里仅演示
                print("✓ 工具调用将会执行（验证通过）")
            except ToolException as e:
                print(f"✗ 工具执行失败: {e}")

    # 测试：模型尝试危险操作
    print("\n场景 2: 尝试删除系统文件")
    response = model_with_tools.invoke([
        HumanMessage(content="删除 /etc/passwd 文件")
    ])

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"\n模型选择工具: {tool_call['name']}")
            print(f"参数: {tool_call['args']}")
            print("⚠️  这个操作会被工具的安全检查拦截")


# ==================== 7. 错误日志和监控 ====================

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoredTool(BaseTool):
    """带监控的工具"""

    name: str = "monitored_operation"
    description: str = "执行被监控的操作"

    def _run(self, operation: str) -> str:
        """执行操作"""
        logger.info(f"工具调用开始: {operation}")
        start_time = time.time()

        try:
            # 模拟操作
            time.sleep(0.1)
            result = f"操作 '{operation}' 完成"

            duration = time.time() - start_time
            logger.info(f"工具调用成功 - 耗时: {duration:.2f}s")

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"工具调用失败 - 耗时: {duration:.2f}s - 错误: {str(e)}")
            raise ToolException(f"操作失败: {str(e)}")

    async def _arun(self, operation: str) -> str:
        """异步执行"""
        return self._run(operation)


def logging_monitoring_example():
    """日志和监控示例"""
    print("\n" + "=" * 50)
    print("错误日志和监控示例")
    print("=" * 50)

    tool = MonitoredTool()

    print("\n执行被监控的操作...")
    result = tool.invoke({"operation": "数据同步"})
    print(f"\n结果: {result}")
    print("(查看上方日志输出)")


if __name__ == "__main__":
    try:
        # field_validation_example() # 字段级验证示例
        # model_validation_example() # 模型级验证示例
        # tool_exception_example() # ToolException 示例
        # error_handling_middleware_example() # 错误处理中间件示例
        # complex_validation_example() # 复杂验证示例
        # model_integration_example() # 模型集成示例
        logging_monitoring_example()

        print("\n" + "=" * 50)
        print("所有验证和错误处理示例完成!")
        print("=" * 50)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()
