"""
案例 1：智能客服系统。

结合：
- Milvus：产品知识与 FAQ 检索
- InMemoryStore：客户资料、购买历史、交互记录
- InMemorySaver：同一会话的短期对话状态
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from model_config import create_configured_model
from retrieval_utils import create_milvus_vectorstore, format_docs


@dataclass
class CustomerContext:
    customer_id: str
    session_id: str


class CustomerServiceAgent:
    def __init__(self) -> None:
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.vectorstore = self._init_knowledge_base()
        self._seed_customer_memory()
        self.agent = self._create_agent()

    # 初始化知识库
    def _init_knowledge_base(self) -> Any:
        product_docs = [
            "产品 A 是企业级协作软件，定价 99 美元/月，包含 SSO、审计日志和高级权限。",
            "产品 B 面向中小企业，定价 49 美元/月，包含基础项目管理和邮件支持。",
            "企业版支持 SLA、专属客户成功经理、私有化部署和年度合同。",
            "退款政策：月付套餐可在购买后 7 天内申请退款；年付套餐需联系账单支持。",
            "账单导出支持 CSV 和 XLSX，字段包含发票号、账期、支付状态和客户 ID。",
            "常见故障：无法登录时请先重置密码，检查 SSO 配置和账号是否被禁用。",
        ]
        return create_milvus_vectorstore(
            collection_name="rm_customer_service_docs",
            texts=product_docs,
        )

    # 初始化客户资料
    def _seed_customer_memory(self) -> None:
        self.store.put(
            ("customers", "cust_001"),
            "profile",
            {"name": "林先生", "tier": "pro", "language": "zh"},
        )
        self.store.put(
            ("customers", "cust_001"),
            "purchases",
            [
                {"product": "产品 B", "plan": "pro", "renewal": "2026-09-01"},
                {"product": "账单导出插件", "plan": "monthly", "renewal": "2026-07-01"},
            ],
        )

    def _create_agent(self) -> Any:
        @tool
        def get_customer_info(runtime: ToolRuntime[CustomerContext]) -> str:
            """Get customer profile from long-term memory."""

            assert runtime.store is not None
            info = runtime.store.get(("customers", runtime.context.customer_id), "profile")
            return f"客户信息：{info.value}" if info else "客户：新用户"

        @tool
        def search_faq(query: str) -> str:
            """Search product knowledge and FAQ from Milvus."""

            docs = self.vectorstore.similarity_search(query, k=3)
            return format_docs(docs)

        @tool
        def save_interaction(
            issue: str,
            resolution: str,
            runtime: ToolRuntime[CustomerContext],
        ) -> str:
            """Save this support interaction into long-term memory."""

            assert runtime.store is not None
            runtime.store.put(
                ("customers", runtime.context.customer_id, "interactions"),
                runtime.context.session_id,
                {
                    "issue": issue,
                    "resolution": resolution,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )
            return "已保存交互记录"

        @tool
        def get_purchase_history(runtime: ToolRuntime[CustomerContext]) -> str:
            """Get customer purchase history."""

            assert runtime.store is not None
            history = runtime.store.get(("customers", runtime.context.customer_id), "purchases")
            return f"购买历史：{history.value}" if history else "暂无购买记录"

        return create_agent(
            model=create_configured_model(),
            tools=[get_customer_info, search_faq, save_interaction, get_purchase_history],
            checkpointer=self.checkpointer,
            store=self.store,
            context_schema=CustomerContext,
            system_prompt=(
                "你是专业客服助手。回答前根据需要查询客户资料、购买历史和产品知识库。"
                "解决客户问题后，使用 save_interaction 保存问题与处理结果。"
                "始终保持友好、专业、简洁。"
            ),
        )

    def handle_message(self, message: str, customer_id: str, session_id: str) -> str:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": session_id}},
            context=CustomerContext(customer_id=customer_id, session_id=session_id),
        )
        return str(result["messages"][-1].content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval & Memory 案例 1：智能客服系统")
    parser.add_argument("--message", default="你们的产品有哪些？我现在买的是哪个？")
    parser.add_argument("--customer-id", default="cust_001")
    parser.add_argument("--session-id", default="rm-customer-service-001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = CustomerServiceAgent()
    print(agent.handle_message(args.message, args.customer_id, args.session_id))


if __name__ == "__main__":
    main()
