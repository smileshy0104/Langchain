# LangChain Multi Agent：Subagents 示例

本目录对应 `langchain-docs/LangChain_Multi_Agent_详细指南.md` 中的：

- Subagents 的基本实现
- Subagents 添加人工审核
- Handoffs（交接模式）案例
- Skills（技能加载模式）案例
- Router（路由模式）案例
- 实战案例

示例保留需要配置模型的版本，并支持 OpenAI 兼容自定义供应商。

## 环境变量

默认会读取本目录 `.env`，也会读取 `../langchain_context_engineering_examples/.env`：

```bash
LANGCHAIN_MODEL_PROVIDER=custom
CUSTOM_MODEL_BASE_URL=https://ai-api.kkidc.com
CUSTOM_MODEL_API_KEY=你的 API Key
LANGCHAIN_DEFAULT_MODEL=gpt-5.5
LANGCHAIN_DEFAULT_TEMPERATURE=0.2
```

`CUSTOM_MODEL_BASE_URL` 可以写 `https://ai-api.kkidc.com`，代码会自动补 `/v1`。

## 运行

```bash
cd /Users/aiyer/Applications/GolandProjects/Langchain/langchain-example/langchain_multi_agent_examples
conda activate langchain-env

python 01_subagents_basic.py
python 02_subagents_with_human_review.py
python 03_handoffs_single_agent_middleware.py
python 04_handoffs_multi_agent_graph.py
python 05_handoffs_customer_support_workflow.py
python 06_skills_basic_loading.py
python 07_skills_constrained_sql.py
python 08_skills_middleware.py
python 09_skills_document_analysis.py
python 10_router_basic_conditional.py
python 11_router_command_node.py
python 12_router_parallel_send.py
python 13_router_as_tool_conversation.py
python 14_practical_multi_domain_customer_service.py
python 15_practical_document_analysis_system.py
```

## 文件说明

- `01_subagents_basic.py`：创建日程子 Agent、邮件子 Agent，并由 Supervisor 调度。
- `02_subagents_with_human_review.py`：在 Supervisor 层使用 `HumanInTheLoopMiddleware` 审核对子 Agent 工具的调用。
- `03_handoffs_single_agent_middleware.py`：单 Agent + Middleware 的轻量交接，按状态切换一线客服/专家提示词和工具。
- `04_handoffs_multi_agent_graph.py`：销售 Agent 与技术支持 Agent 作为独立图节点，通过 `Command.PARENT` 交接控制权。
- `05_handoffs_customer_support_workflow.py`：保修信息收集、问题分类、解决方案专家的多步骤客服工作流。
- `06_skills_basic_loading.py`：基础技能加载，Agent 先调用 `load_skill` 获取领域 schema 和规则。
- `07_skills_constrained_sql.py`：强约束 SQL，必须先加载技能并写入 `skills_loaded` 状态，才能验证 SQL。
- `08_skills_middleware.py`：自定义中间件动态注入技能目录，再按需加载完整技能。
- `09_skills_document_analysis.py`：智能文档分析系统，根据文档类型加载财务/法律分析技能。
- `10_router_basic_conditional.py`：条件边单路由，根据请求分类到 research/code/writing 专家。
- `11_router_command_node.py`：显式 router 节点返回 `Command(goto=...)`，并记录路由结果。
- `12_router_parallel_send.py`：使用 `Send` 并行查询 GitHub/Notion/Slack 风格知识源，再汇总。
- `13_router_as_tool_conversation.py`：把 stateless router 包装成工具，由外层对话 Agent 统一回复。
- `14_practical_multi_domain_customer_service.py`：实战案例 1，账单专家与技术支持专家组成完整客服 handoff 图。
- `15_practical_document_analysis_system.py`：实战案例 2，Supervisor + Skills 的智能文档分析系统。
- `model_config.py`：统一模型配置，支持自定义 OpenAI 兼容供应商。
- `hitl_utils.py`：HITL 中断展示、决策构造和恢复执行工具函数。

## 常用命令

编辑 Supervisor 转交给子 Agent 的请求：

```bash
python 02_subagents_with_human_review.py \
  --decision edit \
  --edited-request "只安排下周二下午2点的设计评审会议，暂时不要发送邮件。"
```

拒绝调用子 Agent：

```bash
python 02_subagents_with_human_review.py --decision reject
```

从销售转接到技术支持：

```bash
python 04_handoffs_multi_agent_graph.py \
  --request "我本来在了解 Pro 套餐，但现在设备一直无法联网，请转技术支持帮我排查。"
```

运行客户支持工作流：

```bash
python 05_handoffs_customer_support_workflow.py \
  --request "我的手机屏幕摔碎了，设备还在保修期内，请问可以免费维修吗？"
```

基础技能加载：

```bash
python 06_skills_basic_loading.py \
  --request "帮我分析上个月的销售额趋势，并给出可以使用的 SQL。"
```

强约束 SQL：

```bash
python 07_skills_constrained_sql.py \
  --request "生成一个 SQL，统计上个月每周已支付订单的销售额趋势。"
```

智能文档分析：

```bash
python 09_skills_document_analysis.py \
  --request "请分析 service_contract.txt，这是一份法律合同。"
```

条件边单路由：

```bash
python 10_router_basic_conditional.py \
  --request "如何用 Python 实现快速排序？"
```

显式 `Command` 路由节点：

```bash
python 11_router_command_node.py \
  --request "帮我润色一封发给客户的项目延期说明邮件。"
```

并行知识源路由：

```bash
python 12_router_parallel_send.py \
  --request "我们准备上线新的账单导出功能，需要确认代码、文档和团队讨论里有哪些注意事项。"
```

完整多领域客服系统：

```bash
python 14_practical_multi_domain_customer_service.py \
  --request "我的发票 #12345 金额是多少？"
```

智能文档分析实战：

```bash
python 15_practical_document_analysis_system.py \
  --request "请分析 quarterly_report.txt，这是一份财务报告。"
```
