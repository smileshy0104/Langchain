# LangChain Human-in-the-Loop 实战案例

本目录对应 `langchain-docs/LangChain_HumanInTheLoop_详细指南.md` 的实战案例，保留需要配置模型的代码版本，并支持 OpenAI 兼容自定义供应商。

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
cd /Users/aiyer/Applications/GolandProjects/Langchain/langchain-example/langchain_hitl_examples
conda activate langchain-env
python 01_database_approval.py
python 02_email_review.py
python 03_finance_approval.py --second-approver manager-001
python 04_audited_hitl.py
```

## 示例说明

- `01_database_approval.py`：SQL 执行和数据导出审批。
- `02_email_review.py`：邮件发送和会议安排审核。
- `03_finance_approval.py`：转账和发票审批，并演示大额/外部转账二级审批。
- `04_audited_hitl.py`：敏感操作审批，并将决策写入 JSONL 审计日志。

每个脚本都支持 `--decision approve|edit|reject`，第四个脚本额外支持 `respond`。运行时会先打印待审批工具调用，再用命令行参数构造人工决策并恢复 Agent。

## 常用命令

修改邮件收件人后再批准：

```bash
python 02_email_review.py --decision edit --edited-to bob@company.com
```

拒绝数据库导出：

```bash
python 01_database_approval.py --decision reject
```

修改发票金额：

```bash
python 03_finance_approval.py \
  --request "请审批发票 INV-1001，金额 9800 美元。" \
  --decision edit \
  --edited-amount 9000
```

查看审计日志：

```bash
tail -n 5 logs/hitl_audit.jsonl
```
