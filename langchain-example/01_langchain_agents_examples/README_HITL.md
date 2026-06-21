# Human-in-the-Loop (HITL) 实现说明

## ⚠️ 重要提示

原始的 `05_human_in_the_loop.py` 文件使用了不存在的 API:
- `human_in_the_loop_middleware` - 不存在
- `HITLResponse`, `Decision` - 不存在

## 正确的实现方式

LangGraph 提供了 `interrupt()` 函数来实现 Human-in-the-Loop:

### 1. 使用 `interrupt()` 暂停执行

```python
from langgraph.types import interrupt

@tool
def sensitive_operation(data: str) -> str:
    """需要人工审批的敏感操作"""

    # 暂停并请求人工批准
    user_approval = interrupt(
        {
            "action": "sensitive_operation",
            "data": data,
            "question": "是否批准此操作?"
        }
    )

    if user_approval:
        return f"已执行操作: {data}"
    else:
        return "操作被拒绝"
```

### 2. 恢复执行

```python
from langgraph.types import Command

# 暂停后恢复
result = graph.invoke(
    Command(resume="yes"),  # 用户的批准
    config={"configurable": {"thread_id": "123"}}
)
```

### 3. ChatZhipuAI 的限制

由于 ChatZhipuAI 不支持完整的 tool choice 模式,建议:

1. **使用 LangGraph 的 StateGraph** 手动构建 agent
2. **或使用支持完整功能的模型** (OpenAI GPT-4, Anthropic Claude)

## 参考资料

- [LangGraph Human-in-the-Loop 文档](https://docs.langchain.com/langgraph-platform/langgraph-basics/4-human-in-the-loop)
- [interrupt() 使用指南](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)
- [LangChain Blog: Making it easier to build HITL agents](https://blog.langchain.com/making-it-easier-to-build-human-in-the-loop-agents-with-interrupt/)

## 简化示例

查看 `05_human_in_the_loop_simple.py` 获取一个可运行的简化示例。
