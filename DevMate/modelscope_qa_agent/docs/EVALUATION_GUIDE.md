# RAG System Evaluation Guide

## 概述

本文档说明如何使用 RAGAs 框架评估 ModelScope QA Agent 的性能。

## 评估指标

### 1. Context Recall (上下文召回率)
- **定义**: 检索到的文档是否包含回答问题所需的所有信息
- **目标**: ≥85%
- **重要性**: 高 - 如果检索不到相关信息，无法生成正确答案

### 2. Faithfulness (答案忠实度)
- **定义**: 生成的答案是否忠实于检索到的文档内容
- **目标**: ≥95%
- **重要性**: 高 - 防止模型"幻觉"，确保答案基于事实

### 3. Answer Relevancy (答案相关性)
- **定义**: 生成的答案是否与用户问题相关
- **目标**: 尽可能高
- **重要性**: 中 - 确保回答切题

### 4. Answer Correctness (答案正确性)
- **定义**: 答案与标准答案的匹配程度
- **目标**: 尽可能高
- **重要性**: 高 - 最终质量指标

### 5. Response Time (响应时间)
- **定义**: 从接收问题到返回答案的时间
- **目标**: <30秒 (P50)
- **重要性**: 中 - 影响用户体验

---

## 准备工作

### 1. 环境检查

确保以下服务正常运行:

```bash
# 检查 Milvus
python scripts/check_infrastructure.py

# 检查向量数据库
python scripts/verify_knowledge_base.py
```

### 2. 配置 API 密钥

在 `.env` 文件中配置:

```bash
DASHSCOPE_API_KEY=your_api_key_here
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 3. 准备评测数据集

评测数据集已准备在 `data/evaluation_dataset.json`，包含 31 个真实 ModelScope 技术问题。

数据集结构:
```json
[
  {
    "question": "用户问题",
    "ground_truth": "标准答案",
    "contexts": ["相关文档片段1", "相关文档片段2"],
    "category": "问题分类"
  }
]
```

---

## 运行评估

### 基本用法

```bash
# 评估所有测试问题
python scripts/evaluate_rag.py

# 评估前 5 个问题 (快速测试)
python scripts/evaluate_rag.py --max-samples 5

# 指定输出路径
python scripts/evaluate_rag.py --output results/my_evaluation.md
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 评测数据集路径 | `data/evaluation_dataset.json` |
| `--output` | 报告输出路径 | `results/evaluation_report.md` |
| `--max-samples` | 最多评估的样本数 | 全部 |
| `--api-key` | DashScope API Key | 从 .env 读取 |

### 输出文件

评估完成后会生成两个文件:

1. **Markdown 报告**: `results/evaluation_report.md`
   - 包含评估指标、统计数据、达标情况
   - 便于人类阅读

2. **JSON 详细结果**: `results/evaluation_report.json`
   - 包含每个问题的详细结果
   - 便于程序化分析

---

## 评估流程

### 1. 加载数据集
```
📥 加载评测数据集
✅ 加载成功: 31 条测试数据

📊 数据集类别分布:
   - model_usage: 8 条
   - platform_usage: 5 条
   - error_handling: 4 条
   ...
```

### 2. 运行推理
```
🤖 运行 Agent 推理
[1/31] 处理问题: 如何使用 transformers 库加载 Qwen-7B 模型？...
   ✅ 完成 (耗时: 15.32s, 置信度: 0.95)
[2/31] 处理问题: ModelScope 平台上如何下载模型？...
   ✅ 完成 (耗时: 12.85s, 置信度: 0.92)
...
```

### 3. RAGAs 评估
```
📊 RAGAs 评估
🔍 开始评估 (共 31 条数据)...
   评估指标: Context Recall, Faithfulness, Answer Relevance, Answer Correctness
✅ 评估完成!
```

### 4. 响应速度评估
```
⏱️  响应速度评估
📊 响应时间统计:
   - 平均: 18.45s
   - P50: 16.23s
   - P95: 25.67s
   - P99: 28.34s

🎯 目标达成情况:
   - 目标阈值: <30s
   - 达标率: 96.8% (30/31)
   ✅ 平均响应时间达标!
```

---

## 评估报告示例

```markdown
# RAG System Evaluation Report

**评估时间**: 2025-12-01T22:30:15
**测试问题数**: 31

---

## RAGAs 评估指标

| 指标 | 得分 | 目标 | 状态 |
|------|------|------|------|
| Context Recall | 87.3% | ≥85% | ✅ 达标 |
| Answer Faithfulness | 96.8% | ≥95% | ✅ 达标 |
| Answer Relevance | 91.5% | - | - |
| Answer Correctness | 88.2% | - | - |

---

## 响应速度评估

| 指标 | 数值 |
|------|------|
| 平均响应时间 | 18.45s |
| P50 (中位数) | 16.23s |
| P95 | 25.67s |
| P99 | 28.34s |

**目标达成情况**:
- 目标阈值: <30s
- 达标率: 96.8% (30/31)
- 状态: ✅ 达标

---

## 总结

- ✅ Context Recall 达到目标 (≥85%)
- ✅ Answer Faithfulness 达到目标 (≥95%)
- ✅ 响应速度达标 (平均 18.45s < 30s)
```

---

## 常见问题

### Q1: 评估需要多长时间？

- 全量评估 (31个问题): 约 10-15 分钟
- 快速测试 (5个问题): 约 2-3 分钟

实际时间取决于:
- LLM API 响应速度
- 向量检索速度
- 网络延迟

### Q2: 评估失败怎么办？

常见原因:
1. **API Key 无效**: 检查 `.env` 文件
2. **Milvus 未运行**: 启动 Milvus 服务
3. **向量库为空**: 运行 `scripts/load_knowledge_base.py`
4. **依赖缺失**: 运行 `pip install ragas==0.3.9`

### Q3: 如何提高评估指标？

参考 `docs/OPTIMIZATION_GUIDE.md` 中的优化建议。

### Q4: 评估指标含义是什么？

**Context Recall (上下文召回率)**:
- 检索到的文档是否包含足够的信息
- 低分说明: 检索不够精准，需要调优检索策略

**Faithfulness (忠实度)**:
- 答案是否基于检索到的文档
- 低分说明: 模型出现"幻觉"，需要调整 Prompt

**Answer Relevancy (答案相关性)**:
- 答案是否切题
- 低分说明: 生成策略有问题，需要优化 Prompt

**Answer Correctness (答案正确性)**:
- 与标准答案的匹配度
- 综合质量指标

### Q5: 可以自定义评测数据集吗？

可以! 创建自己的 JSON 文件，格式如下:

```json
[
  {
    "question": "你的问题",
    "ground_truth": "标准答案",
    "contexts": ["相关文档"],
    "category": "分类"
  }
]
```

然后运行:
```bash
python scripts/evaluate_rag.py --dataset your_dataset.json
```

---

## 下一步

完成评估后，根据结果进行优化:

1. **如果 Context Recall 低**: 调优检索权重和参数
2. **如果 Faithfulness 低**: 调整 Prompt，强调"基于文档回答"
3. **如果响应时间慢**: 优化检索数量、使用缓存
4. **如果 Answer Correctness 低**: 微调 LLM 或改进 Prompt

详见: [优化指南](OPTIMIZATION_GUIDE.md)

---

## 参考资料

- [RAGAs 官方文档](https://docs.ragas.io/)
- [LangChain 评估指南](https://python.langchain.com/docs/guides/evaluation/)
- [ModelScope 文档](https://modelscope.cn/docs)
