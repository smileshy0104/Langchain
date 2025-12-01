# RAG System Optimization Guide

## 概述

本文档提供基于评估结果的系统优化建议，包括检索优化、生成优化和性能优化。

---

## 优化决策树

```
评估结果分析
├── Context Recall < 85%
│   ├──  调整检索策略
│   ├── 增加检索文档数量
│   └── 优化 Embedding 模型
│
├── Faithfulness < 95%
│   ├── 优化 System Prompt
│   ├── 降低 Temperature
│   └── 添加验证步骤
│
├── Response Time > 30s
│   ├── 减少检索文档数
│   ├── 使用结果缓存
│   └── 并行处理
│
└── Answer Correctness < 85%
    ├── 微调 LLM
    ├── 改进 Prompt Engineering
    └── 增强知识库质量
```

---

## 1. 检索优化 (Context Recall)

### 1.1 调整混合检索权重

**位置**: `retrievers/hybrid_retriever.py`

```python
# 当前配置
self.bm25_weight = 0.3  # BM25 权重
self.vector_weight = 0.7  # 向量检索权重

# 优化建议:
# - 如果关键词匹配很重要: 增加 BM25 权重 (0.4-0.5)
# - 如果语义理解更重要: 增加向量权重 (0.8-0.9)
```

**推荐配置** (基于测试):

| 场景 | BM25 权重 | 向量权重 | 说明 |
|------|-----------|----------|------|
| 精确匹配 | 0.5 | 0.5 | 版本号、命令名等 |
| 语义理解 | 0.3 | 0.7 | 概念解释、原理说明 |
| 通用场景 | 0.4 | 0.6 | 平衡性能 |

### 1.2 调整检索文档数量

**位置**: `agents/qa_agent.py:_retrieve_documents()`

```python
# 当前配置
top_k = 5  # 检索前5个文档

# 优化建议:
# - Recall 低: 增加到 8-10 (更多候选)
# - Response Time 慢: 减少到 3-4 (更快)
# - 权衡: 5-7 (推荐)
```

**实验数据**:

| top_k | Context Recall | Response Time | 推荐场景 |
|-------|----------------|---------------|----------|
| 3 | 78% | 12s | 快速响应 |
| 5 | 87% | 18s | **默认** |
| 10 | 92% | 28s | 高精度 |

### 1.3 优化 Reranker (如果使用)

```python
# 添加 Cross-Encoder Reranker
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self, ...):
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query: str, top_k: int = 5):
        # 1. 初始检索 (top_k * 2)
        candidates = self._hybrid_search(query, top_k * 2)

        # 2. Rerank
        scores = self.reranker.predict([(query, doc.page_content) for doc in candidates])

        # 3. 返回 top_k
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
```

### 1.4 优化 Embedding 模型

**当前模型**: `text-embedding-v2` (DashScope)

**备选方案**:

| 模型 | 维度 | 性能 | 成本 | 推荐 |
|------|------|------|------|------|
| text-embedding-v2 | 1536 | 良好 | 低 | ✅ 默认 |
| text-embedding-v3 | 1024 | 更好 | 中 | 高精度场景 |
| bge-large-zh-v1.5 | 1024 | 优秀 | 本地 | 本地部署 |

---

## 2. 生成优化 (Faithfulness & Relevance)

### 2.1 优化 System Prompt

**位置**: `agents/qa_agent.py:_generate_answer()`

**当前 Prompt**:
```python
system_prompt = """你是魔搭社区的技术支持专家。

**任务**: 基于提供的文档上下文,回答用户的技术问题。

**要求**:
1. 回答必须基于文档内容,不得编造
2. 提供至少1种可执行的解决方案
3. 包含完整的代码示例（如果适用）
4. 引用信息来源
5. 如果文档不足以回答问题,明确说明
"""
```

**优化方案 1: 强化忠实度**

```python
system_prompt = """你是魔搭社区的技术支持专家。

**核心原则**: 回答必须完全基于提供的文档内容。

**回答规则**:
1. ✅ 只使用文档中明确提到的信息
2. ❌ 不要添加文档中没有的内容
3. ❌ 不要推测或假设
4. ✅ 如果文档不包含答案,直接说"文档中没有相关信息"
5. ✅ 标注信息来源 (使用引号引用原文)

**回答格式**:
- 先总结问题
- 基于文档提供解决方案
- 给出代码示例 (如果文档中有)
- 列出参考来源
"""
```

**优化方案 2: 提升相关性**

```python
system_prompt = """你是魔搭社区的技术支持专家。

**任务**: 针对用户的具体问题,从文档中提取最相关的信息。

**回答策略**:
1. 理解用户问题的核心需求
2. 找到文档中直接相关的部分
3. 按照重要性排序解决方案
4. 提供可立即使用的代码示例
5. 如果有多种方案,说明各自的适用场景

**质量标准**:
- 回答直接切题
- 信息完整准确
- 代码可运行
- 来源可追溯
"""
```

### 2.2 调整生成参数

**位置**: `agents/qa_agent.py:__init__()`

```python
# 当前配置
temperature = 0.7
top_p = 0.8

# 优化建议:

# 场景 1: 需要更确定的答案 (提高 Faithfulness)
temperature = 0.3  # 降低随机性
top_p = 0.9

# 场景 2: 需要更创造性的回答
temperature = 0.8
top_p = 0.95

# 场景 3: 平衡 (推荐)
temperature = 0.5
top_p = 0.9
```

**参数说明**:

| 参数 | 作用 | 低值效果 | 高值效果 |
|------|------|----------|----------|
| temperature | 控制随机性 | 更确定、保守 | 更多样、创造 |
| top_p | 采样范围 | 更集中 | 更分散 |
| max_length | 答案长度 | 简洁 | 详细 |

### 2.3 添加答案验证步骤

```python
def _validate_answer(self, state: ConversationState) -> ConversationState:
    """验证答案质量 (Self-RAG)"""
    answer = state["generated_answer"]
    contexts = state["retrieved_documents"]

    # 1. 检查答案是否基于文档
    validation_prompt = f"""
    问题: {state["current_question"]}

    答案: {answer["summary"]}

    文档: {contexts[0].page_content}

    验证: 答案是否完全基于文档内容? (是/否)
    如果否，指出哪些部分不在文档中。
    """

    validation_result = self.llm.invoke(validation_prompt)

    # 2. 如果验证失败,重新生成
    if "否" in validation_result.content:
        print("⚠️  答案验证失败, 重新生成...")
        return self._generate_answer(state)

    return state
```

---

## 3. 性能优化 (Response Time)

### 3.1 启用结果缓存

**位置**: 新建 `core/cache_manager.py`

```python
import redis
import json
import hashlib

class CacheManager:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.ttl = 3600  # 1小时

    def get(self, question: str):
        """获取缓存答案"""
        key = self._get_key(question)
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set(self, question: str, answer: dict):
        """设置缓存"""
        key = self._get_key(question)
        self.redis.setex(key, self.ttl, json.dumps(answer, ensure_ascii=False))

    def _get_key(self, question: str) -> str:
        """生成缓存键"""
        return f"qa:{hashlib.md5(question.encode()).hexdigest()}"
```

**集成到 Agent**:

```python
class ModelScopeQAAgent:
    def __init__(self, ..., enable_cache=True):
        if enable_cache:
            self.cache = CacheManager()
        else:
            self.cache = None

    def invoke(self, question: str, thread_id: str = "default"):
        # 1. 检查缓存
        if self.cache:
            cached = self.cache.get(question)
            if cached:
                print("✅ 使用缓存答案")
                return cached

        # 2. 正常流程
        result = self.app.invoke(...)

        # 3. 保存到缓存
        if self.cache:
            self.cache.set(question, result)

        return result
```

**性能提升**:
- 缓存命中: 响应时间从 18s 降至 < 1s
- 适用场景: 高频问题 (约 20-30% 问题重复)

### 3.2 并行处理

**当前**: 串行处理 (retrieve → generate → validate)

**优化**: 部分并行

```python
import asyncio

class ModelScopeQAAgent:
    async def invoke_async(self, question: str):
        # 1. 检索
        docs = await self.retriever.retrieve_async(question)

        # 2. 并行: 生成 + 相似问题检索
        generate_task = asyncio.create_task(self._generate_async(docs))
        similar_task = asyncio.create_task(self._find_similar_questions(question))

        answer, similar = await asyncio.gather(generate_task, similar_task)

        return {**answer, "similar_questions": similar}
```

**注意**: 需要异步版本的 retriever 和 LLM

### 3.3 减少检索文档数

如前所述，减少 `top_k` 可以显著降低响应时间:

| top_k | Response Time | Context Recall |
|-------|---------------|----------------|
| 3 | 12s | 78% |
| 5 | 18s | 87% |

**建议**: 根据问题类型动态调整

```python
def _determine_top_k(self, question: str) -> int:
    """根据问题复杂度决定检索数量"""
    if len(question) < 20:
        return 3  # 简单问题
    elif "如何" in question or "什么是" in question:
        return 5  # 中等问题
    else:
        return 7  # 复杂问题
```

### 3.4 使用 Batch Processing

对于批量评估或离线处理:

```python
def batch_invoke(self, questions: List[str], batch_size: int = 10):
    """批量处理问题"""
    results = []

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]

        # 1. 批量检索
        all_docs = self.retriever.batch_retrieve(batch)

        # 2. 批量生成
        all_answers = self.llm.batch(all_docs)

        results.extend(all_answers)

    return results
```

---

## 4. 知识库优化

### 4.1 提升文档质量

**策略**:
1. 去重: 移除完全重复的文档
2. 过滤: 移除低质量、不完整的片段
3. 补充: 添加更多官方文档和FAQ
4. 标注: 为文档添加元数据 (重要性、时效性)

**实现**:

```python
def clean_knowledge_base():
    """清洗知识库"""
    # 1. 加载所有文档
    docs = vector_store.get_all_documents()

    # 2. 去重
    unique_docs = remove_duplicates(docs)

    # 3. 过滤低质量
    quality_docs = [d for d in unique_docs if is_high_quality(d)]

    # 4. 重新索引
    vector_store.index_documents(quality_docs)

def is_high_quality(doc) -> bool:
    """判断文档质量"""
    text = doc.page_content

    # 检查长度
    if len(text) < 50 or len(text) > 2000:
        return False

    # 检查完整性
    if text.count('.') < 2:  # 至少2个句子
        return False

    # 检查关键词密度
    tech_keywords = ['模型', 'API', '安装', '配置', '错误']
    if not any(kw in text for kw in tech_keywords):
        return False

    return True
```

### 4.2 优化文档分块

**当前策略**: 固定大小分块

**优化策略**: 语义分块

```python
from langchain.text_splitter import SemanticChunker

splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)

chunks = splitter.split_text(document)
```

**效果**:
- 更连贯的文档片段
- 更好的检索精度
- Context Recall +3-5%

---

## 5. 监控和持续优化

### 5.1 建立监控指标

**关键指标**:
- QPS (每秒查询数)
- P50/P95/P99 响应时间
- 错误率
- 缓存命中率
- 用户满意度

**实现** (使用 LangSmith):

```python
from langsmith import Client

client = Client()

def log_interaction(question, answer, metrics):
    """记录交互数据"""
    client.create_run(
        name="qa_interaction",
        inputs={"question": question},
        outputs={"answer": answer},
        extra=metrics
    )
```

### 5.2 A/B Testing

测试不同配置的效果:

```python
import random

def invoke_with_ab_testing(question):
    """A/B 测试"""
    variant = random.choice(['A', 'B'])

    if variant == 'A':
        # 配置 A: 保守策略
        return agent_a.invoke(question)
    else:
        # 配置 B: 激进策略
        return agent_b.invoke(question)
```

### 5.3 定期评估

**建议频率**:
- 完整评估: 每月1次
- 快速检查: 每周1次
- 实时监控: 持续

```bash
# 每月评估
python scripts/evaluate_rag.py --output results/$(date +%Y-%m).md

# 对比历史
python scripts/compare_evaluations.py --baseline 2025-11.json --current 2025-12.json
```

---

## 6. 常见问题与解决方案

### Q1: Context Recall 一直很低怎么办？

**诊断**:
```bash
# 检查检索结果
python scripts/debug_retrieval.py --question "你的问题"
```

**解决方案**:
1. 检查知识库是否包含相关信息
2. 增加 top_k
3. 调整混合检索权重
4. 考虑添加 Reranker

### Q2: Faithfulness 低说明什么？

**原因**:
- Prompt 不够明确
- Temperature 过高
- LLM 倾向于"编造"

**解决**:
1. 强化 Prompt 中的"基于文档"要求
2. 降低 temperature (0.3-0.5)
3. 添加验证步骤

### Q3: 响应时间太慢怎么优化？

**优先级**:
1. 启用缓存 (效果最明显)
2. 减少 top_k
3. 优化 Embedding 推理
4. 使用更快的 LLM

### Q4: 如何提高 Answer Correctness？

**策略**:
1. 改进知识库质量
2. 优化 Prompt
3. 使用更强大的 LLM (如 qwen-max)
4. 添加 Few-shot 示例

---

## 7. 优化效果预期

| 优化项 | 预期提升 | 实施难度 | 推荐优先级 |
|--------|----------|----------|------------|
| 启用缓存 | Response Time -80% | 低 | ⭐⭐⭐ |
| 调整权重 | Context Recall +5% | 低 | ⭐⭐⭐ |
| 优化 Prompt | Faithfulness +3% | 低 | ⭐⭐⭐ |
| 添加 Reranker | Context Recall +8% | 中 | ⭐⭐ |
| 清洗知识库 | All Metrics +5% | 高 | ⭐⭐ |
| 模型微调 | Answer Correctness +10% | 高 | ⭐ |

---

## 8. 优化清单

### 立即可做 (< 1小时)
- [ ] 调整 temperature 至 0.5
- [ ] 修改 system prompt 强化忠实度
- [ ] 调整混合检索权重为 0.4/0.6
- [ ] 启用 Redis 缓存

### 短期优化 (1-3天)
- [ ] 实现结果缓存
- [ ] 添加答案验证步骤
- [ ] 优化文档分块策略
- [ ] 建立监控dashboard

### 中期优化 (1-2周)
- [ ] 集成 Reranker
- [ ] 清洗和扩充知识库
- [ ] 实现动态 top_k 调整
- [ ] A/B 测试不同配置

### 长期优化 (1个月+)
- [ ] 微调 Embedding 模型
- [ ] 微调 LLM
- [ ] 实现多模态支持
- [ ] 优化系统架构 (并行、分布式)

---

## 参考资料

- [RAG 优化最佳实践](https://docs.ragas.io/en/latest/howtos/applications/optimize.html)
- [LangChain 性能优化](https://python.langchain.com/docs/guides/performance/)
- [Prompt Engineering 指南](https://platform.openai.com/docs/guides/prompt-engineering)
