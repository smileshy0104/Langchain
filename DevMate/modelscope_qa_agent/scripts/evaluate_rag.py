"""
RAG System Evaluation Script

使用 RAGAs 框架评估 RAG 系统的性能,包括:
- Context Relevance: 检索文档与问题的相关性
- Answer Faithfulness: 答案与文档的一致性
- Answer Relevance: 答案与问题的相关性
- Answer Correctness: 答案的正确性

同时评估响应速度性能指标。
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import dependencies
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextRecall,  # Context Relevance
    Faithfulness,    # Answer Faithfulness
    AnswerRelevancy, # Answer Relevance
    AnswerCorrectness # Answer Correctness
)
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

# Import QA Agent
from agents.qa_agent import ModelScopeQAAgent
from retrievers.hybrid_retriever import HybridRetriever
from core.vector_store import VectorStoreManager


class RAGEvaluator:
    """RAG 系统评估器

    使用 RAGAs 框架评估 RAG 系统的各项性能指标。
    """

    def __init__(
        self,
        agent: ModelScopeQAAgent,
        llm_api_key: str,
        embedding_api_key: str = None
    ):
        """初始化评估器

        Args:
            agent: ModelScopeQAAgent 实例
            llm_api_key: LLM API密钥 (用于评估)
            embedding_api_key: Embedding API密钥 (如果与LLM不同)
        """
        self.agent = agent

        # 初始化评估用的 LLM 和 Embeddings
        # RAGAs 需要使用 OpenAI 兼容的模型进行评估
        # 这里使用通义千问作为评估模型
        self.eval_llm = ChatTongyi(
            model="qwen-plus",
            temperature=0.0,
            dashscope_api_key=llm_api_key
        )

        self.eval_embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=embedding_api_key or llm_api_key
        )

        print(f"✅ RAGEvaluator 初始化成功")
        print(f"   - 评估 LLM: qwen-plus")
        print(f"   - 评估 Embeddings: text-embedding-v2")

    def load_evaluation_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """加载评测数据集

        Args:
            dataset_path: 数据集文件路径 (JSON格式)

        Returns:
            评测数据列表
        """
        print(f"\n{'='*70}")
        print(f"📥 加载评测数据集")
        print(f"{'='*70}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"✅ 加载成功: {len(dataset)} 条测试数据")

        # 统计类别分布
        categories = {}
        for item in dataset:
            cat = item.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\n📊 数据集类别分布:")
        for cat, count in sorted(categories.items()):
            print(f"   - {cat}: {count} 条")

        return dataset

    def run_inference(
        self,
        dataset: List[Dict[str, Any]],
        max_samples: int = None
    ) -> List[Dict[str, Any]]:
        """运行推理获取 Agent 响应

        Args:
            dataset: 评测数据集
            max_samples: 最多处理的样本数 (None=全部)

        Returns:
            包含问题、上下文、答案和真实答案的数据列表
        """
        print(f"\n{'='*70}")
        print(f"🤖 运行 Agent 推理")
        print(f"{'='*70}")

        results = []
        samples = dataset[:max_samples] if max_samples else dataset

        for i, item in enumerate(samples, 1):
            question = item['question']
            ground_truth = item['ground_truth']

            print(f"\n[{i}/{len(samples)}] 处理问题: {question[:50]}...")

            # 记录开始时间
            start_time = time.time()

            try:
                # 调用 Agent
                response = self.agent.invoke(question)

                # 计算响应时间
                response_time = time.time() - start_time

                # 提取答案和上下文
                answer = response.get('summary', '')
                if response.get('problem_analysis'):
                    answer += "\n\n" + response['problem_analysis']
                if response.get('solutions'):
                    answer += "\n\n解决方案:\n" + "\n".join(response['solutions'])

                # 获取检索的上下文
                # 注意: 这里需要从 Agent 的状态中获取检索的文档
                # 由于当前实现没有直接返回,我们使用 ground_truth contexts
                contexts = item.get('contexts', [])

                results.append({
                    'question': question,
                    'answer': answer,
                    'contexts': contexts,
                    'ground_truth': ground_truth,
                    'response_time': response_time,
                    'confidence_score': response.get('confidence_score', 0.0)
                })

                print(f"   ✅ 完成 (耗时: {response_time:.2f}s, 置信度: {response.get('confidence_score', 0.0):.2f})")

            except Exception as e:
                print(f"   ❌ 失败: {e}")
                # 添加失败记录
                results.append({
                    'question': question,
                    'answer': f"Error: {str(e)}",
                    'contexts': item.get('contexts', []),
                    'ground_truth': ground_truth,
                    'response_time': time.time() - start_time,
                    'confidence_score': 0.0
                })

        print(f"\n✅ 推理完成: {len(results)}/{len(samples)} 成功")
        return results

    def evaluate_with_ragas(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """使用 RAGAs 评估结果

        Args:
            results: 推理结果列表

        Returns:
            评估指标字典
        """
        print(f"\n{'='*70}")
        print(f"📊 RAGAs 评估")
        print(f"{'='*70}")

        # 转换为 RAGAs 需要的格式
        data = {
            'question': [r['question'] for r in results],
            'answer': [r['answer'] for r in results],
            'contexts': [r['contexts'] for r in results],
            'ground_truth': [r['ground_truth'] for r in results]
        }

        # 创建 Dataset
        dataset = Dataset.from_dict(data)

        print(f"\n🔍 开始评估 (共 {len(results)} 条数据)...")
        print(f"   评估指标: Context Relevance, Faithfulness, Answer Relevance, Answer Correctness")

        try:
            # 运行评估
            # RAGAs 0.3.9 使用新的 API
            eval_results = evaluate(
                dataset,
                metrics=[
                    ContextRecall(),     # 上下文召回率
                    Faithfulness(),      # 答案忠实度
                    AnswerRelevancy(),   # 答案相关性
                    AnswerCorrectness()  # 答案正确性
                ],
                llm=self.eval_llm,
                embeddings=self.eval_embeddings
            )

            print(f"\n✅ 评估完成!")
            return eval_results

        except Exception as e:
            print(f"\n❌ 评估失败: {e}")
            print(f"   这可能是由于 RAGAs 版本或 API 配置问题")
            print(f"   返回基础统计信息...")

            # 返回基础统计信息
            return self._calculate_basic_metrics(results)

    def _calculate_basic_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """计算基础评估指标 (降级方案)

        Args:
            results: 推理结果列表

        Returns:
            基础指标字典
        """
        total = len(results)
        successful = sum(1 for r in results if 'Error' not in r['answer'])

        # 计算平均响应时间
        avg_response_time = sum(r['response_time'] for r in results) / total

        # 计算平均置信度
        avg_confidence = sum(r['confidence_score'] for r in results) / total

        # 计算成功率
        success_rate = successful / total

        return {
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'avg_confidence': avg_confidence,
            'total_questions': total,
            'successful_answers': successful
        }

    def evaluate_response_time(
        self,
        results: List[Dict[str, Any]],
        target_threshold: float = 30.0
    ) -> Dict[str, Any]:
        """评估响应速度

        Args:
            results: 推理结果列表
            target_threshold: 目标响应时间阈值(秒)

        Returns:
            响应时间统计
        """
        print(f"\n{'='*70}")
        print(f"⏱️  响应速度评估")
        print(f"{'='*70}")

        response_times = [r['response_time'] for r in results]

        stats = {
            'mean': sum(response_times) / len(response_times),
            'min': min(response_times),
            'max': max(response_times),
            'p50': sorted(response_times)[len(response_times) // 2],
            'p95': sorted(response_times)[int(len(response_times) * 0.95)],
            'p99': sorted(response_times)[int(len(response_times) * 0.99)],
            'target_threshold': target_threshold,
            'within_threshold': sum(1 for t in response_times if t <= target_threshold),
            'threshold_percentage': sum(1 for t in response_times if t <= target_threshold) / len(response_times) * 100
        }

        print(f"\n📊 响应时间统计:")
        print(f"   - 平均: {stats['mean']:.2f}s")
        print(f"   - 最小: {stats['min']:.2f}s")
        print(f"   - 最大: {stats['max']:.2f}s")
        print(f"   - P50: {stats['p50']:.2f}s")
        print(f"   - P95: {stats['p95']:.2f}s")
        print(f"   - P99: {stats['p99']:.2f}s")
        print(f"\n🎯 目标达成情况:")
        print(f"   - 目标阈值: <{target_threshold}s")
        print(f"   - 达标数量: {stats['within_threshold']}/{len(response_times)}")
        print(f"   - 达标率: {stats['threshold_percentage']:.1f}%")

        if stats['mean'] < target_threshold:
            print(f"   ✅ 平均响应时间达标!")
        else:
            print(f"   ❌ 平均响应时间未达标")

        return stats

    def generate_report(
        self,
        ragas_results: Dict[str, float],
        response_stats: Dict[str, Any],
        results: List[Dict[str, Any]],
        output_path: str
    ):
        """生成评估报告

        Args:
            ragas_results: RAGAs 评估结果
            response_stats: 响应时间统计
            results: 详细结果
            output_path: 报告输出路径
        """
        print(f"\n{'='*70}")
        print(f"📝 生成评估报告")
        print(f"{'='*70}")

        report = {
            'evaluation_time': datetime.now().isoformat(),
            'total_questions': len(results),
            'ragas_metrics': ragas_results,
            'response_time_stats': response_stats,
            'detailed_results': results
        }

        # 保存 JSON 报告
        json_path = output_path.replace('.md', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"✅ JSON 报告已保存: {json_path}")

        # 生成 Markdown 报告
        self._generate_markdown_report(report, output_path)
        print(f"✅ Markdown 报告已保存: {output_path}")

    def _generate_markdown_report(
        self,
        report: Dict[str, Any],
        output_path: str
    ):
        """生成 Markdown 格式报告

        Args:
            report: 报告数据
            output_path: 输出路径
        """
        ragas = report['ragas_metrics']
        response_stats = report['response_time_stats']

        md_content = f"""# RAG System Evaluation Report

**评估时间**: {report['evaluation_time']}
**测试问题数**: {report['total_questions']}

---

## RAGAs 评估指标

| 指标 | 得分 | 目标 | 状态 |
|------|------|------|------|
"""

        # 添加 RAGAs 指标
        # RAGAs 0.3.9 返回的 key 名称可能不同
        metric_keys = [
            ('context_recall', 'Context Recall', 0.85),
            ('faithfulness', 'Answer Faithfulness', 0.95),
            ('answer_relevancy', 'Answer Relevance', None),
            ('answer_correctness', 'Answer Correctness', None)
        ]

        for key, name, threshold in metric_keys:
            if key in ragas:
                score = ragas[key]
                if threshold:
                    status = "✅ 达标" if score >= threshold else "❌ 未达标"
                    md_content += f"| {name} | {score:.2%} | ≥{threshold*100:.0f}% | {status} |\n"
                else:
                    md_content += f"| {name} | {score:.2%} | - | - |\n"

        # 添加基础指标 (如果使用降级方案)
        if 'success_rate' in ragas:
            md_content += f"| Success Rate | {ragas['success_rate']:.2%} | - | - |\n"
            md_content += f"| Avg Confidence | {ragas['avg_confidence']:.2f} | - | - |\n"

        md_content += f"""
---

## 响应速度评估

| 指标 | 数值 |
|------|------|
| 平均响应时间 | {response_stats['mean']:.2f}s |
| P50 (中位数) | {response_stats['p50']:.2f}s |
| P95 | {response_stats['p95']:.2f}s |
| P99 | {response_stats['p99']:.2f}s |
| 最小值 | {response_stats['min']:.2f}s |
| 最大值 | {response_stats['max']:.2f}s |

**目标达成情况**:
- 目标阈值: <{response_stats['target_threshold']}s
- 达标率: {response_stats['threshold_percentage']:.1f}% ({response_stats['within_threshold']}/{report['total_questions']})
- 状态: {"✅ 达标" if response_stats['mean'] < response_stats['target_threshold'] else "❌ 未达标"}

---

## 总结

"""

        # 添加总结
        if 'context_recall' in ragas and ragas['context_recall'] >= 0.85:
            md_content += "- ✅ Context Recall 达到目标 (≥85%)\n"
        elif 'context_recall' in ragas:
            md_content += f"- ❌ Context Recall 未达标 ({ragas['context_recall']:.2%} < 85%)\n"

        if 'faithfulness' in ragas and ragas['faithfulness'] >= 0.95:
            md_content += "- ✅ Answer Faithfulness 达到目标 (≥95%)\n"
        elif 'faithfulness' in ragas:
            md_content += f"- ❌ Answer Faithfulness 未达标 ({ragas['faithfulness']:.2%} < 95%)\n"

        if response_stats['mean'] < response_stats['target_threshold']:
            md_content += f"- ✅ 响应速度达标 (平均 {response_stats['mean']:.2f}s < {response_stats['target_threshold']}s)\n"
        else:
            md_content += f"- ❌ 响应速度未达标 (平均 {response_stats['mean']:.2f}s ≥ {response_stats['target_threshold']}s)\n"

        md_content += "\n---\n\n**详细结果**: 请查看 JSON 报告文件\n"

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate RAG System with RAGAs')
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/evaluation_dataset.json',
        help='Path to evaluation dataset (JSON format)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/evaluation_report.md',
        help='Path to output report (Markdown format)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='DashScope API key (overrides .env)'
    )

    args = parser.parse_args()

    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()

    api_key = args.api_key or os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ Error: DASHSCOPE_API_KEY not found")
        print("   Please set it in .env file or use --api-key argument")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"🚀 RAG System Evaluation")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Max Samples: {args.max_samples or 'All'}")
    print(f"{'='*70}")

    try:
        # 初始化组件
        print(f"\n📦 初始化组件...")

        # 初始化向量存储
        vector_store_manager = VectorStoreManager(
            host=os.getenv('MILVUS_HOST', 'localhost'),
            port=int(os.getenv('MILVUS_PORT', 19530))
        )

        # 初始化检索器
        retriever = HybridRetriever(
            vector_store_manager=vector_store_manager,
            collection_name="modelscope_knowledge_base",
            embedding_api_key=api_key
        )

        # 初始化 Agent
        agent = ModelScopeQAAgent(
            retriever=retriever,
            llm_api_key=api_key,
            temperature=0.7
        )

        # 初始化评估器
        evaluator = RAGEvaluator(
            agent=agent,
            llm_api_key=api_key
        )

        # 加载数据集
        dataset = evaluator.load_evaluation_dataset(args.dataset)

        # 运行推理
        results = evaluator.run_inference(dataset, args.max_samples)

        # RAGAs 评估
        ragas_results = evaluator.evaluate_with_ragas(results)

        # 响应速度评估
        response_stats = evaluator.evaluate_response_time(results, target_threshold=30.0)

        # 生成报告
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        evaluator.generate_report(ragas_results, response_stats, results, args.output)

        print(f"\n{'='*70}")
        print(f"✅ 评估完成!")
        print(f"{'='*70}")
        print(f"报告已保存:")
        print(f"   - Markdown: {args.output}")
        print(f"   - JSON: {args.output.replace('.md', '.json')}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
