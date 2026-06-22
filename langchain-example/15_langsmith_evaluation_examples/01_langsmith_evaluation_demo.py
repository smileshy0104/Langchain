"""
LangSmith Evaluation - 基础调用示例

本示例演示:
1. 使用 LangSmith Client 创建或复用数据集
2. 写一个 target function 作为被评测对象
3. 使用本目录 model_config.py 配置并调用真实大模型
4. 使用 code evaluator 运行一次离线评测实验
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from langsmith import Client, evaluate


ENV_FILE = Path(__file__).with_name(".env")
DATASET_NAME = "Test_Demo"
EXPERIMENT_PREFIX = "Test_Demo Real Evaluation"


EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {"question": "Which country is Mount Kilimanjaro located in?"},
        "outputs": {
            "answer": "Mount Kilimanjaro is located in Tanzania.",
            "must_include": ["Tanzania"],
        },
    },
    {
        "inputs": {"question": "What is Earth's lowest point?"},
        "outputs": {
            "answer": "Earth's lowest point is The Dead Sea.",
            "must_include": ["Dead Sea"],
        },
    },
]


def load_env_file(env_file: Path = ENV_FILE) -> None:
    """从本目录 .env 加载环境变量，已存在的系统环境变量优先。"""
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def ensure_environment() -> None:
    """检查运行评测所需的 LangSmith 环境变量。"""
    load_env_file()
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    required_vars = ["LANGSMITH_API_KEY"]
    missing = [name for name in required_vars if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "缺少必要环境变量: "
            + ", ".join(missing)
            + "\n请先执行: export LANGSMITH_API_KEY='<your-langsmith-api-key>'"
        )


def get_or_create_dataset(client: Client, dataset_name: str) -> Any:
    """创建或复用 LangSmith 数据集。"""
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"复用已有数据集: {dataset.name}")
        return dataset
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="LangSmith evaluation demo dataset.",
        )
        print(f"创建新数据集: {dataset.name}")
        return dataset


def seed_examples_if_empty(client: Client, dataset_id: str) -> None:
    """写入示例；若样本已存在，则同步为当前 demo 的 reference outputs。"""
    existing_examples = list(client.list_examples(dataset_id=dataset_id))
    if existing_examples:
        for existing, example in zip(existing_examples, EXAMPLES):
            client.update_example(
                existing.id,
                inputs=example["inputs"],
                outputs=example["outputs"],
            )
        synced_count = min(len(existing_examples), len(EXAMPLES))
        print(f"数据集已有样本，已同步前 {synced_count} 条样本。")
        return

    client.create_examples(dataset_id=dataset_id, examples=EXAMPLES)
    print(f"已导入 {len(EXAMPLES)} 条样本。")


def _extract_answer(outputs: Any) -> str:
    """从不同 target 输出形态中提取最终回答。"""
    if isinstance(outputs, dict):
        answer = (
            outputs.get("response")
            or outputs.get("answer")
            or outputs.get("output")
        )
        return str(answer or "").strip()

    if hasattr(outputs, "outputs") and isinstance(outputs.outputs, dict):
        answer = (
            outputs.outputs.get("response")
            or outputs.outputs.get("answer")
            or outputs.outputs.get("output")
        )
        return str(answer or "").strip()

    return str(outputs or "").strip()


def _extract_reference(reference_outputs: Optional[dict[str, Any]]) -> str:
    """从 reference outputs 中提取参考答案。"""
    if not isinstance(reference_outputs, dict):
        return ""

    for key in ("response", "answer", "output", "expected_answer"):
        value = reference_outputs.get(key)
        if value:
            return str(value).strip()

    return ""


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    """兼容 LangChain 对象和普通 dict 的字段读取。"""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalize_usage_metadata(usage: Any) -> dict[str, Any]:
    """将模型返回的 usage 转为 LangSmith 常见 usage_metadata。"""
    if not usage:
        return {}

    input_tokens = (
        _get_attr_or_key(usage, "input_tokens", None)
        or _get_attr_or_key(usage, "prompt_tokens", 0)
        or 0
    )
    output_tokens = (
        _get_attr_or_key(usage, "output_tokens", None)
        or _get_attr_or_key(usage, "completion_tokens", 0)
        or 0
    )
    total_tokens = _get_attr_or_key(usage, "total_tokens", None)

    metadata: dict[str, Any] = {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens or input_tokens + output_tokens),
    }

    prompt_details = _get_attr_or_key(usage, "prompt_tokens_details", None)
    cached_tokens = _get_attr_or_key(prompt_details, "cached_tokens", None)
    if cached_tokens:
        metadata["input_token_details"] = {"cache_read": int(cached_tokens)}

    completion_details = _get_attr_or_key(usage, "completion_tokens_details", None)
    reasoning_tokens = _get_attr_or_key(completion_details, "reasoning_tokens", None)
    if reasoning_tokens:
        metadata["output_token_details"] = {"reasoning": int(reasoning_tokens)}

    return metadata


def _message_content(response: Any) -> str:
    """提取 LangChain AIMessage 或普通响应中的文本。"""
    if hasattr(response, "content"):
        return str(response.content or "").strip()
    return str(response or "").strip()


def _response_usage_metadata(response: Any) -> dict[str, Any]:
    """从 LangChain response 中提取 usage metadata。"""
    usage_metadata = _get_attr_or_key(response, "usage_metadata", None)
    if usage_metadata:
        return _normalize_usage_metadata(usage_metadata)

    response_metadata = _get_attr_or_key(response, "response_metadata", None)
    token_usage = _get_attr_or_key(response_metadata, "token_usage", None)
    return _normalize_usage_metadata(token_usage)


def local_stub_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """默认 target: 不调用模型，只用于稳定演示 LangSmith 评测链路。"""
    known_answers = {
        "Which country is Mount Kilimanjaro located in?": (
            "Mount Kilimanjaro is located in Tanzania."
        ),
        "What is Earth's lowest point?": "Earth's lowest point is The Dead Sea.",
    }
    question = inputs["question"]
    session_id = f"langsmith_eval_{uuid4().hex}"
    return {
        "response": known_answers.get(question, "I don't know the answer."),
        "rag_trace": {},
        "session_id": session_id,
    }


def langchain_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """真实模型 target: 使用本目录 model_config.py 调用大模型。"""
    from model_config import create_configured_model

    model = create_configured_model()
    session_id = f"langsmith_eval_{uuid4().hex}"

    response = model.invoke(
        [
            ("system", "Answer the question accurately and concisely."),
            ("user", inputs["question"]),
        ]
    )
    usage_metadata = _response_usage_metadata(response)
    if not usage_metadata:
        print("Model response did not include usage metadata.")

    return {
        "response": _message_content(response),
        "rag_trace": {},
        "session_id": session_id,
        "model": os.getenv("LANGCHAIN_DEFAULT_MODEL", ""),
        "provider": os.getenv("LANGCHAIN_MODEL_PROVIDER", "custom"),
        "usage_metadata": usage_metadata,
    }


def select_target() -> Any:
    """通过 LANGSMITH_DEMO_TARGET 切换本地 stub 或真实 LangChain 模型调用。"""
    mode = os.getenv("LANGSMITH_DEMO_TARGET", "langchain").strip().lower()
    if mode in {"langchain", "model", "real"}:
        print("Target: LangChain configured model call")
        return langchain_target

    print("Target: local stub")
    return local_stub_target


def print_runtime_diagnostics() -> None:
    """打印运行路径和 target 模式，方便排查是否跑到旧脚本。"""
    print(f"Script: {Path(__file__).resolve()}")
    print(f"Python: {sys.executable}")
    print(f"LANGSMITH_DEMO_TARGET={os.getenv('LANGSMITH_DEMO_TARGET', 'langchain')}")
    print(f"LANGCHAIN_MODEL_PROVIDER={os.getenv('LANGCHAIN_MODEL_PROVIDER', 'custom')}")
    print(f"LANGCHAIN_DEFAULT_MODEL={os.getenv('LANGCHAIN_DEFAULT_MODEL', '')}")
    print(f"langchain_openai loaded={'langchain_openai' in sys.modules}")


def exact_match(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> bool:
    """检查最终回答是否和 reference 完全一致。"""
    return _extract_answer(outputs) == _extract_reference(reference_outputs)


def contains_required_answer(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    """检查回答是否包含 reference 中要求出现的关键词。"""
    answer = _extract_answer(outputs).lower()
    required_terms = reference_outputs.get("must_include") or []
    missing = [term for term in required_terms if str(term).lower() not in answer]
    return {
        "key": "contains_required_answer",
        "score": len(missing) == 0,
        "comment": "missing: " + ", ".join(missing) if missing else "all terms found",
    }


def custom_evaluator(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> bool:
    """评估最终答案，不评估检索块或中间 trace。"""
    answer = _extract_answer(outputs)
    if not answer:
        return False

    if "Retrieved Chunks:" in answer:
        return False

    reference = _extract_reference(reference_outputs)
    if not reference:
        return True

    answer_chars = {ch for ch in answer if not ch.isspace()}
    ref_chars = {ch for ch in reference if not ch.isspace()}
    if not answer_chars or not ref_chars:
        return False

    overlap = len(answer_chars & ref_chars) / max(1, len(ref_chars))
    return overlap >= 0.2


def main() -> None:
    ensure_environment()
    print_runtime_diagnostics()

    client = Client()
    dataset = get_or_create_dataset(client, DATASET_NAME)
    seed_examples_if_empty(client, str(dataset.id))

    print("\n开始运行 LangSmith evaluation...")
    results = evaluate(
        select_target(),
        data=DATASET_NAME,
        evaluators=[custom_evaluator, exact_match, contains_required_answer],
        experiment_prefix=EXPERIMENT_PREFIX,
        client=client,
    )

    print("评测已提交。")
    print(f"Experiment results: {results}")


if __name__ == "__main__":
    main()
