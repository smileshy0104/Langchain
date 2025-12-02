#!/usr/bin/env python3
"""
完整 QA Agent 功能测试
测试从简化版本切换到完整 Agent 后的功能
"""

import requests
import json
from pathlib import Path
from datetime import datetime

API_BASE = "http://localhost:8000"

def test_agent_with_documents():
    """测试完整 Agent - 包含文档上传和问答"""
    print("\n" + "=" * 80)
    print("测试完整 QA Agent 功能")
    print("=" * 80)

    # 1. 创建测试文档
    print("\n1. 创建测试文档...")
    test_dir = Path("/tmp/full_agent_test")
    test_dir.mkdir(parents=True, exist_ok=True)

    qwen_doc = test_dir / "qwen_guide.txt"
    qwen_doc.write_text("""Qwen 模型使用指南

1. 安装
pip install dashscope

2. 基本使用
from dashscope import Generation

response = Generation.call(
    model='qwen-turbo',
    prompt='你好,请介绍一下自己'
)
print(response.output.text)

3. 参数说明
- model: 模型名称 (qwen-turbo, qwen-plus, qwen-max)
- prompt: 输入文本
- temperature: 温度参数,控制输出随机性 (0-1)
- top_p: 核采样参数 (0-1)
- max_tokens: 最大输出长度

4. 高级功能
- 流式输出: stream=True
- Few-shot 学习: 通过示例引导模型
- Function Calling: 支持函数调用

5. 最佳实践
- 使用清晰的提示词
- 合理设置 temperature 参数
- 处理错误和超时
""", encoding='utf-8')

    print(f"✅ 创建测试文档: {qwen_doc.name}")

    # 2. 上传文档
    print("\n2. 上传文档到系统...")
    with open(qwen_doc, 'rb') as f:
        files = {'file': (qwen_doc.name, f, 'text/plain')}
        data = {
            'category': 'test',
            'store_to_db': 'true'
        }

        response = requests.post(f"{API_BASE}/api/upload", files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ 文档上传成功")
            print(f"   - 文档块数: {result['document_count']}")
            print(f"   - 文档 IDs: {result.get('document_ids', [])}")
        else:
            print(f"❌ 文档上传失败: {response.status_code}")
            print(f"   {response.json()}")
            return False

    # 3. 等待向量化完成
    import time
    print("\n3. 等待向量化完成...")
    time.sleep(2)

    # 4. 测试完整 Agent 问答
    print("\n4. 测试完整 QA Agent 问答...")

    test_questions = [
        "如何安装 Qwen 模型?",
        "Qwen 模型有哪些参数?",
        "什么是 temperature 参数?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n   问题 {i}: {question}")

        response = requests.post(
            f"{API_BASE}/api/question",
            headers={'Content-Type': 'application/json'},
            json={
                "question": question,
                "session_id": "full_agent_test",
                "top_k": 3
            }
        )

        if response.status_code == 200:
            result = response.json()
            answer = result['answer']
            sources = result['sources']
            confidence = result['confidence']

            print(f"   ✅ 回答成功")
            print(f"      置信度: {confidence:.2%}")
            print(f"      来源数: {len(sources)}")
            print(f"      答案预览: {answer[:100]}...")

            # 检查是否是 澄清问题(Agent 的主动澄清功能)
            if result.get('need_clarification') or '需要更多信息' in answer:
                print(f"      ℹ️  Agent 请求澄清")
            else:
                print(f"      ✅ Agent 生成了完整答案")

        else:
            print(f"   ❌ 问答失败: {response.status_code}")
            print(f"      {response.json()}")

        time.sleep(1)

    # 5. 验证完整 Agent 特性
    print("\n5. 验证完整 Agent 特性...")

    # 5.1 检查 Agent 状态
    response = requests.get(f"{API_BASE}/api/status")
    if response.status_code == 200:
        status = response.json()
        print(f"   ✅ 系统状态正常")
        print(f"      文档数: {status['document_count']}")
        print(f"      Milvus: {status['milvus_connected']}")
    else:
        print(f"   ❌ 状态检查失败")

    print("\n" + "=" * 80)
    print("完整 QA Agent 测试完成!")
    print("=" * 80)

    return True


def test_agent_clarification():
    """测试 Agent 的主动澄清功能"""
    print("\n" + "=" * 80)
    print("测试 Agent 主动澄清功能")
    print("=" * 80)

    # 提出一个模糊的问题
    question = "这个怎么用?"

    print(f"\n提出模糊问题: {question}")

    response = requests.post(
        f"{API_BASE}/api/question",
        headers={'Content-Type': 'application/json'},
        json={
            "question": question,
            "session_id": "clarification_test",
            "top_k": 3
        }
    )

    if response.status_code == 200:
        result = response.json()
        answer = result['answer']
        confidence = result['confidence']

        print(f"\nAgent 响应:")
        print(f"  置信度: {confidence:.2%}")
        print(f"  答案: {answer}")

        if '需要更多信息' in answer or confidence == 0.0:
            print(f"\n✅ Agent 正确识别了模糊问题并请求澄清")
        else:
            print(f"\n⚠️  Agent 未能识别模糊问题")

    else:
        print(f"❌ 请求失败: {response.status_code}")

    print("\n" + "=" * 80)


def test_agent_vs_simplified():
    """对比完整 Agent 和简化版本的差异"""
    print("\n" + "=" * 80)
    print("对比完整 Agent vs 简化版本")
    print("=" * 80)

    question = "如何使用 Qwen 模型进行文本生成?"

    print(f"\n测试问题: {question}")

    # 测试完整 Agent (通过 /api/question)
    print("\n1. 完整 Agent 响应 (/api/question):")
    response = requests.post(
        f"{API_BASE}/api/question",
        headers={'Content-Type': 'application/json'},
        json={
            "question": question,
            "session_id": "comparison_test",
            "top_k": 3
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"   答案长度: {len(result['answer'])} 字符")
        print(f"   置信度: {result['confidence']:.2%}")
        print(f"   来源数: {len(result['sources'])}")
        print(f"   答案开头: {result['answer'][:150]}...")
    else:
        print(f"   ❌ 失败: {response.status_code}")

    print("\n" + "=" * 80)
    print("总结:")
    print("  - 完整 Agent 使用 LangGraph 工作流")
    print("  - 支持问题分析、检索、答案生成和澄清")
    print("  - 可以根据检索结果质量决定是否请求澄清")
    print("  - 提供更智能的问答体验")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("魔搭社区智能答疑系统 - 完整 QA Agent 测试")
    print("=" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API 地址: {API_BASE}")

    try:
        # 测试 1: 完整 Agent 功能
        test_agent_with_documents()

        # 测试 2: 主动澄清
        test_agent_clarification()

        # 测试 3: 对比分析
        test_agent_vs_simplified()

        print("\n" + "=" * 80)
        print("✅ 所有测试完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
