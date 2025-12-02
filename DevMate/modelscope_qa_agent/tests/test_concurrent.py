"""
测试并发和多用户隔离 (T050)

测试多用户同时使用系统时的会话隔离
"""
import pytest
import asyncio
import httpx
from typing import List


# 测试配置
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0


@pytest.mark.asyncio
async def test_multi_user_isolation():
    """
    测试多用户会话隔离

    场景:
    1. 用户A和用户B同时创建会话
    2. 用户A提问并得到回答
    3. 用户B提问,验证不受用户A影响
    4. 验证两个用户的会话数据完全隔离
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # 1. 创建两个独立会话
        print("\n📝 创建用户A和用户B的会话...")

        # 用户A创建会话
        response_a = await client.post(
            f"{API_BASE_URL}/api/v2/sessions",
            json={"user_id": "user_a"}
        )
        assert response_a.status_code == 200
        session_a = response_a.json()
        session_id_a = session_a["session_id"]
        print(f"✅ 用户A会话创建成功: {session_id_a}")

        # 用户B创建会话
        response_b = await client.post(
            f"{API_BASE_URL}/api/v2/sessions",
            json={"user_id": "user_b"}
        )
        assert response_b.status_code == 200
        session_b = response_b.json()
        session_id_b = session_b["session_id"]
        print(f"✅ 用户B会话创建成功: {session_id_b}")

        # 2. 用户A提问
        print("\n📝 用户A提问...")
        question_a = "什么是LangChain?"
        response = await client.post(
            f"{API_BASE_URL}/api/question",
            json={
                "question": question_a,
                "session_id": session_id_a
            }
        )
        assert response.status_code == 200
        answer_a = response.json()
        print(f"✅ 用户A得到回答 (置信度: {answer_a.get('confidence', 0):.2f})")

        # 3. 用户B提问（完全不同的问题）
        print("\n📝 用户B提问...")
        question_b = "如何上传文档?"
        response = await client.post(
            f"{API_BASE_URL}/api/question",
            json={
                "question": question_b,
                "session_id": session_id_b
            }
        )
        assert response.status_code == 200
        answer_b = response.json()
        print(f"✅ 用户B得到回答 (置信度: {answer_b.get('confidence', 0):.2f})")

        # 4. 验证会话隔离
        print("\n📝 验证会话隔离...")

        # 获取用户A的对话历史
        response = await client.get(f"{API_BASE_URL}/api/v2/sessions/{session_id_a}/history")
        assert response.status_code == 200
        history_a = response.json()
        assert len(history_a) == 1
        assert history_a[0]["question"] == question_a
        print(f"✅ 用户A历史: {len(history_a)} 轮")

        # 获取用户B的对话历史
        response = await client.get(f"{API_BASE_URL}/api/v2/sessions/{session_id_b}/history")
        assert response.status_code == 200
        history_b = response.json()
        assert len(history_b) == 1
        assert history_b[0]["question"] == question_b
        print(f"✅ 用户B历史: {len(history_b)} 轮")

        # 5. 清理
        await client.delete(f"{API_BASE_URL}/api/v2/sessions/{session_id_a}")
        await client.delete(f"{API_BASE_URL}/api/v2/sessions/{session_id_b}")
        print("\n✅ 多用户隔离测试通过!")


@pytest.mark.asyncio
async def test_concurrent_questions():
    """
    测试并发提问

    场景:
    1. 创建多个会话
    2. 同时发送多个问题
    3. 验证所有请求都得到正确响应
    """
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        print("\n📝 创建3个并发会话...")

        # 创建会话
        session_ids = []
        for i in range(3):
            response = await client.post(
                f"{API_BASE_URL}/api/v2/sessions",
                json={"user_id": f"concurrent_user_{i}"}
            )
            assert response.status_code == 200
            session_ids.append(response.json()["session_id"])

        print(f"✅ 创建了 {len(session_ids)} 个会话")

        # 并发发送问题
        print("\n📝 并发发送问题...")
        questions = [
            "什么是RAG?",
            "如何使用向量数据库?",
            "LangChain有哪些功能?"
        ]

        tasks = []
        for session_id, question in zip(session_ids, questions):
            task = client.post(
                f"{API_BASE_URL}/api/question",
                json={
                    "question": question,
                    "session_id": session_id
                }
            )
            tasks.append(task)

        # 等待所有请求完成
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 验证所有响应
        success_count = 0
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"❌ 请求 {i+1} 失败: {response}")
            else:
                assert response.status_code == 200
                success_count += 1
                print(f"✅ 请求 {i+1} 成功")

        assert success_count == len(questions)
        print(f"\n✅ 并发测试通过! {success_count}/{len(questions)} 请求成功")

        # 清理
        for session_id in session_ids:
            await client.delete(f"{API_BASE_URL}/api/v2/sessions/{session_id}")


if __name__ == "__main__":
    print("=" * 70)
    print("多用户并发测试")
    print("=" * 70)

    asyncio.run(test_multi_user_isolation())
    asyncio.run(test_concurrent_questions())

    print("\n" + "=" * 70)
    print("✅ 所有测试通过!")
    print("=" * 70)
