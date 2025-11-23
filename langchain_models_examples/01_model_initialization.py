"""
LangChain Models - 模型初始化示例
演示不同方式初始化模型、配置参数
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_KEY", "your-api-key-here")


# ==================== 1. Chat Models 基本使用 ====================

def basic_chat_model_example():
    """Chat Model 基本示例"""
    print("=" * 50)
    print("Chat Model 基本示例")
    print("=" * 50)

    # 初始化 Chat Model
    model = ChatZhipuAI(model="glm-4.6")

    # 使用消息列表调用
    messages = [
        SystemMessage(content="你是一个有帮助的 AI 助手"),
        HumanMessage(content="什么是量子计算?")
    ]

    response = model.invoke(messages)
    print(f"\n问题: 什么是量子计算?")
    print(f"回答: {response.content[:200]}...")


# ==================== 2. 模型参数配置 ====================

def model_parameters_example():
    """模型参数配置示例"""
    print("\n" + "=" * 50)
    print("模型参数配置示例")
    print("=" * 50)

    # 完整参数配置
    model = ChatZhipuAI(
        model="glm-4.6",

        # 温度: 控制输出随机性 (0-1)
        # 0 = 确定性输出, 1 = 高度随机
        temperature=0.7,

        # 最大 tokens: 限制响应长度
        max_tokens=1024,

        # Top P: 核采样参数 (0-1)
        top_p=0.9,

        # 超时设置 (秒)
        timeout=60,

        # 最大重试次数
        max_retries=3
    )

    response = model.invoke([
        HumanMessage(content="简要解释什么是机器学习")
    ])

    print(f"\n回答: {response.content[:200]}...")


# ==================== 3. Temperature 使用指南 ====================

def temperature_examples():
    """Temperature 参数示例"""
    print("\n" + "=" * 50)
    print("Temperature 参数示例")
    print("=" * 50)

    # temperature = 0: 确定性输出
    # 适用场景: 数据提取、分类、结构化输出、代码生成
    deterministic_model = ChatZhipuAI(
        model="glm-4.6",
        temperature=0
    )

    print("\n--- Temperature = 0 (确定性输出) ---")
    response = deterministic_model.invoke([
        HumanMessage(content="将以下文本分类: '这个产品非常好用'")
    ])
    print(f"回答: {response.content}")

    # temperature = 0.5: 平衡
    # 适用场景: 客服对话、问答系统、技术文档
    balanced_model = ChatZhipuAI(
        model="glm-4.6",
        temperature=0.5
    )

    print("\n--- Temperature = 0.5 (平衡模式) ---")
    response = balanced_model.invoke([
        HumanMessage(content="如何学习 Python?")
    ])
    print(f"回答: {response.content[:200]}...")

    # temperature = 0.9: 高创造性
    # 适用场景: 创意写作、头脑风暴、故事生成
    creative_model = ChatZhipuAI(
        model="glm-4.6",
        temperature=0.9
    )

    print("\n--- Temperature = 0.9 (高创造性) ---")
    response = creative_model.invoke([
        HumanMessage(content="写一个关于未来城市的故事开头")
    ])
    print(f"回答: {response.content[:200]}...")


# ==================== 4. Max Tokens 配置 ====================

def max_tokens_examples():
    """Max Tokens 配置示例"""
    print("\n" + "=" * 50)
    print("Max Tokens 配置示例")
    print("=" * 50)

    # 短回答 (节省成本)
    short_model = ChatZhipuAI(
        model="glm-4.6",
        max_tokens=50
    )

    print("\n--- Max Tokens = 50 (短回答) ---")
    response = short_model.invoke([
        HumanMessage(content="什么是 AI?")
    ])
    print(f"回答: {response.content}")

    # 标准响应
    standard_model = ChatZhipuAI(
        model="glm-4.6",
        max_tokens=500
    )

    print("\n--- Max Tokens = 500 (标准响应) ---")
    response = standard_model.invoke([
        HumanMessage(content="解释深度学习的基本概念")
    ])
    print(f"回答: {response.content[:300]}...")

    # 长内容生成
    long_model = ChatZhipuAI(
        model="glm-4.6",
        max_tokens=2000
    )

    print("\n--- Max Tokens = 2000 (长内容) ---")
    response = long_model.invoke([
        HumanMessage(content="详细介绍 Python 编程语言的特点和应用")
    ])
    print(f"回答: {response.content[:300]}...")


# ==================== 5. 系统提示词使用 ====================

def system_prompt_example():
    """系统提示词示例"""
    print("\n" + "=" * 50)
    print("系统提示词示例")
    print("=" * 50)

    system_prompt = """你是一个专业的客户服务助手。

你的职责:
- 礼貌、专业地回答客户问题
- 如果不确定答案，诚实说明并寻求帮助
- 使用简洁、清晰的语言

你的限制:
- 不要提供医疗或法律建议
- 不要分享客户的个人信息
- 不要做出公司无法兑现的承诺
"""

    model = ChatZhipuAI(model="glm-4.6", temperature=0.7)

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="我的订单什么时候到?")
    ])

    print(f"\n客户问题: 我的订单什么时候到?")
    print(f"助手回答: {response.content}")


# ==================== 6. 对话历史管理 ====================

def conversation_history_example():
    """对话历史管理示例"""
    print("\n" + "=" * 50)
    print("对话历史管理示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.7)

    # 模拟多轮对话
    messages = [
        SystemMessage(content="你是一个友好的助手"),
        HumanMessage(content="我叫张三"),
    ]

    # 第一轮
    response1 = model.invoke(messages)
    print(f"\n用户: 我叫张三")
    print(f"助手: {response1.content}")

    # 添加对话历史
    messages.append(AIMessage(content=response1.content))
    messages.append(HumanMessage(content="我刚才叫什么名字?"))

    # 第二轮 - 模型应该记得之前的对话
    response2 = model.invoke(messages)
    print(f"\n用户: 我刚才叫什么名字?")
    print(f"助手: {response2.content}")


# ==================== 7. 不同模型选择 ====================

def model_selection_example():
    """不同模型选择示例"""
    print("\n" + "=" * 50)
    print("不同模型选择示例")
    print("=" * 50)

    # glm-4.6: 最强大，适合复杂任务
    plus_model = ChatZhipuAI(model="glm-4.6")

    print("\n--- glm-4.6 (复杂推理) ---")
    response = plus_model.invoke([
        HumanMessage(content="分析人工智能对未来教育的影响")
    ])
    print(f"回答: {response.content[:200]}...")

    # GLM-4-Flash: 快速响应，适合简单任务
    flash_model = ChatZhipuAI(model="glm-4-flash")

    print("\n--- GLM-4-Flash (快速响应) ---")
    response = flash_model.invoke([
        HumanMessage(content="今天天气怎么样?")
    ])
    print(f"回答: {response.content}")


# ==================== 8. 错误处理示例 ====================

def error_handling_example():
    """错误处理示例"""
    print("\n" + "=" * 50)
    print("错误处理示例")
    print("=" * 50)

    model = ChatZhipuAI(
        model="glm-4.6",
        max_retries=3,  # 最多重试 3 次
        timeout=60,  # 超时时间
    )

    try:
        response = model.invoke([
            HumanMessage(content="你好，请介绍一下自己")
        ])
        print(f"\n成功响应: {response.content[:200]}...")

    except Exception as e:
        print(f"\n错误: {str(e)}")


# ==================== 9. 获取响应元数据 ====================

def response_metadata_example():
    """响应元数据示例"""
    print("\n" + "=" * 50)
    print("响应元数据示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")

    response = model.invoke([
        HumanMessage(content="介绍人工智能")
    ])

    print("\n响应内容:")
    print(response.content[:200] + "...")

    print("\n响应元数据:")
    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        print(f"  模型: {metadata.get('model', 'N/A')}")
        print(f"  Token使用: {metadata.get('token_usage', 'N/A')}")
        print(f"  完成原因: {metadata.get('finish_reason', 'N/A')}")

    print(f"\n消息 ID: {response.id if hasattr(response, 'id') else 'N/A'}")


# ==================== 10. 流式处理预览 ====================

def streaming_preview():
    """流式处理预览"""
    print("\n" + "=" * 50)
    print("流式处理预览")
    print("=" * 50)

    model = ChatZhipuAI(
        model="glm-4.6",
        streaming=True
    )

    print("\n生成中: ", end="", flush=True)
    for chunk in model.stream([HumanMessage(content="写一首关于春天的短诗")]):
        print(chunk.content, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    try:
        basic_chat_model_example()
        model_parameters_example()
        temperature_examples()
        max_tokens_examples()
        system_prompt_example()
        conversation_history_example()
        model_selection_example()
        error_handling_example()
        response_metadata_example()
        streaming_preview()

        print("\n" + "=" * 50)
        print("所有示例运行完成!")
        print("=" * 50)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
