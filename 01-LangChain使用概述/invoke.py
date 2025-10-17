import os
import dotenv
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatZhipuAI
dotenv.load_dotenv()

# 检查API密钥
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key or api_key == "your-zhipu-api-key-here":
    print("❌ 错误：请在.env文件中设置您的ZHIPUAI_API_KEY")
    exit(1)

# 初始化GLM大模型
chat_model = ChatZhipuAI(
    model="glm-4.6",
    temperature=0.7,
    api_key=api_key
)

# 创建消息
messages = [HumanMessage(content="你好，请介绍一下自己")]

# 非流式调用GLM获取响应
response = chat_model.invoke(messages)

# 打印响应内容
print("🤖 GLM-4 回答：")
print(response.content)