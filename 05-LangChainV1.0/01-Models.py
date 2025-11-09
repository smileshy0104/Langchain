
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatAnthropic,ChatZhipuAI,ChatOpenAI

model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key="your-api-key",  # 或从环境变量读取
    temperature=0.7,
    max_tokens=1024
)

model = ChatZhipuAI(
    model="glm-4.6",
    temperature=0.3,
    api_key="your-api-key",
)

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    max_tokens=1000,
    time_out=60,
)

# Anthropic Claude
model = init_chat_model(
    model="claude-3-5-sonnet-20241022",
    model_provider="anthropic",
    temperature=0.7
)

# OpenAI GPT
openai_model = init_chat_model(
    model="gpt-4o",
    model_provider="openai",
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    temperature=0.5,
    max_tokens=1000,
    time_out=60,
    max_retries=3
)

# Google Gemini
google_model = init_chat_model(
    model="gemini-2.0-flash-exp",
    model_provider="google_genai",
    temperature=0,
)