# LangChain 学习与智能体实践集合

本仓库是一个围绕 LangChain、LangGraph、Agent、RAG、ModelScope 智能问答和 PyTorch 基础实践整理的学习与示例项目集合。内容既包含可运行的 Python 示例，也包含中文教程、章节总结、项目规格文档和端到端智能体应用。

## 项目内容

- `langchain-example/`: LangChain 主题示例集合，覆盖 Models、Messages、Tools、Agents、Memory、Streaming、Structured Output、MCP、Human-in-the-loop、Multi-agent、Retrieval Memory 等方向。
- `DevMate/modelscope_qa_agent/`: 基于 LangChain/LangGraph、Milvus、Redis、DashScope/Qwen 的魔搭社区智能答疑 Agent 项目，包含 API、RAG、爬虫、检索、评估和测试模块。
- `langchain-docs/`: LangChain v1 相关中文详细指南，按主题拆分为 Agents、Models、Messages、Tools、Memory、Streaming、Middleware、MCP、Multi-agent 等文档。
- `agent-docs/`: 《Hello Agents》章节通俗总结、习题解答和 LangChain v1 转换指南。
- `agent-langchain-code/`: 《Hello Agents》相关章节的 LangChain 代码实现，包含 ReAct、Plan-and-Solve、Reflection 等智能体范式示例。
- `langchain-demo-old/`: 早期 LangChain 示例归档，包含基础调用、Model IO、Chains、Agents 和 LangChain v1.0 入门示例。
- `pytorch-docs/`: PyTorch 教程、分类流程文档和一个可运行的 PyTorch 项目样例。
- `claude-skill/`: SQL 转 GORM 相关技能与测试素材。
- `script/`: 图片生成、图片校验等辅助脚本。

## 目录结构

```text
.
├── DevMate/
│   ├── modelscope_qa_agent/      # 魔搭社区智能答疑 Agent
│   └── specs/                    # 需求、计划、任务拆解
├── agent-docs/                   # Hello Agents 中文学习资料
├── agent-langchain-code/         # Hello Agents 章节代码
├── claude-skill/                 # SQL to GORM 技能素材
├── langchain-demo-old/           # 旧版 LangChain 示例归档
├── langchain-docs/               # LangChain 中文详细指南
├── langchain-example/            # LangChain 主题示例集合
├── pytorch-docs/                 # PyTorch 教程与示例项目
├── script/                       # 辅助脚本
├── .env.example                  # 根目录环境变量模板
└── pyrightconfig.json            # Python 类型检查配置
```

## 环境要求

- Python 3.11 推荐；部分历史示例可在 Python 3.8+ 运行。
- 建议使用 Conda 或 venv 创建独立环境。
- 需要按示例选择配置模型服务 API Key，例如 OpenAI、智谱 AI、Anthropic 或 DashScope。
- 运行 `DevMate/modelscope_qa_agent` 时，可能需要额外准备 Milvus、Redis、MySQL 等基础设施。

当前 `pyrightconfig.json` 指向的开发环境为：

```text
venvPath: /opt/homebrew/Caskroom/miniconda/base/envs
venv: langchain-env
pythonVersion: 3.11
```

## 快速开始

### 1. 创建并激活环境

```bash
conda create -n langchain-env python=3.11 -y
conda activate langchain-env
```

也可以使用已有的 `langchain-env` 环境。

### 2. 配置环境变量

```bash
cp .env.example .env
```

按需填写：

```env
OPENAI_API_KEY=your-openai-api-key-here
ZHIPUAI_API_KEY=your-zhipu-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

魔搭 QA Agent 相关配置请参考：

```bash
cp DevMate/modelscope_qa_agent/.env.example DevMate/modelscope_qa_agent/.env
```

### 3. 安装某个子项目依赖

本仓库没有统一的根依赖文件。建议进入目标示例目录后安装该目录自己的 `requirements.txt`。

例如运行 LangChain Agents 示例：

```bash
cd langchain-example/langchain_agents_examples
pip install -r requirements.txt
python 01_basic_agent.py
```

运行 Multi-agent 示例：

```bash
cd langchain-example/langchain_multi_agent_examples
pip install -r requirements.txt
python 01_subagents_basic.py
```

运行 PyTorch 示例项目：

```bash
cd pytorch-docs/pytorch_project
pip install -r requirements.txt
python main.py
```

## 常用入口

### LangChain 示例

```text
langchain-example/langchain_models_examples/             # 模型初始化、工具调用、结构化输出、流式调用
langchain-example/langchain_messages_examples/           # 消息类型、多模态内容、消息元数据和历史
langchain-example/langchain_tools_examples/              # 工具定义、校验、错误处理、异步工具
langchain-example/langchain_agents_examples/             # Agent 基础、中间件、记忆、人机协作
langchain-example/langchain_short_term_memory_examples/  # 短期记忆、线程、裁剪、摘要、自定义状态
langchain-example/langchain_streaming_examples/          # 流式输出
langchain-example/langchain_structured_output_examples/  # 结构化输出
langchain-example/langchain_mcp_examples/                # MCP 工具接入
langchain-example/langchain_hitl_examples/               # Human-in-the-loop 审批流
langchain-example/langchain_multi_agent_examples/        # 子智能体、handoff、router、skills
langchain-example/langchain_retrieval_memory_examples/   # 检索增强记忆与 RAG
```

### 魔搭社区智能答疑 Agent

进入项目目录：

```bash
cd DevMate/modelscope_qa_agent
pip install -r requirements.txt
```

检查基础设施：

```bash
python scripts/check_infrastructure.py
```

运行测试：

```bash
pytest tests/
```

更多说明见：

- `DevMate/modelscope_qa_agent/README.md`
- `DevMate/modelscope_qa_agent/QUICK_START.md`
- `DevMate/modelscope_qa_agent/WORKFLOW.md`
- `DevMate/modelscope_qa_agent/docs/`

### Hello Agents 代码

```bash
cd agent-langchain-code/HelloAgents_Chapter4_Code
pip install -r requirements.txt
python 01_react_agent.py
python 02_plan_and_solve.py
python 03_reflection_agent.py
```

相关章节还包括：

```text
agent-langchain-code/HelloAgents_Chapter6_Code/
agent-langchain-code/HelloAgents_Chapter7_Code/
```

### 中文文档

如果只是学习概念，优先阅读：

```text
langchain-docs/langchain_v1.md
langchain-docs/01_LangChain_Agents_详细总结.md
langchain-docs/13_LangChain_Multi_Agent_详细指南.md
langchain-docs/14_LangChain_Retrieval_Memory_详细指南.md
agent-docs/HelloAgents_项目分析.md
```

## 配置说明

根目录 `.env.example` 包含通用模型配置：

```env
OPENAI_API_KEY=your-openai-api-key-here
ZHIPUAI_API_KEY=your-zhipu-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

不同子项目可能还有独立 `.env.example`，请优先以子项目文档为准。例如：

- `langchain-example/langchain_first/.env.example`
- `agent-langchain-code/HelloAgents_Chapter4_Code/.env.example`
- `DevMate/modelscope_qa_agent/.env.example`

## 测试与检查

不同模块的测试命令不同，常见方式如下：

```bash
# 魔搭 QA Agent
cd DevMate/modelscope_qa_agent
pytest tests/

# Hello Agents Chapter 4
cd agent-langchain-code/HelloAgents_Chapter4_Code
python test_plan_solve.py

# LangChain Agents 示例环境检查
cd langchain-example/langchain_agents_examples
python test_setup.py
```

类型检查可参考根目录 `pyrightconfig.json`，在 IDE 或命令行中使用 Pyright。

## 开发约定

- 根目录是多项目集合，不建议在根目录直接安装全部依赖。
- 新增示例时，建议在对应主题目录下补充 `README.md` 和 `requirements.txt`。
- API Key、数据库密码、向量库连接信息等敏感配置只放入本地 `.env`，不要提交到版本库。
- 对于实验性示例，建议在文件开头注明依赖的 LangChain 版本和模型服务提供方。
- 历史示例保留在 `langchain-demo-old/`，新增内容优先放入 `langchain-example/` 或具体业务子项目目录。

## 许可证

本仓库主要用于学习、研究和示例演示。使用其中涉及的模型服务、第三方 API 和外部数据时，请遵循对应平台的服务条款与许可证要求。
