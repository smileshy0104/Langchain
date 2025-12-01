# 文档上传功能实现文档

## 概述

本功能实现了完整的文档上传、处理和向量化存储流程,支持多种文档格式和存储后端,使用火山引擎(VolcEngine)豆包模型进行向量化和问答。

## 功能特性

### 1. 支持的文档格式

- **PDF** (.pdf) - 使用 PyPDFLoader
- **Word** (.docx, .doc) - 使用 Docx2txtLoader
- **Excel** (.xlsx, .xls) - 使用 UnstructuredExcelLoader
- **PowerPoint** (.pptx, .ppt) - 使用 UnstructuredPowerPointLoader
- **文本** (.txt) - 使用 TextLoader
- **Markdown** (.md) - 使用 UnstructuredMarkdownLoader
- **JSON** (.json) - 自定义加载器
- **XML** (.xml) - 使用 UnstructuredXMLLoader
- **HTML** (.html) - 使用 UnstructuredHTMLLoader
- **RTF** (.rtf) - 使用 UnstructuredRTFLoader

### 2. 存储后端

- **MinIO**: 对象存储 (已实现)
- **Local**: 本地文件系统 (已实现)
- **OSS**: 阿里云对象存储 (待实现)
- **S3**: AWS S3 对象存储 (待实现)

### 3. AI 服务提供商

- **VolcEngine**: 火山引擎豆包模型 (已实现)
  - Embedding: doubao-embedding-text-240715 (1024维)
  - Chat: doubao-seed-1-6-250615
- **DashScope**: 通义千问 (已支持)
  - Embedding: text-embedding-v2 (1536维)
- **OpenAI**: OpenAI 模型 (已支持)
  - Embedding: text-embedding-3-small

### 4. 向量数据库

- **Milvus**: 向量检索引擎
  - 自动创建 Collection
  - 配置向量索引 (IVF_FLAT)
  - 支持元数据过滤

## 配置说明

### 配置文件: `config.yaml`

```yaml
# AI服务配置
ai:
  provider: "volcengine"  # volcengine, openai, dashscope
  api_key: "your-api-key"
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  models:
    embedding: "doubao-embedding-text-240715"
    chat: "doubao-seed-1-6-250615"
  parameters:
    temperature: 0.3
    top_p: 0.8
    max_tokens: 4000

# 存储配置
storage:
  type: "minio"  # local, minio, oss, s3
  local:
    upload_path: "./uploads"
  minio:
    endpoint: "localhost:9000"
    access_key: "minioadmin"
    secret_key: "minioadmin"
    bucket_name: "ai-documents"
    use_ssl: false
  max_file_size: "100MB"
  allowed_extensions: [".pdf", ".docx", ".txt", ".md", ".json", ...]

# Milvus 向量数据库配置
milvus:
  host: "localhost"
  port: 19530
  collection_name: "modelscope_docs"
  vector_dim: 1024  # doubao-embedding 维度

# 检索配置
retrieval:
  top_k: 3
  min_confidence_score: 0.7
  vector_weight: 0.6
  bm25_weight: 0.4

# Agent 配置
agent:
  max_conversation_turns: 10
  context_window_size: 4000
  progress_threshold: 5
```

## 使用方式

### 方式 1: 使用 DocumentUploadService

```python
from services.document_upload_service import DocumentUploadService

# 初始化服务
service = DocumentUploadService()

# 处理本地文件
result = service.process_local_file(
    local_file_path="document.pdf",
    metadata={"category": "technical", "source": "upload"},
    clean=True,
    split=True,
    calculate_score=True,
    store_to_vector_db=True  # 存储到 Milvus
)

print(f"处理完成!")
print(f"文档块数: {result['document_count']}")
print(f"文档IDs: {result['document_ids']}")
```

### 方式 2: 使用 DocumentProcessor

```python
from core.document_processor import DocumentProcessor

# 初始化处理器
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

# 加载并处理文件
processed_docs = processor.load_and_process_file(
    file_path="document.pdf",
    clean=True,
    split=True,
    calculate_score=True
)

print(f"处理后文档块数: {len(processed_docs)}")
```

### 方式 3: 使用 FileUploadLoader

```python
from data.loaders.file_upload_loader import FileUploadLoader

# 加载文件
loader = FileUploadLoader("document.pdf", verbose=True)
documents = loader.load()

print(f"加载了 {len(documents)} 个文档块")
```

## 处理流程

```
1. 文件上传
   ↓
2. 文件验证 (大小、格式)
   ↓
3. 文档加载 (根据格式选择加载器)
   ↓
4. 文档清洗 (HTML清理、格式化)
   ↓
5. 智能分块 (Markdown标题分块 + 递归字符分块)
   ↓
6. 代码块保护 (防止代码被切断)
   ↓
7. 质量评分 (0-1评分)
   ↓
8. 向量化 (使用配置的Embedding模型)
   ↓
9. 存储到 Milvus (向量+元数据)
   ↓
10. 支持问答检索
```

## 核心模块

### 1. 配置管理 (`config/config_loader.py`)
- YAML 配置加载
- 类型安全的配置访问
- 多provider支持

### 2. 存储管理 (`storage/storage_manager.py`)
- 统一存储接口
- 文件验证
- MinIO/Local存储支持

### 3. 文档加载 (`data/loaders/file_upload_loader.py`)
- 多格式支持
- 自动格式检测
- 元数据提取

### 4. 文档处理 (`core/document_processor.py`)
- 文档清洗
- 语义分块
- 质量评分

### 5. Embeddings (`core/embeddings.py`)
- VolcEngine Embeddings
- DashScope Embeddings
- OpenAI Embeddings

### 6. 向量存储 (`core/vector_store.py`)
- Milvus集成
- 索引管理
- LangChain兼容

### 7. 上传服务 (`services/document_upload_service.py`)
- 一站式上传处理
- 批量处理
- 错误处理

## 测试验证

### 运行测试

```bash
# 1. 测试配置加载
python config/config_loader.py

# 2. 测试存储管理器
python -m storage.storage_manager

# 3. 测试文件加载器
python -m data.loaders.file_upload_loader

# 4. 测试文档上传
python tests/test_document_upload.py

# 5. 测试 Embeddings
python -m core.embeddings

# 6. 测试完整服务
python services/document_upload_service.py
```

### 测试结果

所有测试均通过 ✅:

1. ✅ 配置加载器 - YAML解析正常
2. ✅ MinIO存储管理器 - 连接成功,创建bucket
3. ✅ 多格式文档加载器 - 10种格式支持
4. ✅ 文档处理流程 - 加载、清洗、分块、评分
5. ✅ VolcEngine Embeddings - 初始化成功
6. ✅ Milvus向量存储 - Collection创建、索引配置
7. ✅ 文档上传服务 - 完整流程测试

## 依赖包

需要安装以下包:

```bash
# 核心依赖
pip install langchain langchain-core langchain-community
pip install pydantic pydantic-settings
pip install pyyaml
pip install openai  # VolcEngine兼容

# 存储
pip install minio  # MinIO支持

# 向量数据库
pip install pymilvus langchain-milvus

# 文档加载器
pip install pypdf  # PDF支持
pip install docx2txt  # Word支持
pip install unstructured  # 多格式支持
pip install openpyxl  # Excel支持
pip install python-pptx  # PowerPoint支持

# 可选
pip install dashscope  # 通义千问
pip install langchain-openai  # OpenAI
```

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     用户/API层                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              DocumentUploadService                            │
│  (统一入口: 文件上传 + 处理 + 存储)                            │
└──┬──────────────┬──────────────┬───────────────────────────┘
   │              │              │
   ↓              ↓              ↓
┌──────────┐  ┌──────────┐  ┌──────────────┐
│ Storage  │  │Document  │  │VectorStore   │
│ Manager  │  │Processor │  │Manager       │
└──────────┘  └──────────┘  └──────────────┘
   │              │              │
   ↓              ↓              ↓
┌──────────┐  ┌──────────┐  ┌──────────────┐
│MinIO/    │  │File      │  │Embeddings    │
│Local     │  │Upload    │  │(VolcEngine)  │
│Storage   │  │Loader    │  └──────────────┘
└──────────┘  └──────────┘         │
                   │                ↓
                   ↓         ┌──────────────┐
            ┌──────────────┐ │Milvus Vector │
            │10+ Format    │ │Database      │
            │Loaders       │ └──────────────┘
            └──────────────┘
```

## 注意事项

1. **API密钥配置**: 请在 `config.yaml` 中配置有效的 API 密钥
2. **MinIO服务**: 如使用MinIO,请确保服务运行在 `localhost:9000`
3. **Milvus服务**: 请确保 Milvus 运行在 `localhost:19530`
4. **向量维度**: 不同模型的向量维度不同,请在配置中正确设置
   - doubao-embedding-text-240715: 1024维
   - text-embedding-v2: 1536维
   - text-embedding-3-small: 1536维
5. **文件大小限制**: 默认最大100MB,可在配置中调整
6. **并发处理**: 当前为同步处理,大批量文件可考虑异步处理

## 扩展性

系统设计支持以下扩展:

1. **新增文档格式**: 在 `FileUploadLoader` 中添加新的加载器
2. **新增存储后端**: 实现 `StorageBackend` 接口
3. **新增AI提供商**: 在 `embeddings.py` 中添加新的provider
4. **自定义分块策略**: 修改 `DocumentProcessor` 的分块逻辑
5. **自定义质量评分**: 调整质量评分算法

## 问题排查

### 问题 1: ModuleNotFoundError

**解决**: 确保在项目根目录运行,或使用 `python -m` 方式运行

### 问题 2: MinIO连接失败

**解决**:
1. 检查 MinIO 服务是否运行
2. 检查端点配置是否正确
3. 检查访问密钥是否正确

### 问题 3: Milvus连接失败

**解决**:
1. 检查 Milvus 服务是否运行
2. 检查端口配置 (默认19530)
3. 查看 Milvus 日志

### 问题 4: 向量维度不匹配

**解决**:
1. 确认配置中的 `vector_dim` 与模型匹配
2. 如已创建 Collection,需删除重建或修改schema

## 总结

本功能实现了完整的文档上传和处理流程,支持:

✅ 10+ 种文档格式
✅ 多种存储后端
✅ 多种AI服务提供商
✅ 智能文档处理
✅ 向量化存储
✅ 问答检索支持

所有功能均已测试验证,可正常编译运行。
