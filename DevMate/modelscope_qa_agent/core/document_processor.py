"""
文档处理器

负责文档的加载、清洗、分块和质量评分。
支持 Markdown 语义分块、代码块完整性保护、多源文档加载。
"""

from typing import List, Optional, Dict
import re
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from markdownify import markdownify as md


class DocumentProcessor:
    """文档处理器

    管理文档加载、清洗、分块和质量评分的完整流程。

    Features:
        - 加载多源文档(Web、GitHub、本地文件)
        - 智能清洗HTML和格式化问题
        - 语义分块(基于 Markdown 标题)
        - 代码块完整性保护
        - 文档质量评分

    Attributes:
        markdown_splitter: Markdown 标题分块器
        text_splitter: 递归字符分块器
        chunk_size: 分块大小(默认1000字符)
        chunk_overlap: 分块重叠(默认200字符)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """初始化文档处理器

        Args:
            chunk_size: 文本分块大小(字符数)
            chunk_overlap: 分块之间的重叠字符数

        Example:
            >>> processor = DocumentProcessor(chunk_size=800, chunk_overlap=150)
            >>> docs = processor.load_modelscope_docs()
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Markdown 标题分块器
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False  # 保留标题在内容中
        )

        # 递归字符分块器(用于进一步拆分大块)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "!", "?", ";", ",", " ", ""],
            length_function=len
        )

        print(f"✅ DocumentProcessor 初始化成功")
        print(f"   - chunk_size: {chunk_size}")
        print(f"   - chunk_overlap: {chunk_overlap}")

    def load_modelscope_docs(
        self,
        urls: Optional[List[str]] = None
    ) -> List[Document]:
        """加载魔搭社区官方文档

        Args:
            urls: 要加载的URL列表。如果为 None,使用默认URL列表。

        Returns:
            List[Document]: 加载的文档列表

        Raises:
            Exception: 加载失败时抛出异常

        Example:
            >>> processor = DocumentProcessor()
            >>> docs = processor.load_modelscope_docs()
            >>> print(f"加载了 {len(docs)} 个文档")
        """
        if urls is None:
            # 默认魔搭社区文档URL列表
            urls = [
                "https://www.modelscope.cn/docs/overview",
                "https://www.modelscope.cn/docs/models",
                "https://www.modelscope.cn/docs/datasets",
                "https://www.modelscope.cn/docs/pipelines",
            ]

        documents = []
        for url in urls:
            try:
                print(f"📥 加载文档: {url}")
                loader = WebBaseLoader(
                    web_paths=[url],
                    bs_kwargs={
                        "parse_only": BeautifulSoup.SoupStrainer(
                            ["article", "main", "div"]
                        )
                    }
                )
                docs = loader.load()

                # 添加元数据
                for doc in docs:
                    doc.metadata["source_url"] = url
                    doc.metadata["source_type"] = "official_docs"
                    doc.metadata["document_type"] = "tutorial"  # 可根据URL调整

                documents.extend(docs)
                print(f"   ✅ 成功加载 {len(docs)} 个文档段落")

            except Exception as e:
                print(f"   ⚠️  加载失败: {e}")
                continue

        print(f"\n✅ 总共加载 {len(documents)} 个文档")
        return documents

    def clean_document(self, doc: Document) -> Document:
        """清洗文档内容

        清理HTML标签、规范化空白、统一代码块格式。

        Args:
            doc: 原始文档

        Returns:
            Document: 清洗后的文档

        Example:
            >>> doc = Document(page_content="<p>Hello</p>\\n\\n\\nWorld")
            >>> clean_doc = processor.clean_document(doc)
            >>> print(clean_doc.page_content)
            Hello

            World
        """
        content = doc.page_content

        # 1. 移除 HTML 标签(如果有残留)
        content = re.sub(r'<[^>]+>', '', content)

        # 2. 移除多余空白行(保留双换行表示段落分隔)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

        # 3. 移除行首行尾空白
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(lines)

        # 4. 统一代码块格式(确保前后有空行)
        content = re.sub(
            r'```(\w+)?\n(.*?)```',
            r'\n```\1\n\2\n```\n',
            content,
            flags=re.DOTALL
        )

        # 5. 移除特殊字符(保留中英文、数字、常用标点)
        # 注意:保留代码块中的特殊字符
        code_blocks = []
        def save_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        # 暂存代码块
        content = re.sub(r'```[\s\S]*?```', save_code_block, content)

        # 清理非代码内容中的特殊字符
        # content = re.sub(r'[^\w\s\u4e00-\u9fff。,!?;:、""''()（）【】《》\-\n#*`]', '', content)

        # 恢复代码块
        for i, code_block in enumerate(code_blocks):
            content = content.replace(f"__CODE_BLOCK_{i}__", code_block)

        # 6. 最终清理:移除首尾空白
        content = content.strip()

        # 更新文档内容
        doc.page_content = content
        return doc

    def split_with_code_protection(self, doc: Document) -> List[Document]:
        """语义分块(保护代码块完整性)

        基于 Markdown 标题进行语义分块,确保代码块不被拆分。

        分块策略:
            1. 首先按 Markdown 标题分块
            2. 对于包含代码块的chunk,保持完整性
            3. 对于纯文本chunk,可以进一步拆分

        Args:
            doc: 待分块的文档

        Returns:
            List[Document]: 分块后的文档列表

        Example:
            >>> doc = Document(page_content="# Title\\n\\nText\\n\\n```python\\ncode\\n```")
            >>> chunks = processor.split_with_code_protection(doc)
            >>> print(len(chunks))
        """
        content = doc.page_content

        # 检查是否是 Markdown 格式(包含标题)
        has_markdown_headers = bool(re.search(r'^#+\s', content, re.MULTILINE))

        if has_markdown_headers:
            # 1. 先按 Markdown 标题分块
            try:
                header_chunks = self.markdown_splitter.split_text(content)
            except Exception as e:
                print(f"⚠️  Markdown 分块失败,使用默认分块: {e}")
                header_chunks = [Document(page_content=content, metadata=doc.metadata.copy())]
        else:
            # 没有 Markdown 标题,作为单个文档处理
            header_chunks = [Document(page_content=content, metadata=doc.metadata.copy())]

        # 2. 对每个标题块进一步处理
        final_chunks = []

        for chunk in header_chunks:
            # 继承原文档的元数据
            if not chunk.metadata:
                chunk.metadata = doc.metadata.copy()
            else:
                # 合并元数据(chunk 的元数据可能包含标题信息)
                chunk.metadata = {**doc.metadata, **chunk.metadata}

            chunk_content = chunk.page_content

            # 检测代码块
            code_blocks = re.findall(r'```[\s\S]*?```', chunk_content)

            if code_blocks and len(chunk_content) > self.chunk_size:
                # 有代码块且内容过长
                # 策略:尝试在代码块边界处拆分

                # 计算代码块总长度
                code_length = sum(len(cb) for cb in code_blocks)
                text_length = len(chunk_content) - code_length

                if code_length > self.chunk_size * 0.8:
                    # 代码块占比过大,保持完整(即使超长)
                    final_chunks.append(chunk)
                else:
                    # 尝试按段落拆分(避免拆散代码块)
                    sub_chunks = self._split_around_code_blocks(chunk)
                    final_chunks.extend(sub_chunks)

            elif len(chunk_content) <= self.chunk_size:
                # 内容合适,直接使用
                final_chunks.append(chunk)

            else:
                # 无代码块或内容较短,可以进一步拆分
                sub_chunks = self.text_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)

        # 3. 为每个chunk添加边界类型元数据
        for i, chunk in enumerate(final_chunks):
            # 判断chunk_boundary类型
            if "Header 1" in chunk.metadata or "Header 2" in chunk.metadata:
                chunk.metadata["chunk_boundary"] = "section"
            elif "Header 3" in chunk.metadata:
                chunk.metadata["chunk_boundary"] = "subsection"
            elif "```" in chunk.page_content:
                chunk.metadata["chunk_boundary"] = "code_block"
            else:
                chunk.metadata["chunk_boundary"] = "paragraph"

        return final_chunks

    def _split_around_code_blocks(self, doc: Document) -> List[Document]:
        """围绕代码块拆分文档

        将文档在代码块边界处拆分,保持代码块完整性。

        Args:
            doc: 待拆分的文档

        Returns:
            List[Document]: 拆分后的文档列表
        """
        content = doc.page_content
        chunks = []

        # 找到所有代码块的位置
        pattern = r'(```[\s\S]*?```)'
        parts = re.split(pattern, content)

        current_chunk = ""
        for part in parts:
            if part.startswith("```"):
                # 这是代码块
                if current_chunk:
                    # 先保存之前的文本
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata=doc.metadata.copy()
                    ))
                    current_chunk = ""

                # 代码块单独作为一个chunk
                chunks.append(Document(
                    page_content=part.strip(),
                    metadata=doc.metadata.copy()
                ))
            else:
                # 普通文本
                if len(current_chunk) + len(part) <= self.chunk_size:
                    current_chunk += part
                else:
                    if current_chunk:
                        chunks.append(Document(
                            page_content=current_chunk.strip(),
                            metadata=doc.metadata.copy()
                        ))
                    current_chunk = part

        # 保存剩余内容
        if current_chunk.strip():
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata=doc.metadata.copy()
            ))

        return chunks if chunks else [doc]

    def calculate_quality_score(self, doc: Document) -> float:
        """计算文档质量评分(0-1)

        评分维度:
            - 长度合理性(100-2000字符): 0.25分
            - 结构完整性(有标题): 0.25分
            - 代码示例(技术文档): 0.25分
            - 来源可信度: 0.25分

        Args:
            doc: 待评分的文档

        Returns:
            float: 质量评分(0.0-1.0)

        Example:
            >>> doc = Document(
            ...     page_content="# Title\\n\\nContent with ```code```",
            ...     metadata={"source_type": "official_docs"}
            ... )
            >>> score = processor.calculate_quality_score(doc)
            >>> print(f"质量评分: {score}")
        """
        score = 0.0
        content = doc.page_content
        metadata = doc.metadata

        # 1. 长度合理性(100-2000字符)
        length = len(content)
        if 100 < length < 2000:
            score += 0.25
        elif 50 < length <= 100:
            score += 0.15  # 稍短,减少分数
        elif 2000 <= length < 3000:
            score += 0.20  # 稍长,减少分数

        # 2. 结构完整性(有 Markdown 标题)
        if re.search(r'^#+\s', content, re.MULTILINE):
            score += 0.25

        # 3. 代码示例(技术文档必备)
        if '```' in content:
            score += 0.25
        elif '`' in content:
            # 有行内代码,给部分分数
            score += 0.10

        # 4. 来源可信度
        source_type = metadata.get("source_type", "unknown")
        if source_type == "official_docs":
            score += 0.25
        elif source_type == "github_docs":
            score += 0.20
        elif source_type == "qa_dataset":
            score += 0.15
        else:
            score += 0.05  # 其他来源给少量分数

        # 确保分数在 0-1 范围内
        return min(1.0, max(0.0, score))

    def process_document(
        self,
        doc: Document,
        clean: bool = True,
        split: bool = True,
        calculate_score: bool = True
    ) -> List[Document]:
        """处理单个文档(清洗、分块、评分)

        完整的文档处理流程:
            1. 清洗文档内容
            2. 语义分块(可选)
            3. 计算质量评分(可选)

        Args:
            doc: 原始文档
            clean: 是否清洗文档
            split: 是否分块
            calculate_score: 是否计算质量评分

        Returns:
            List[Document]: 处理后的文档列表

        Example:
            >>> processor = DocumentProcessor()
            >>> raw_doc = Document(page_content="<p>Hello</p>")
            >>> processed_docs = processor.process_document(raw_doc)
            >>> print(f"处理后: {len(processed_docs)} 个文档")
        """
        # 1. 清洗
        if clean:
            doc = self.clean_document(doc)

        # 2. 分块
        if split:
            chunks = self.split_with_code_protection(doc)
        else:
            chunks = [doc]

        # 3. 计算质量评分
        if calculate_score:
            for chunk in chunks:
                quality_score = self.calculate_quality_score(chunk)
                chunk.metadata["quality_score"] = quality_score

        return chunks

    def process_documents(
        self,
        docs: List[Document],
        clean: bool = True,
        split: bool = True,
        calculate_score: bool = True
    ) -> List[Document]:
        """批量处理文档

        Args:
            docs: 原始文档列表
            clean: 是否清洗文档
            split: 是否分块
            calculate_score: 是否计算质量评分

        Returns:
            List[Document]: 处理后的文档列表

        Example:
            >>> processor = DocumentProcessor()
            >>> raw_docs = [Document(page_content="Doc 1"), Document(page_content="Doc 2")]
            >>> processed_docs = processor.process_documents(raw_docs)
        """
        all_chunks = []

        for i, doc in enumerate(docs):
            try:
                chunks = self.process_document(
                    doc,
                    clean=clean,
                    split=split,
                    calculate_score=calculate_score
                )
                all_chunks.extend(chunks)

                if (i + 1) % 10 == 0:
                    print(f"   处理进度: {i + 1}/{len(docs)}")

            except Exception as e:
                print(f"   ⚠️  处理文档 {i} 失败: {e}")
                continue

        print(f"\n✅ 批量处理完成: {len(docs)} 个文档 → {len(all_chunks)} 个chunks")
        return all_chunks
