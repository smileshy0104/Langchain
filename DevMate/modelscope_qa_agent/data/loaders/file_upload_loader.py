"""
文件上传加载器

支持多种文档格式的本地文件加载:
- PDF (.pdf)
- Word (.docx, .doc)
- 纯文本 (.txt)
- Markdown (.md)
- JSON (.json)
- XML (.xml)
- HTML (.html)
- RTF (.rtf)
- Excel (.xlsx, .xls)
- PowerPoint (.pptx, .ppt)
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from langchain_core.documents import Document

# PDF 加载器
try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Word 加载器
try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# 文本加载器
try:
    from langchain_community.document_loaders import TextLoader
    TEXT_AVAILABLE = True
except ImportError:
    TEXT_AVAILABLE = False

# Unstructured 加载器 (Excel, PowerPoint, RTF等)
try:
    from langchain_community.document_loaders import (
        UnstructuredExcelLoader,
        UnstructuredPowerPointLoader,
        UnstructuredHTMLLoader,
        UnstructuredMarkdownLoader,
        UnstructuredXMLLoader,
        UnstructuredRTFLoader
    )
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False


class FileUploadLoader:
    """文件上传加载器

    支持多种文档格式的本地文件加载,自动根据文件扩展名选择合适的加载器。

    Supported Formats:
        - PDF: .pdf
        - Word: .docx, .doc
        - Text: .txt
        - Markdown: .md
        - JSON: .json
        - XML: .xml
        - HTML: .html
        - RTF: .rtf
        - Excel: .xlsx, .xls
        - PowerPoint: .pptx, .ppt

    Attributes:
        file_path: 文件路径
        file_ext: 文件扩展名
        metadata: 自定义元数据

    Example:
        >>> loader = FileUploadLoader("document.pdf")
        >>> documents = loader.load()
        >>> print(f"加载了 {len(documents)} 个文档块")
    """

    # 支持的格式映射
    SUPPORTED_FORMATS = {
        ".pdf": "PDF",
        ".docx": "Word",
        ".doc": "Word",
        ".txt": "Text",
        ".md": "Markdown",
        ".json": "JSON",
        ".xml": "XML",
        ".html": "HTML",
        ".rtf": "RTF",
        ".xlsx": "Excel",
        ".xls": "Excel",
        ".pptx": "PowerPoint",
        ".ppt": "PowerPoint"
    }

    def __init__(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ):
        """初始化文件上传加载器

        Args:
            file_path: 文件路径
            metadata: 自定义元数据
            verbose: 是否输出详细日志

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        self.file_path = Path(file_path)
        self.verbose = verbose
        self.custom_metadata = metadata or {}

        # 验证文件存在
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        # 获取文件扩展名
        self.file_ext = self.file_path.suffix.lower()

        # 验证文件格式
        if self.file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的文件格式: {self.file_ext}\n"
                f"支持的格式: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            )

        if verbose:
            print(f"✅ 文件上传加载器初始化成功")
            print(f"   - 文件: {self.file_path.name}")
            print(f"   - 类型: {self.SUPPORTED_FORMATS[self.file_ext]}")
            print(f"   - 大小: {self.file_path.stat().st_size} 字节")

    def _extract_base_metadata(self) -> Dict[str, Any]:
        """提取基础文件元数据

        Returns:
            Dict[str, Any]: 元数据字典
        """
        stat = self.file_path.stat()

        metadata = {
            "source": str(self.file_path.absolute()),
            "source_type": "file_upload",
            "filename": self.file_path.name,
            "file_ext": self.file_ext,
            "file_type": self.SUPPORTED_FORMATS[self.file_ext],
            "file_size": stat.st_size,
            "created_at": int(stat.st_ctime),
            "modified_at": int(stat.st_mtime)
        }

        # 合并自定义元数据
        metadata.update(self.custom_metadata)

        return metadata

    def _load_pdf(self) -> List[Document]:
        """加载 PDF 文件

        Returns:
            List[Document]: 文档列表
        """
        if not PDF_AVAILABLE:
            raise ImportError(
                "PDF 支持需要安装 pypdf:\n"
                "  pip install pypdf"
            )

        loader = PyPDFLoader(str(self.file_path))
        documents = loader.load()

        if self.verbose:
            print(f"✅ PDF 加载成功: {len(documents)} 页")

        return documents

    def _load_word(self) -> List[Document]:
        """加载 Word 文件

        Returns:
            List[Document]: 文档列表
        """
        if not DOCX_AVAILABLE:
            raise ImportError(
                "Word 支持需要安装 docx2txt:\n"
                "  pip install docx2txt"
            )

        loader = Docx2txtLoader(str(self.file_path))
        documents = loader.load()

        if self.verbose:
            print(f"✅ Word 加载成功: {len(documents)} 个文档")

        return documents

    def _load_text(self) -> List[Document]:
        """加载纯文本文件

        Returns:
            List[Document]: 文档列表
        """
        if not TEXT_AVAILABLE:
            raise ImportError(
                "文本支持需要安装 langchain-community"
            )

        loader = TextLoader(str(self.file_path), encoding="utf-8")
        documents = loader.load()

        if self.verbose:
            print(f"✅ 文本加载成功: {len(documents)} 个文档")

        return documents

    def _load_markdown(self) -> List[Document]:
        """加载 Markdown 文件

        Returns:
            List[Document]: 文档列表
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "Markdown 支持需要安装 unstructured:\n"
                "  pip install unstructured"
            )

        loader = UnstructuredMarkdownLoader(str(self.file_path))
        documents = loader.load()

        if self.verbose:
            print(f"✅ Markdown 加载成功: {len(documents)} 个文档")

        return documents

    def _load_html(self) -> List[Document]:
        """加载 HTML 文件

        Returns:
            List[Document]: 文档列表
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "HTML 支持需要安装 unstructured:\n"
                "  pip install unstructured"
            )

        loader = UnstructuredHTMLLoader(str(self.file_path))
        documents = loader.load()

        if self.verbose:
            print(f"✅ HTML 加载成功: {len(documents)} 个文档")

        return documents

    def _load_xml(self) -> List[Document]:
        """加载 XML 文件

        Returns:
            List[Document]: 文档列表
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "XML 支持需要安装 unstructured:\n"
                "  pip install unstructured"
            )

        loader = UnstructuredXMLLoader(str(self.file_path))
        documents = loader.load()

        if self.verbose:
            print(f"✅ XML 加载成功: {len(documents)} 个文档")

        return documents

    def _load_json(self) -> List[Document]:
        """加载 JSON 文件

        Returns:
            List[Document]: 文档列表
        """
        import json

        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 将 JSON 转换为字符串
        content = json.dumps(data, indent=2, ensure_ascii=False)

        doc = Document(
            page_content=content,
            metadata=self._extract_base_metadata()
        )

        if self.verbose:
            print(f"✅ JSON 加载成功: 1 个文档")

        return [doc]

    def _load_rtf(self) -> List[Document]:
        """加载 RTF 文件

        Returns:
            List[Document]: 文档列表
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "RTF 支持需要安装 unstructured:\n"
                "  pip install unstructured"
            )

        loader = UnstructuredRTFLoader(str(self.file_path))
        documents = loader.load()

        if self.verbose:
            print(f"✅ RTF 加载成功: {len(documents)} 个文档")

        return documents

    def _load_excel(self) -> List[Document]:
        """加载 Excel 文件

        Returns:
            List[Document]: 文档列表
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "Excel 支持需要安装 unstructured 和 openpyxl:\n"
                "  pip install unstructured openpyxl"
            )

        loader = UnstructuredExcelLoader(str(self.file_path))
        documents = loader.load()

        if self.verbose:
            print(f"✅ Excel 加载成功: {len(documents)} 个文档")

        return documents

    def _load_powerpoint(self) -> List[Document]:
        """加载 PowerPoint 文件

        Returns:
            List[Document]: 文档列表
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError(
                "PowerPoint 支持需要安装 unstructured 和 python-pptx:\n"
                "  pip install unstructured python-pptx"
            )

        loader = UnstructuredPowerPointLoader(str(self.file_path))
        documents = loader.load()

        if self.verbose:
            print(f"✅ PowerPoint 加载成功: {len(documents)} 个文档")

        return documents

    def load(self) -> List[Document]:
        """加载文件

        根据文件扩展名自动选择合适的加载器。

        Returns:
            List[Document]: 文档列表

        Example:
            >>> loader = FileUploadLoader("document.pdf")
            >>> documents = loader.load()
            >>> print(f"加载了 {len(documents)} 个文档块")
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"开始加载文件: {self.file_path.name}")
            print(f"{'='*70}\n")

        # 根据文件扩展名选择加载器
        if self.file_ext == ".pdf":
            documents = self._load_pdf()
        elif self.file_ext in [".docx", ".doc"]:
            documents = self._load_word()
        elif self.file_ext == ".txt":
            documents = self._load_text()
        elif self.file_ext == ".md":
            documents = self._load_markdown()
        elif self.file_ext == ".json":
            documents = self._load_json()
        elif self.file_ext == ".xml":
            documents = self._load_xml()
        elif self.file_ext == ".html":
            documents = self._load_html()
        elif self.file_ext == ".rtf":
            documents = self._load_rtf()
        elif self.file_ext in [".xlsx", ".xls"]:
            documents = self._load_excel()
        elif self.file_ext in [".pptx", ".ppt"]:
            documents = self._load_powerpoint()
        else:
            raise ValueError(f"不支持的文件格式: {self.file_ext}")

        # 添加基础元数据
        base_metadata = self._extract_base_metadata()
        for doc in documents:
            # 合并元数据 (保留加载器提供的元数据)
            doc.metadata.update({
                k: v for k, v in base_metadata.items()
                if k not in doc.metadata
            })

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ 文件加载完成")
            print(f"{'='*70}")
            print(f"总计加载: {len(documents)} 个文档块")
            total_chars = sum(len(doc.page_content) for doc in documents)
            print(f"总字符数: {total_chars}")
            print(f"{'='*70}\n")

        return documents

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """获取支持的文件格式列表

        Returns:
            List[str]: 支持的文件扩展名列表
        """
        return list(cls.SUPPORTED_FORMATS.keys())

    @classmethod
    def is_supported(cls, file_path: Union[str, Path]) -> bool:
        """检查文件格式是否支持

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否支持
        """
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_FORMATS


# 便捷函数

def load_uploaded_file(
    file_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> List[Document]:
    """加载上传文件的便捷函数

    Args:
        file_path: 文件路径
        metadata: 自定义元数据
        verbose: 详细输出

    Returns:
        List[Document]: 文档列表

    Example:
        >>> documents = load_uploaded_file("document.pdf", verbose=True)
        >>> print(f"加载了 {len(documents)} 个文档块")
    """
    loader = FileUploadLoader(file_path, metadata=metadata, verbose=verbose)
    return loader.load()


# 示例用法
if __name__ == "__main__":
    print("=" * 70)
    print("文件上传加载器测试")
    print("=" * 70)

    # 显示支持的格式
    print("\n支持的文件格式:")
    print("-" * 70)
    for ext, file_type in FileUploadLoader.SUPPORTED_FORMATS.items():
        print(f"  {ext:10s} - {file_type}")

    # 测试文件格式检查
    print("\n文件格式检查测试:")
    print("-" * 70)

    test_files = [
        "document.pdf",
        "report.docx",
        "data.xlsx",
        "presentation.pptx",
        "notes.txt",
        "readme.md",
        "config.json",
        "data.xml",
        "page.html",
        "document.exe"  # 不支持
    ]

    for filename in test_files:
        is_supported = FileUploadLoader.is_supported(filename)
        status = "✅" if is_supported else "❌"
        print(f"{status} {filename}")

    print("\n" + "=" * 70)
    print("提示: 要测试实际文件加载,请提供有效的文件路径")
    print("=" * 70)
