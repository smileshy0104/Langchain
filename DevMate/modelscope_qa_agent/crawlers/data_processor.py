"""
Data Processor

处理爬取的数据并准备导入向量数据库
"""

import json
import os
from typing import List, Dict
from pathlib import Path
from datetime import datetime


class DataProcessor:
    """数据处理器"""

    def __init__(self, crawled_data_dir: str = "data/crawled", processed_data_dir: str = "data/processed"):
        """
        初始化数据处理器

        Args:
            crawled_data_dir: 爬取数据目录
            processed_data_dir: 处理后数据目录
        """
        self.crawled_data_dir = Path(crawled_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def load_json_files(self, directory: Path) -> List[Dict]:
        """
        加载目录中的所有JSON文件

        Args:
            directory: 目录路径

        Returns:
            JSON数据列表
        """
        data = []

        if not directory.exists():
            print(f"⚠️  目录不存在: {directory}")
            return data

        for json_file in directory.glob("**/*.json"):
            # 跳过汇总文件和检查点文件
            if json_file.name in ['summary.json', 'checkpoint.json']:
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    # 如果是包含items的目录文件,展开items
                    if isinstance(content, dict) and 'items' in content:
                        data.extend(content['items'])
                    elif isinstance(content, dict):
                        data.append(content)
                    elif isinstance(content, list):
                        data.extend(content)
            except Exception as e:
                print(f"❌ 加载文件失败: {json_file}")
                print(f"   错误: {e}")

        return data

    def clean_text(self, text: str) -> str:
        """
        清理文本

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        if not text:
            return ""

        # 移除多余空白
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def chunk_text(self, text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
        """
        分块文本

        Args:
            text: 文本
            chunk_size: 块大小
            chunk_overlap: 块重叠

        Returns:
            文本块列表
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # 尝试在句号、换行符等处分割
            if end < len(text):
                for sep in ['\n\n', '\n', '。', '. ', '! ', '? ']:
                    last_sep = chunk.rfind(sep)
                    if last_sep > chunk_size * 0.5:  # 至少保留一半内容
                        chunk = text[start:start + last_sep + len(sep)]
                        break

            chunks.append(chunk.strip())
            start += len(chunk) - chunk_overlap

        return chunks

    def process_document(self, doc: Dict, source_type: str) -> List[Dict]:
        """
        处理单个文档

        Args:
            doc: 文档数据
            source_type: 源类型

        Returns:
            处理后的文档块列表
        """
        processed_chunks = []

        # 提取内容
        content = doc.get('content', '') or doc.get('description', '') or doc.get('readme', '')

        if not content or len(content) < 50:
            return []

        # 清理文本
        content = self.clean_text(content)

        # 分块
        chunks = self.chunk_text(content)

        # 生成元数据
        base_metadata = {
            'source_type': source_type,
            'title': doc.get('title', 'Untitled'),
            'url': doc.get('url', ''),
            'original_source': doc.get('source', source_type),
        }

        # 添加额外元数据
        if 'author' in doc:
            base_metadata['author'] = doc['author']
        if 'date' in doc:
            base_metadata['date'] = doc['date']
        if 'tags' in doc and doc['tags']:
            base_metadata['tags'] = ','.join(doc['tags'])
        if 'language' in doc:
            base_metadata['language'] = doc['language']
        if 'stars' in doc:
            base_metadata['stars'] = doc['stars']

        # 为每个块创建文档
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                'content': chunk,
                'metadata': {
                    **base_metadata,
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                }
            }
            processed_chunks.append(chunk_doc)

        return processed_chunks

    def process_all(self) -> Dict[str, List[Dict]]:
        """
        处理所有爬取的数据

        Returns:
            按源类型分组的处理后数据
        """
        print("=" * 70)
        print("开始处理爬取数据")
        print("=" * 70)

        results = {}

        # 处理各类数据
        source_dirs = {
            'docs': self.crawled_data_dir / 'docs',
            'learn': self.crawled_data_dir / 'learn',
            'github': self.crawled_data_dir / 'github',
            'catalog': self.crawled_data_dir / 'catalog',
        }

        total_processed = 0

        for source_type, source_dir in source_dirs.items():
            print(f"\n📂 处理 {source_type} 数据...")

            # 加载原始数据
            raw_data = self.load_json_files(source_dir)
            print(f"   加载了 {len(raw_data)} 个原始文档")

            # 处理每个文档
            processed_docs = []
            for doc in raw_data:
                chunks = self.process_document(doc, source_type)
                processed_docs.extend(chunks)

            print(f"   生成了 {len(processed_docs)} 个文档块")
            total_processed += len(processed_docs)

            results[source_type] = processed_docs

            # 保存处理后的数据
            if processed_docs:
                output_file = self.processed_data_dir / f"{source_type}_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_docs, f, ensure_ascii=False, indent=2)
                print(f"✅ 已保存到: {output_file}")

        # 保存汇总
        summary = {
            'total_chunks': total_processed,
            'by_source': {k: len(v) for k, v in results.items()},
            'processed_at': datetime.now().isoformat(),
        }

        summary_file = self.processed_data_dir / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print(f"✅ 处理完成! 共 {total_processed} 个文档块")
        for source, count in summary['by_source'].items():
            print(f"   {source}: {count} 块")
        print("=" * 70)

        return results

    def export_for_ingestion(self, output_file: str = "data/processed/all_documents.jsonl"):
        """
        导出为JSONL格式,用于导入向量数据库

        Args:
            output_file: 输出文件路径
        """
        print(f"\n📤 导出数据到 {output_file}...")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for json_file in self.processed_data_dir.glob("*_processed.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as jf:
                        docs = json.load(jf)
                        for doc in docs:
                            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                            count += 1
                except Exception as e:
                    print(f"❌ 导出失败: {json_file}")
                    print(f"   错误: {e}")

        print(f"✅ 已导出 {count} 个文档到 {output_path}")
        return count
