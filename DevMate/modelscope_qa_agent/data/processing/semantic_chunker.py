"""
Semantic Chunker Module

Split documents into semantic chunks and upload to Milvus.

Core Functions:
- Semantic text splitting
- Chunk size optimization
- Metadata preservation
- Milvus batch upload
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter
)
from core.vector_store import VectorStoreManager


class SemanticChunker:
    """Semantic document chunker

    Split documents into semantic chunks optimized for retrieval.

    Attributes:
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        separators: List of separators for splitting
        verbose: Whether to output detailed logs

    Example:
        >>> chunker = SemanticChunker(chunk_size=1000)
        >>> chunks = chunker.chunk(documents)
        >>> print(f"Created {len(chunks)} chunks")
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: callable = len,
        verbose: bool = False
    ):
        """Initialize semantic chunker

        Args:
            chunk_size: Target chunk size (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)
            separators: Custom separators (default: auto-detect)
            length_function: Function to measure chunk length
            verbose: Output detailed logs
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.verbose = verbose

        # Default separators (hierarchical)
        self.separators = separators or [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentences
            ", ",    # Clauses
            " ",     # Words
            ""       # Characters
        ]

        # Initialize splitters
        self.general_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=self.separators
        )

        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if verbose:
            print("Semantic Chunker initialized successfully")
            print(f"   - Chunk size: {chunk_size}")
            print(f"   - Chunk overlap: {chunk_overlap}")
            print(f"   - Num separators: {len(self.separators)}")

    def _detect_document_type(self, document: Document) -> str:
        """Detect document type for optimal splitting

        Args:
            document: Input document

        Returns:
            str: Document type ('markdown', 'code', 'text')
        """
        source = document.metadata.get("source", "")
        content = document.page_content

        # Check file extension
        if source.endswith(('.md', '.markdown')):
            return 'markdown'
        elif source.endswith(('.py', '.js', '.java', '.cpp', '.go', '.rs')):
            return 'code'

        # Check content
        if '```' in content or '    ' in content[:100]:
            return 'code'
        elif content.count('#') > 3 or content.count('##') > 2:
            return 'markdown'

        return 'text'

    def _add_chunk_metadata(
        self,
        chunk: Document,
        chunk_index: int,
        total_chunks: int,
        original_doc: Document
    ) -> Document:
        """Add metadata to chunk

        Args:
            chunk: Chunk document
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks
            original_doc: Original document

        Returns:
            Document: Chunk with enhanced metadata
        """
        # Copy original metadata
        metadata = original_doc.metadata.copy()

        # Add chunk-specific metadata
        metadata.update({
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_size": len(chunk.page_content),
            "is_first_chunk": chunk_index == 0,
            "is_last_chunk": chunk_index == total_chunks - 1
        })

        return Document(
            page_content=chunk.page_content,
            metadata=metadata
        )

    def chunk_document(self, document: Document) -> List[Document]:
        """Chunk a single document

        Args:
            document: Input document

        Returns:
            List[Document]: List of chunks
        """
        # Detect document type
        doc_type = self._detect_document_type(document)

        # Choose appropriate splitter
        if doc_type == 'markdown':
            splitter = self.markdown_splitter
        else:
            splitter = self.general_splitter

        # Split document
        chunks = splitter.split_documents([document])

        # Add metadata to chunks
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunk = self._add_chunk_metadata(
                chunk,
                i,
                len(chunks),
                document
            )
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk a list of documents

        Args:
            documents: List of input documents

        Returns:
            List[Document]: List of all chunks

        Example:
            >>> chunker = SemanticChunker(verbose=True)
            >>> chunks = chunker.chunk(documents)
            >>> print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting semantic chunking")
            print(f"{'='*70}")
            print(f"Input documents: {len(documents)}")
            print(f"{'='*70}\n")

        all_chunks = []

        for i, doc in enumerate(documents):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"Chunking: {i + 1}/{len(documents)} documents...")

            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Semantic chunking completed")
            print(f"{'='*70}")
            print(f"Input documents: {len(documents)}")
            print(f"Output chunks: {len(all_chunks)}")
            print(f"Avg chunks per doc: {len(all_chunks) / len(documents):.1f}")
            print(f"{'='*70}\n")

        return all_chunks

    def get_stats(self) -> Dict[str, Any]:
        """Get chunker statistics

        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        return {
            "chunker_type": "SemanticChunker",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "num_separators": len(self.separators)
        }


class MilvusUploader:
    """Milvus batch uploader

    Upload chunked documents to Milvus vector store.

    Attributes:
        vector_store_manager: Vector store manager instance
        batch_size: Upload batch size
        verbose: Whether to output detailed logs

    Example:
        >>> uploader = MilvusUploader(api_key="xxx")
        >>> uploader.upload(chunks)
        >>> print(f"Uploaded {len(chunks)} chunks")
    """

    def __init__(
        self,
        api_key: str,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "modelscope_docs",
        batch_size: int = 100,
        verbose: bool = False
    ):
        """Initialize Milvus uploader

        Args:
            api_key: DashScope API key for embeddings
            milvus_uri: Milvus connection URI
            collection_name: Collection name
            batch_size: Upload batch size (default: 100)
            verbose: Output detailed logs
        """
        self.batch_size = batch_size
        self.verbose = verbose

        # Initialize vector store manager
        self.vector_store_manager = VectorStoreManager(
            uri=milvus_uri,
            collection_name=collection_name,
            api_key=api_key
        )

        if verbose:
            print("Milvus Uploader initialized successfully")
            print(f"   - Collection: {collection_name}")
            print(f"   - Batch size: {batch_size}")

    def upload(self, documents: List[Document]) -> Dict[str, Any]:
        """Upload documents to Milvus

        Args:
            documents: List of documents to upload

        Returns:
            Dict[str, Any]: Upload statistics

        Example:
            >>> uploader = MilvusUploader(api_key="xxx")
            >>> stats = uploader.upload(chunks)
            >>> print(f"Uploaded {stats['total_uploaded']} documents")
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting Milvus upload")
            print(f"{'='*70}")
            print(f"Total documents: {len(documents)}")
            print(f"Batch size: {self.batch_size}")
            print(f"{'='*70}\n")

        total_uploaded = 0
        num_batches = (len(documents) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(documents))
            batch = documents[start_idx:end_idx]

            if self.verbose:
                print(f"Uploading batch {batch_idx + 1}/{num_batches} ({len(batch)} documents)...")

            try:
                # Get vector store and add documents
                vector_store = self.vector_store_manager.get_vector_store()
                vector_store.add_documents(batch)

                total_uploaded += len(batch)

                if self.verbose:
                    print(f"Batch {batch_idx + 1} uploaded successfully")

            except Exception as e:
                if self.verbose:
                    print(f"Error uploading batch {batch_idx + 1}: {str(e)}")
                continue

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Milvus upload completed")
            print(f"{'='*70}")
            print(f"Total uploaded: {total_uploaded}/{len(documents)}")
            print(f"{'='*70}\n")

        return {
            "total_documents": len(documents),
            "total_uploaded": total_uploaded,
            "num_batches": num_batches,
            "batch_size": self.batch_size
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics

        Returns:
            Dict[str, Any]: Collection statistics
        """
        return self.vector_store_manager.get_stats()


# Convenience functions

def chunk_and_upload(
    documents: List[Document],
    api_key: str,
    milvus_uri: str = "http://localhost:19530",
    collection_name: str = "modelscope_docs",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 100,
    verbose: bool = False
) -> Dict[str, Any]:
    """Chunk documents and upload to Milvus - convenience function

    Args:
        documents: Input documents
        api_key: DashScope API key
        milvus_uri: Milvus URI
        collection_name: Collection name
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        batch_size: Upload batch size
        verbose: Verbose output

    Returns:
        Dict[str, Any]: Processing statistics

    Example:
        >>> stats = chunk_and_upload(
        ...     documents,
        ...     api_key="xxx",
        ...     verbose=True
        ... )
        >>> print(f"Uploaded {stats['total_uploaded']} chunks")
    """
    # Step 1: Chunk documents
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        verbose=verbose
    )
    chunks = chunker.chunk(documents)

    # Step 2: Upload to Milvus
    uploader = MilvusUploader(
        api_key=api_key,
        milvus_uri=milvus_uri,
        collection_name=collection_name,
        batch_size=batch_size,
        verbose=verbose
    )
    upload_stats = uploader.upload(chunks)

    # Combine stats
    return {
        "input_documents": len(documents),
        "total_chunks": len(chunks),
        "avg_chunks_per_doc": len(chunks) / len(documents) if documents else 0,
        **upload_stats
    }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Semantic Chunker Example")
    print("=" * 70)

    # Example 1: Basic usage
    print("\nExample 1: Create chunker")
    print("-" * 70)

    chunker = SemanticChunker(
        chunk_size=1000,
        chunk_overlap=200,
        verbose=True
    )

    # Get statistics
    stats = chunker.get_stats()
    print("\nChunker statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Example 2: Chunk document
    print("\nExample 2: Chunk document")
    print("-" * 70)

    sample_doc = Document(
        page_content="# Introduction\n\nThis is a sample document.\n\n" * 50,
        metadata={"source": "test.md", "title": "Test"}
    )

    chunks = chunker.chunk_document(sample_doc)
    print(f"Created {len(chunks)} chunks from document")
    print(f"First chunk length: {len(chunks[0].page_content)} characters")

    print("\n" + "=" * 70)
    print("Example execution completed")
    print("=" * 70)
