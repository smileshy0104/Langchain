#!/usr/bin/env python
"""
Knowledge Base Loader Script

Load documents from multiple sources and build the knowledge base in Milvus.

Usage:
    python scripts/load_knowledge_base.py --source official
    python scripts/load_knowledge_base.py --source github --repo modelscope/modelscope
    python scripts/load_knowledge_base.py --source all --verbose

Features:
- Load from official docs and GitHub
- Clean and process documents
- Chunk semantically
- Score quality
- Upload to Milvus
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from data.loaders import OfficialDocsLoader, GitHubDocsLoader
from data.processing import (
    DocumentCleaner,
    SemanticChunker,
    MilvusUploader,
    QualityScorer
)


class KnowledgeBaseBuilder:
    """Knowledge base builder

    Orchestrate the full pipeline:
    1. Load documents from sources
    2. Clean and normalize content
    3. Score quality
    4. Chunk semantically
    5. Upload to Milvus

    Example:
        >>> builder = KnowledgeBaseBuilder(api_key="xxx")
        >>> builder.build_from_official_docs()
        >>> stats = builder.get_stats()
    """

    def __init__(
        self,
        api_key: str,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "modelscope_docs",
        github_token: str = None,
        verbose: bool = False
    ):
        """Initialize knowledge base builder

        Args:
            api_key: DashScope API key
            milvus_uri: Milvus connection URI
            collection_name: Milvus collection name
            github_token: GitHub token (optional)
            verbose: Verbose output
        """
        self.api_key = api_key
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.github_token = github_token
        self.verbose = verbose

        # Initialize components
        self.cleaner = DocumentCleaner(verbose=verbose)
        self.scorer = QualityScorer(verbose=verbose)
        self.chunker = SemanticChunker(verbose=verbose)
        self.uploader = MilvusUploader(
            api_key=api_key,
            milvus_uri=milvus_uri,
            collection_name=collection_name,
            verbose=verbose
        )

        # Statistics
        self.stats = {
            "total_loaded": 0,
            "total_cleaned": 0,
            "total_chunks": 0,
            "total_uploaded": 0
        }

        if verbose:
            print("=" * 70)
            print("Knowledge Base Builder Initialized")
            print("=" * 70)
            print(f"Milvus URI: {milvus_uri}")
            print(f"Collection: {collection_name}")
            print("=" * 70)

    def process_pipeline(self, documents: List[Document]) -> List[Document]:
        """Process documents through the full pipeline

        Args:
            documents: Raw documents

        Returns:
            List[Document]: Processed and chunked documents
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Processing Pipeline")
            print(f"{'='*70}\n")

        # Step 1: Clean documents
        if self.verbose:
            print("Step 1: Cleaning documents...")
        cleaned_docs = self.cleaner.clean(documents)
        self.stats["total_loaded"] += len(documents)
        self.stats["total_cleaned"] += len(cleaned_docs)

        if not cleaned_docs:
            print("Warning: No documents passed cleaning stage")
            return []

        # Step 2: Score quality
        if self.verbose:
            print("\nStep 2: Scoring quality...")
        scored_docs = self.scorer.score(cleaned_docs)

        # Step 3: Chunk documents
        if self.verbose:
            print("\nStep 3: Chunking documents...")
        chunks = self.chunker.chunk(scored_docs)
        self.stats["total_chunks"] += len(chunks)

        return chunks

    def build_from_official_docs(
        self,
        base_url: str = "https://modelscope.cn/docs",
        max_depth: int = 2
    ) -> dict:
        """Build knowledge base from official docs

        Args:
            base_url: Base URL for official docs
            max_depth: Maximum crawl depth

        Returns:
            dict: Build statistics
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Loading Official Documentation")
            print(f"{'='*70}\n")

        # Load documents
        loader = OfficialDocsLoader(
            base_url=base_url,
            max_depth=max_depth,
            verbose=self.verbose
        )

        try:
            # For demo purposes, load from specific URLs instead of recursive
            # In production, use loader.load() for full crawling
            sample_urls = [
                f"{base_url}/intro",
                f"{base_url}/models",
                f"{base_url}/datasets"
            ]

            if self.verbose:
                print(f"Note: Loading from {len(sample_urls)} sample URLs")
                print(f"For full crawl, use loader.load() instead")
                print()

            documents = loader.load_from_urls(sample_urls)

        except Exception as e:
            if self.verbose:
                print(f"Error loading official docs: {str(e)}")
                print("Continuing with empty document set...")
            documents = []

        if not documents:
            if self.verbose:
                print("No documents loaded from official sources")
            return {"error": "No documents loaded"}

        # Process pipeline
        chunks = self.process_pipeline(documents)

        if not chunks:
            return {"error": "No chunks created"}

        # Upload to Milvus
        if self.verbose:
            print("\nStep 4: Uploading to Milvus...")

        upload_stats = self.uploader.upload(chunks)
        self.stats["total_uploaded"] += upload_stats["total_uploaded"]

        return upload_stats

    def build_from_github(
        self,
        repo_owner: str,
        repo_name: str,
        branch: str = "main",
        max_depth: int = 10
    ) -> dict:
        """Build knowledge base from GitHub repository

        Args:
            repo_owner: Repository owner
            repo_name: Repository name
            branch: Branch name
            max_depth: Maximum directory depth

        Returns:
            dict: Build statistics
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Loading GitHub Repository")
            print(f"{'='*70}\n")

        # Load documents
        loader = GitHubDocsLoader(
            repo_owner=repo_owner,
            repo_name=repo_name,
            github_token=self.github_token,
            branch=branch,
            verbose=self.verbose
        )

        try:
            # For demo, load specific files
            # In production, use loader.load(max_depth=max_depth)
            sample_files = [
                "README.md",
                "docs/README.md",
                "CONTRIBUTING.md"
            ]

            if self.verbose:
                print(f"Note: Loading {len(sample_files)} sample files")
                print(f"For full repo, use loader.load() instead")
                print()

            documents = loader.load_specific_files(sample_files)

        except Exception as e:
            if self.verbose:
                print(f"Error loading GitHub repo: {str(e)}")
                print("Continuing with empty document set...")
            documents = []

        if not documents:
            if self.verbose:
                print("No documents loaded from GitHub")
            return {"error": "No documents loaded"}

        # Process pipeline
        chunks = self.process_pipeline(documents)

        if not chunks:
            return {"error": "No chunks created"}

        # Upload to Milvus
        if self.verbose:
            print("\nStep 4: Uploading to Milvus...")

        upload_stats = self.uploader.upload(chunks)
        self.stats["total_uploaded"] += upload_stats["total_uploaded"]

        return upload_stats

    def get_stats(self) -> dict:
        """Get builder statistics

        Returns:
            dict: Statistics
        """
        return self.stats.copy()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Load knowledge base from various sources"
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["official", "github", "all"],
        default="official",
        help="Data source to load from"
    )

    parser.add_argument(
        "--repo",
        type=str,
        help="GitHub repository (format: owner/repo)"
    )

    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="GitHub branch name"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("DASHSCOPE_API_KEY"),
        help="DashScope API key (or set DASHSCOPE_API_KEY env var)"
    )

    parser.add_argument(
        "--github-token",
        type=str,
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub token (or set GITHUB_TOKEN env var)"
    )

    parser.add_argument(
        "--milvus-uri",
        type=str,
        default=os.getenv("MILVUS_URI", "http://localhost:19530"),
        help="Milvus connection URI"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="modelscope_docs",
        help="Milvus collection name"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate API key
    if not args.api_key:
        print("Error: DashScope API key is required")
        print("Set DASHSCOPE_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Initialize builder
    builder = KnowledgeBaseBuilder(
        api_key=args.api_key,
        milvus_uri=args.milvus_uri,
        collection_name=args.collection,
        github_token=args.github_token,
        verbose=args.verbose
    )

    # Execute based on source
    if args.source == "official" or args.source == "all":
        print("\n" + "=" * 70)
        print("Building from Official Docs")
        print("=" * 70)

        stats = builder.build_from_official_docs()
        print(f"\nOfficial docs stats: {stats}")

    if args.source == "github" or args.source == "all":
        if not args.repo:
            print("\nError: --repo is required for GitHub source")
            print("Format: --repo owner/repo")
            sys.exit(1)

        # Parse repo
        try:
            repo_owner, repo_name = args.repo.split("/")
        except ValueError:
            print("\nError: Invalid repo format")
            print("Expected: owner/repo (e.g., modelscope/modelscope)")
            sys.exit(1)

        print("\n" + "=" * 70)
        print(f"Building from GitHub: {args.repo}")
        print("=" * 70)

        stats = builder.build_from_github(
            repo_owner=repo_owner,
            repo_name=repo_name,
            branch=args.branch
        )
        print(f"\nGitHub stats: {stats}")

    # Print final statistics
    print("\n" + "=" * 70)
    print("Final Statistics")
    print("=" * 70)

    final_stats = builder.get_stats()
    for key, value in final_stats.items():
        print(f"{key}: {value}")

    print("=" * 70)
    print("Knowledge Base Loading Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
