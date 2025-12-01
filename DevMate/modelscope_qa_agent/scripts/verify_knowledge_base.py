#!/usr/bin/env python
"""
Knowledge Base Verification Script

Verify the knowledge base is properly loaded and functional.

Usage:
    python scripts/verify_knowledge_base.py
    python scripts/verify_knowledge_base.py --verbose

Features:
- Check Milvus connection
- Verify collection exists
- Count documents
- Test retrieval
- Show sample documents
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from core.vector_store import VectorStoreManager


class KnowledgeBaseVerifier:
    """Knowledge base verifier

    Verify that the knowledge base is properly set up and functional.

    Example:
        >>> verifier = KnowledgeBaseVerifier(api_key="xxx")
        >>> results = verifier.run_all_checks()
        >>> print(results)
    """

    def __init__(
        self,
        api_key: str,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "modelscope_docs",
        verbose: bool = False
    ):
        """Initialize verifier

        Args:
            api_key: DashScope API key
            milvus_uri: Milvus URI
            collection_name: Collection name
            verbose: Verbose output
        """
        self.api_key = api_key
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.verbose = verbose

        # Initialize vector store manager
        try:
            self.vector_store_manager = VectorStoreManager(
                uri=milvus_uri,
                collection_name=collection_name,
                api_key=api_key
            )
            self.initialized = True
        except Exception as e:
            if verbose:
                print(f"Error initializing vector store: {str(e)}")
            self.initialized = False

    def check_connection(self) -> dict:
        """Check Milvus connection

        Returns:
            dict: Connection check results
        """
        if self.verbose:
            print("\nChecking Milvus connection...")

        if not self.initialized:
            return {
                "status": "failed",
                "message": "Failed to initialize vector store"
            }

        try:
            # Try to get stats to verify connection
            stats = self.vector_store_manager.get_stats()

            if self.verbose:
                print("Milvus connection successful")

            return {
                "status": "success",
                "uri": self.milvus_uri,
                "collection": self.collection_name
            }

        except Exception as e:
            if self.verbose:
                print(f"Milvus connection failed: {str(e)}")

            return {
                "status": "failed",
                "error": str(e)
            }

    def check_collection_stats(self) -> dict:
        """Check collection statistics

        Returns:
            dict: Collection statistics
        """
        if self.verbose:
            print("\nChecking collection statistics...")

        if not self.initialized:
            return {
                "status": "failed",
                "message": "Vector store not initialized"
            }

        try:
            stats = self.vector_store_manager.get_stats()

            if self.verbose:
                print(f"Collection stats retrieved successfully")
                for key, value in stats.items():
                    print(f"  {key}: {value}")

            return {
                "status": "success",
                **stats
            }

        except Exception as e:
            if self.verbose:
                print(f"Failed to get collection stats: {str(e)}")

            return {
                "status": "failed",
                "error": str(e)
            }

    def test_retrieval(self, query: str = "How to use ModelScope?", k: int = 3) -> dict:
        """Test retrieval functionality

        Args:
            query: Test query
            k: Number of results to retrieve

        Returns:
            dict: Retrieval test results
        """
        if self.verbose:
            print(f"\nTesting retrieval with query: '{query}'...")

        if not self.initialized:
            return {
                "status": "failed",
                "message": "Vector store not initialized"
            }

        try:
            # Get vector store and test retrieval
            vector_store = self.vector_store_manager.get_vector_store()
            results = vector_store.similarity_search(query, k=k)

            if self.verbose:
                print(f"Retrieved {len(results)} documents")
                print("\nSample results:")
                for i, doc in enumerate(results[:2], 1):
                    preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    print(f"\n  Result {i}:")
                    print(f"  Content: {preview}")
                    print(f"  Metadata: {doc.metadata}")

            return {
                "status": "success",
                "query": query,
                "num_results": len(results),
                "results_preview": [
                    {
                        "content_preview": doc.page_content[:100],
                        "metadata": doc.metadata
                    }
                    for doc in results[:2]
                ]
            }

        except Exception as e:
            if self.verbose:
                print(f"Retrieval test failed: {str(e)}")

            return {
                "status": "failed",
                "error": str(e)
            }

    def run_all_checks(self) -> dict:
        """Run all verification checks

        Returns:
            dict: All check results
        """
        if self.verbose:
            print("=" * 70)
            print("Knowledge Base Verification")
            print("=" * 70)

        results = {}

        # Check 1: Connection
        results["connection"] = self.check_connection()

        # Check 2: Collection stats
        results["collection_stats"] = self.check_collection_stats()

        # Check 3: Retrieval
        if results["connection"]["status"] == "success":
            results["retrieval"] = self.test_retrieval()
        else:
            results["retrieval"] = {
                "status": "skipped",
                "message": "Connection failed, skipping retrieval test"
            }

        # Summary
        all_passed = all(
            r.get("status") in ["success", "skipped"]
            for r in results.values()
        )

        results["summary"] = {
            "all_checks_passed": all_passed,
            "status": "success" if all_passed else "failed"
        }

        if self.verbose:
            print("\n" + "=" * 70)
            print("Verification Summary")
            print("=" * 70)
            print(f"Overall Status: {results['summary']['status'].upper()}")
            print("=" * 70)

        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Verify knowledge base setup and functionality"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("DASHSCOPE_API_KEY"),
        help="DashScope API key (or set DASHSCOPE_API_KEY env var)"
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
        "--query",
        type=str,
        default="How to use ModelScope?",
        help="Test query for retrieval"
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

    # Initialize verifier
    verifier = KnowledgeBaseVerifier(
        api_key=args.api_key,
        milvus_uri=args.milvus_uri,
        collection_name=args.collection,
        verbose=args.verbose
    )

    # Run all checks
    results = verifier.run_all_checks()

    # Print results if not verbose (verbose mode prints during execution)
    if not args.verbose:
        import json
        print("\nVerification Results:")
        print(json.dumps(results, indent=2))

    # Exit with appropriate code
    if results["summary"]["status"] == "success":
        print("\n✅ All verifications passed!")
        sys.exit(0)
    else:
        print("\n❌ Some verifications failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
