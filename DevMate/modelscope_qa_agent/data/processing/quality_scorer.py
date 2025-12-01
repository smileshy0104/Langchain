"""
Quality Scorer Module

Score document quality and add metadata tags.

Core Functions:
- Content quality scoring
- Code presence detection
- Technical term extraction
- Metadata enrichment
"""

import re
from typing import List, Dict, Any
from langchain_core.documents import Document


class QualityScorer:
    """Document quality scorer

    Score documents based on multiple quality metrics:
    - Content length and completeness
    - Code block presence
    - Technical terminology
    - Structure and formatting

    Attributes:
        min_length: Minimum content length for quality check
        code_weight: Weight for code presence (0-1)
        term_weight: Weight for technical terms (0-1)
        structure_weight: Weight for structure (0-1)
        verbose: Whether to output detailed logs

    Example:
        >>> scorer = QualityScorer()
        >>> scored_docs = scorer.score(documents)
        >>> print(f"Scored {len(scored_docs)} documents")
    """

    def __init__(
        self,
        min_length: int = 100,
        code_weight: float = 0.3,
        term_weight: float = 0.4,
        structure_weight: float = 0.3,
        verbose: bool = False
    ):
        """Initialize quality scorer

        Args:
            min_length: Minimum content length (default: 100)
            code_weight: Weight for code blocks (default: 0.3)
            term_weight: Weight for technical terms (default: 0.4)
            structure_weight: Weight for structure (default: 0.3)
            verbose: Output detailed logs
        """
        self.min_length = min_length
        self.code_weight = code_weight
        self.term_weight = term_weight
        self.structure_weight = structure_weight
        self.verbose = verbose

        # Technical terms dictionary
        self.technical_terms = {
            'api', 'model', 'training', 'inference', 'dataset', 'pipeline',
            'transformer', 'embedding', 'tokenizer', 'checkpoint', 'framework',
            'pytorch', 'tensorflow', 'huggingface', 'modelscope', 'configuration',
            'parameter', 'hyperparameter', 'optimization', 'loss', 'accuracy',
            'precision', 'recall', 'metrics', 'evaluation', 'benchmark',
            'deployment', 'serving', 'endpoint', 'gpu', 'cuda', 'tensor'
        }

        if verbose:
            print("Quality Scorer initialized successfully")
            print(f"   - Min length: {min_length}")
            print(f"   - Weights: code={code_weight}, terms={term_weight}, structure={structure_weight}")

    def _score_code_presence(self, content: str) -> float:
        """Score code block presence

        Args:
            content: Document content

        Returns:
            float: Code score (0-1)
        """
        # Count code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        inline_code = re.findall(r'`[^`]+`', content)

        # Calculate score
        code_score = 0.0

        # Code blocks contribute more
        if len(code_blocks) > 0:
            code_score += min(len(code_blocks) * 0.3, 0.7)

        # Inline code contributes less
        if len(inline_code) > 0:
            code_score += min(len(inline_code) * 0.05, 0.3)

        return min(code_score, 1.0)

    def _score_technical_terms(self, content: str) -> float:
        """Score technical terminology presence

        Args:
            content: Document content

        Returns:
            float: Term score (0-1)
        """
        content_lower = content.lower()

        # Count technical terms
        term_count = sum(1 for term in self.technical_terms if term in content_lower)

        # Calculate score (normalized)
        max_expected_terms = 10
        term_score = min(term_count / max_expected_terms, 1.0)

        return term_score

    def _score_structure(self, content: str) -> float:
        """Score document structure

        Args:
            content: Document content

        Returns:
            float: Structure score (0-1)
        """
        score = 0.0

        # Check for headings
        headings = re.findall(r'^#+\s+.+', content, re.MULTILINE)
        if len(headings) > 0:
            score += min(len(headings) * 0.1, 0.3)

        # Check for lists
        lists = re.findall(r'^\s*[-*+]\s+.+', content, re.MULTILINE)
        if len(lists) > 0:
            score += min(len(lists) * 0.05, 0.2)

        # Check for paragraphs
        paragraphs = content.split('\n\n')
        if len(paragraphs) >= 2:
            score += 0.2

        # Check for links
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        if len(links) > 0:
            score += min(len(links) * 0.05, 0.15)

        # Check for emphasis
        emphasis = re.findall(r'\*\*([^*]+)\*\*|__([^_]+)__', content)
        if len(emphasis) > 0:
            score += min(len(emphasis) * 0.03, 0.15)

        return min(score, 1.0)

    def calculate_quality_score(self, document: Document) -> float:
        """Calculate overall quality score

        Args:
            document: Input document

        Returns:
            float: Quality score (0-1)
        """
        content = document.page_content

        # Length check
        if len(content) < self.min_length:
            return 0.0

        # Calculate component scores
        code_score = self._score_code_presence(content)
        term_score = self._score_technical_terms(content)
        structure_score = self._score_structure(content)

        # Weighted average
        quality_score = (
            code_score * self.code_weight +
            term_score * self.term_weight +
            structure_score * self.structure_weight
        )

        return quality_score

    def extract_tags(self, document: Document) -> List[str]:
        """Extract metadata tags from document

        Args:
            document: Input document

        Returns:
            List[str]: List of tags
        """
        tags = set()
        content = document.page_content.lower()

        # Source type tag
        source_type = document.metadata.get("source_type", "")
        if source_type:
            tags.add(source_type)

        # Code presence tags
        if '```python' in content:
            tags.add('python')
            tags.add('code')
        if '```javascript' in content or '```js' in content:
            tags.add('javascript')
            tags.add('code')
        if '```java' in content:
            tags.add('java')
            tags.add('code')
        if '```cpp' in content or '```c++' in content:
            tags.add('cpp')
            tags.add('code')

        # Technical area tags
        if any(term in content for term in ['model', 'training', 'inference']):
            tags.add('ml')
        if any(term in content for term in ['api', 'endpoint', 'request']):
            tags.add('api')
        if any(term in content for term in ['dataset', 'data']):
            tags.add('data')
        if any(term in content for term in ['deployment', 'serving']):
            tags.add('deployment')

        # Document type tags
        if document.metadata.get("file_path", "").endswith('.md'):
            tags.add('markdown')
        if 'readme' in document.metadata.get("file_name", "").lower():
            tags.add('readme')
        if 'tutorial' in document.metadata.get("source", "").lower():
            tags.add('tutorial')
        if 'guide' in document.metadata.get("source", "").lower():
            tags.add('guide')

        return sorted(list(tags))

    def enrich_document(self, document: Document) -> Document:
        """Enrich document with quality score and tags

        Args:
            document: Input document

        Returns:
            Document: Enriched document
        """
        # Calculate quality score
        quality_score = self.calculate_quality_score(document)

        # Extract tags
        tags = self.extract_tags(document)

        # Create enriched metadata
        enriched_metadata = document.metadata.copy()
        enriched_metadata.update({
            'quality_score': quality_score,
            'tags': tags,
            'has_code': any('code' in tag or tag in ['python', 'javascript', 'java', 'cpp'] for tag in tags),
            'content_length': len(document.page_content)
        })

        return Document(
            page_content=document.page_content,
            metadata=enriched_metadata
        )

    def score(self, documents: List[Document]) -> List[Document]:
        """Score and enrich a list of documents

        Args:
            documents: List of input documents

        Returns:
            List[Document]: List of enriched documents

        Example:
            >>> scorer = QualityScorer(verbose=True)
            >>> scored_docs = scorer.score(documents)
            >>> avg_score = sum(d.metadata['quality_score'] for d in scored_docs) / len(scored_docs)
            >>> print(f"Average quality score: {avg_score:.2f}")
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting quality scoring")
            print(f"{'='*70}")
            print(f"Input documents: {len(documents)}")
            print(f"{'='*70}\n")

        enriched_documents = []

        for i, doc in enumerate(documents):
            if self.verbose and (i + 1) % 50 == 0:
                print(f"Scoring: {i + 1}/{len(documents)} documents...")

            enriched_doc = self.enrich_document(doc)
            enriched_documents.append(enriched_doc)

        if self.verbose:
            # Calculate statistics
            scores = [d.metadata.get('quality_score', 0) for d in enriched_documents]
            avg_score = sum(scores) / len(scores) if scores else 0
            high_quality = sum(1 for s in scores if s >= 0.7)
            medium_quality = sum(1 for s in scores if 0.4 <= s < 0.7)
            low_quality = sum(1 for s in scores if s < 0.4)

            print(f"\n{'='*70}")
            print(f"Quality scoring completed")
            print(f"{'='*70}")
            print(f"Total documents: {len(enriched_documents)}")
            print(f"Average quality: {avg_score:.2f}")
            print(f"High quality (>=0.7): {high_quality}")
            print(f"Medium quality (0.4-0.7): {medium_quality}")
            print(f"Low quality (<0.4): {low_quality}")
            print(f"{'='*70}\n")

        return enriched_documents

    def get_stats(self) -> Dict[str, Any]:
        """Get scorer statistics

        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        return {
            "scorer_type": "QualityScorer",
            "min_length": self.min_length,
            "code_weight": self.code_weight,
            "term_weight": self.term_weight,
            "structure_weight": self.structure_weight,
            "num_technical_terms": len(self.technical_terms)
        }


# Convenience function

def score_documents(
    documents: List[Document],
    min_length: int = 100,
    verbose: bool = False
) -> List[Document]:
    """Score documents - convenience function

    Args:
        documents: Input documents
        min_length: Minimum content length
        verbose: Verbose output

    Returns:
        List[Document]: Scored documents

    Example:
        >>> scored_docs = score_documents(docs, verbose=True)
        >>> high_quality = [d for d in scored_docs if d.metadata['quality_score'] >= 0.7]
        >>> print(f"Found {len(high_quality)} high-quality documents")
    """
    scorer = QualityScorer(
        min_length=min_length,
        verbose=verbose
    )
    return scorer.score(documents)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Quality Scorer Example")
    print("=" * 70)

    # Example 1: Basic usage
    print("\nExample 1: Create scorer")
    print("-" * 70)

    scorer = QualityScorer(verbose=True)

    # Get statistics
    stats = scorer.get_stats()
    print("\nScorer statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Example 2: Score document
    print("\nExample 2: Score document")
    print("-" * 70)

    sample_doc = Document(
        page_content="""
# Model Training Guide

This guide shows how to train a model using ModelScope.

## Installation

```python
pip install modelscope
```

## Usage

Import the necessary modules:

```python
from modelscope.trainers import Trainer
from modelscope.models import Model
```

Train your model:

```python
trainer = Trainer(model, dataset)
trainer.train()
```

For more information, visit the [documentation](https://modelscope.cn/docs).
        """,
        metadata={"source": "training_guide.md", "file_path": "docs/training_guide.md"}
    )

    enriched_doc = scorer.enrich_document(sample_doc)
    print(f"\nQuality score: {enriched_doc.metadata['quality_score']:.2f}")
    print(f"Tags: {enriched_doc.metadata['tags']}")
    print(f"Has code: {enriched_doc.metadata['has_code']}")

    print("\n" + "=" * 70)
    print("Example execution completed")
    print("=" * 70)
