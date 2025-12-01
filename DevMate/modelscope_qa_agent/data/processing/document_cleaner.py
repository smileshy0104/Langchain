"""
Document Cleaner Module

Clean and normalize document content for knowledge base.

Core Functions:
- Remove HTML tags
- Normalize code blocks
- Remove special characters
- Standardize formatting
"""

import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from bs4 import BeautifulSoup


class DocumentCleaner:
    """Document content cleaner

    Clean and normalize document content, including:
    - HTML tag removal
    - Code block normalization
    - Whitespace cleanup
    - Special character handling

    Attributes:
        remove_html: Whether to remove HTML tags
        normalize_code: Whether to normalize code blocks
        remove_urls: Whether to remove URLs
        verbose: Whether to output detailed logs

    Example:
        >>> cleaner = DocumentCleaner()
        >>> documents = cleaner.clean(raw_documents)
        >>> print(f"Cleaned {len(documents)} documents")
    """

    def __init__(
        self,
        remove_html: bool = True,
        normalize_code: bool = True,
        remove_urls: bool = False,
        min_content_length: int = 50,
        verbose: bool = False
    ):
        """Initialize document cleaner

        Args:
            remove_html: Remove HTML tags (default: True)
            normalize_code: Normalize code blocks (default: True)
            remove_urls: Remove URLs from content (default: False)
            min_content_length: Minimum content length (default: 50)
            verbose: Output detailed logs
        """
        self.remove_html = remove_html
        self.normalize_code = normalize_code
        self.remove_urls = remove_urls
        self.min_content_length = min_content_length
        self.verbose = verbose

        if verbose:
            print("Document Cleaner initialized successfully")
            print(f"   - Remove HTML: {remove_html}")
            print(f"   - Normalize code: {normalize_code}")
            print(f"   - Remove URLs: {remove_urls}")
            print(f"   - Min content length: {min_content_length}")

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text

        Args:
            text: Input text

        Returns:
            str: Text with HTML tags removed
        """
        # Use BeautifulSoup to parse and extract text
        soup = BeautifulSoup(text, "html.parser")

        # Remove script and style tags
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def _normalize_code_blocks(self, text: str) -> str:
        """Normalize code blocks to standard format

        Args:
            text: Input text

        Returns:
            str: Text with normalized code blocks
        """
        # Pattern for code blocks with language specification
        # Matches: ```python, ```javascript, etc.
        code_pattern = r'```(\w+)?\n(.*?)```'

        def replace_code_block(match):
            lang = match.group(1) or ""
            code = match.group(2)

            # Remove leading/trailing whitespace from code
            code = code.strip()

            # Standardize format
            if lang:
                return f"```{lang}\n{code}\n```"
            else:
                return f"```\n{code}\n```"

        text = re.sub(code_pattern, replace_code_block, text, flags=re.DOTALL)

        # Also handle inline code
        # Normalize multiple backticks to single
        text = re.sub(r'`{2,}([^`]+)`{2,}', r'`\1`', text)

        return text

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text

        Args:
            text: Input text

        Returns:
            str: Text with URLs removed
        """
        # Pattern for URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        text = re.sub(url_pattern, '', text)

        return text

    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace

        Args:
            text: Input text

        Returns:
            str: Text with cleaned whitespace
        """
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)

        # Remove multiple newlines (keep maximum 2)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing/leading whitespace
        text = text.strip()

        return text

    def _remove_special_characters(self, text: str) -> str:
        """Remove special characters while preserving structure

        Args:
            text: Input text

        Returns:
            str: Text with special characters removed
        """
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

        # Remove control characters except newline and tab
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)

        return text

    def clean_text(self, text: str) -> str:
        """Clean a single text string

        Args:
            text: Input text

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Remove HTML tags
        if self.remove_html:
            text = self._remove_html_tags(text)

        # Normalize code blocks
        if self.normalize_code:
            text = self._normalize_code_blocks(text)

        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)

        # Remove special characters
        text = self._remove_special_characters(text)

        # Clean whitespace
        text = self._clean_whitespace(text)

        return text

    def clean_document(self, document: Document) -> Optional[Document]:
        """Clean a single document

        Args:
            document: Input document

        Returns:
            Optional[Document]: Cleaned document or None if too short
        """
        # Clean content
        cleaned_content = self.clean_text(document.page_content)

        # Check minimum length
        if len(cleaned_content) < self.min_content_length:
            if self.verbose:
                source = document.metadata.get("source", "unknown")
                print(f"Skipping document (too short): {source}")
            return None

        # Create new document with cleaned content
        return Document(
            page_content=cleaned_content,
            metadata=document.metadata.copy()
        )

    def clean(self, documents: List[Document]) -> List[Document]:
        """Clean a list of documents

        Args:
            documents: List of input documents

        Returns:
            List[Document]: List of cleaned documents

        Example:
            >>> cleaner = DocumentCleaner(verbose=True)
            >>> cleaned_docs = cleaner.clean(raw_documents)
            >>> print(f"Cleaned {len(cleaned_docs)} documents")
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting document cleaning")
            print(f"{'='*70}")
            print(f"Input documents: {len(documents)}")
            print(f"{'='*70}\n")

        cleaned_documents = []

        for i, doc in enumerate(documents):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"Processing: {i + 1}/{len(documents)} documents...")

            cleaned_doc = self.clean_document(doc)

            if cleaned_doc:
                cleaned_documents.append(cleaned_doc)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Document cleaning completed")
            print(f"{'='*70}")
            print(f"Input documents: {len(documents)}")
            print(f"Output documents: {len(cleaned_documents)}")
            print(f"Filtered out: {len(documents) - len(cleaned_documents)}")
            print(f"{'='*70}\n")

        return cleaned_documents

    def get_stats(self) -> Dict[str, Any]:
        """Get cleaner statistics

        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        return {
            "cleaner_type": "DocumentCleaner",
            "remove_html": self.remove_html,
            "normalize_code": self.normalize_code,
            "remove_urls": self.remove_urls,
            "min_content_length": self.min_content_length
        }


# Convenience function

def clean_documents(
    documents: List[Document],
    remove_html: bool = True,
    normalize_code: bool = True,
    remove_urls: bool = False,
    min_content_length: int = 50,
    verbose: bool = False
) -> List[Document]:
    """Clean documents - convenience function

    Args:
        documents: Input documents
        remove_html: Remove HTML tags
        normalize_code: Normalize code blocks
        remove_urls: Remove URLs
        min_content_length: Minimum content length
        verbose: Verbose output

    Returns:
        List[Document]: Cleaned documents

    Example:
        >>> cleaned_docs = clean_documents(raw_docs, verbose=True)
        >>> print(f"Cleaned {len(cleaned_docs)} documents")
    """
    cleaner = DocumentCleaner(
        remove_html=remove_html,
        normalize_code=normalize_code,
        remove_urls=remove_urls,
        min_content_length=min_content_length,
        verbose=verbose
    )
    return cleaner.clean(documents)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Document Cleaner Example")
    print("=" * 70)

    # Example 1: Basic usage
    print("\nExample 1: Create cleaner")
    print("-" * 70)

    cleaner = DocumentCleaner(
        remove_html=True,
        normalize_code=True,
        verbose=True
    )

    # Get statistics
    stats = cleaner.get_stats()
    print("\nCleaner statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Example 2: Clean text
    print("\nExample 2: Clean text")
    print("-" * 70)

    sample_text = """
    <html>
    <body>
    <h1>Sample Document</h1>
    <p>This is a test document with <strong>HTML tags</strong>.</p>

    <p>Code example:</p>
    <pre>
    ```python
    def hello():
        print("Hello World")
    ```
    </pre>

    <p>Visit https://example.com for more info.</p>
    </body>
    </html>
    """

    cleaned = cleaner.clean_text(sample_text)
    print(f"Original length: {len(sample_text)} characters")
    print(f"Cleaned length: {len(cleaned)} characters")
    print(f"\nCleaned text preview:\n{cleaned[:200]}...")

    # Example 3: Clean document
    print("\nExample 3: Clean document")
    print("-" * 70)

    sample_doc = Document(
        page_content=sample_text,
        metadata={"source": "test.html", "title": "Test Document"}
    )

    cleaned_doc = cleaner.clean_document(sample_doc)
    if cleaned_doc:
        print(f"Document cleaned successfully")
        print(f"Metadata preserved: {cleaned_doc.metadata}")

    print("\n" + "=" * 70)
    print("Example execution completed")
    print("=" * 70)
