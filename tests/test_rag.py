from pathlib import Path
from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document

from src.rag.indexer import (
    load_policy_documents,
    split_documents,
    validate_docs_directory,
)
from src.rag.retriever import AzurePolicyRetriever


def test_validate_docs_directory_success():
    """Verify that existing data/docs directory passes validation."""
    docs = validate_docs_directory("data/docs")
    assert len(docs) >= 1
    assert any(p.suffix.lower() == ".pdf" for p in docs)


def test_validate_docs_directory_non_existent(tmp_path):
    """Ensure missing folder raises FileNotFoundError."""
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        validate_docs_directory(missing_dir)


def test_validate_docs_directory_empty(tmp_path):
    """Ensure empty directory raises ValueError."""
    empty_dir = tmp_path / "empty_docs"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        validate_docs_directory(empty_dir)


def test_load_policy_documents():
    """Verify loading real policy documents and metadata extraction."""
    docs = load_policy_documents("data/docs")
    assert len(docs) > 0
    for doc in docs:
        assert isinstance(doc, Document)
        assert len(doc.page_content.strip()) > 0
        assert "source" in doc.metadata
        assert "page" in doc.metadata


def test_semantic_chunking_strategy():
    """Verify chunk size, overlap, and metadata preservation."""
    sample_doc = Document(
        page_content="## Retention Policy Overview\n" + ("Rule 1: Offer 15% discount for 2-year commitments.\n" * 20),
        metadata={"source": "test_policy.pdf", "page": 1},
    )
    chunks = split_documents([sample_doc], chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 600  # Within boundary
        assert chunk.metadata["source"] == "test_policy.pdf"
        assert chunk.metadata["page"] == 1


def test_retriever_formatting_with_mock_vector_store():
    """Verify AzurePolicyRetriever formats output properly with clean document source citations."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [
        Document(
            page_content="Customers with tenure > 24 months are eligible for Fiber upgrade discount.",
            metadata={"source": "retention_policy.pdf"},
        ),
        Document(
            page_content="Early termination fee is waived for moving to non-serviceable areas.",
            metadata={"source": "plan_comparison.pdf"},
        ),
    ]

    retriever = AzurePolicyRetriever(top_k=2, vector_store=mock_vs)
    output = retriever.search("retention discount")

    assert "[Policy Excerpt 1 (Source: retention_policy.pdf)]:" in output
    assert "Fiber upgrade discount" in output
    assert "\n\n---\n\n" in output
    assert "[Policy Excerpt 2 (Source: plan_comparison.pdf)]:" in output
    assert "Early termination fee is waived" in output


def test_retriever_empty_query():
    """Verify empty query returns fallback message without raising error."""
    retriever = AzurePolicyRetriever(vector_store=MagicMock())
    assert retriever.search("") == "No matching company policy or retention terms found."
    assert retriever.search("   ") == "No matching company policy or retention terms found."


def test_retriever_handles_exception_gracefully():
    """Ensure vector store connection errors return a descriptive message rather than crashing."""
    failing_vs = MagicMock()
    failing_vs.similarity_search.side_effect = ConnectionError("Azure Search connection timed out")

    retriever = AzurePolicyRetriever(vector_store=failing_vs)
    result = retriever.search("What is the discount rate?")

    assert "Policy retrieval currently unavailable" in result
    assert "Azure Search connection timed out" in result


def test_retriever_fallback_source_when_missing():
    """Verify that documents without explicit source metadata fallback to 'Unknown Policy Document'."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [
        Document(
            page_content="Policy starting page content",
            metadata={},
        ),
    ]

    retriever = AzurePolicyRetriever(top_k=1, vector_store=mock_vs)
    output = retriever.search("policy details")

    assert "[Policy Excerpt 1 (Source: Unknown Policy Document)]:" in output


def test_ensure_index_ready_when_missing_credentials(monkeypatch):
    """Verify ensure_index_ready returns False gracefully when Azure credentials missing."""
    from src.rag.indexer import ensure_index_ready

    monkeypatch.setattr("src.config.AZURE_SEARCH_ENDPOINT", "")
    monkeypatch.setattr("src.config.AZURE_SEARCH_KEY", "")

    result = ensure_index_ready()
    assert result is False


def test_ensure_index_ready_skips_when_already_populated(monkeypatch):
    """Verify ensure_index_ready skips re-indexing when index contains documents."""
    from src.rag.indexer import ensure_index_ready

    monkeypatch.setattr(
        "src.config.validate_azure_credentials",
        lambda: {"valid": True, "missing_keys": []},
    )
    monkeypatch.setattr("src.rag.indexer.is_index_populated", lambda: True)

    result = ensure_index_ready()
    assert result is True
