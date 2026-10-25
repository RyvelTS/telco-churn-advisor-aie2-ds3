from pathlib import Path
from typing import List, Optional, Union
import os

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_SEARCH_KEY,
    DOCS_DIR,
)


def validate_docs_directory(docs_dir: Union[str, Path]) -> List[Path]:
    """Validates that the policy directory exists and contains at least one readable document.

    Args:
        docs_dir (Union[str, Path]): Path to policy documentation directory.

    Returns:
        List[Path]: List of valid document paths (.pdf, .txt, .md).

    Raises:
        FileNotFoundError: If the directory does not exist or is not a directory.
        ValueError: If no supported documents are found in the directory.
    """
    path = Path(docs_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"Policy documentation directory '{path}' does not exist. "
            "Create the directory and place retention policies (.pdf, .txt, .md) inside."
        )

    supported_extensions = {".pdf", ".txt", ".md"}
    doc_files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in supported_extensions]

    if not doc_files:
        raise ValueError(
            f"No valid policy documents (.pdf, .txt, .md) found in '{path}'. "
            "At least one document is required to populate the knowledge base."
        )

    return doc_files


def load_policy_documents(docs_dir: Union[str, Path]) -> List[Document]:
    """Loads all PDF and Markdown/Text policy documents from the directory.

    Preserves source path and page metadata for vector attribution.

    Args:
        docs_dir (Union[str, Path]): Path to policy documentation directory.

    Returns:
        List[Document]: Parsed LangChain Document objects.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no supported documents are found.
    """
    path = Path(docs_dir)
    validate_docs_directory(path)

    documents: List[Document] = []

    # 1. Load PDF manuals with PyPDFLoader
    try:
        pdf_loader = DirectoryLoader(
            str(path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=False,
            use_multithreading=True,
        )
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
    except Exception as e:
        print(f"[Warning] PDF loader encountered an error: {e}")

    # 2. Load Text & Markdown policy files
    for pattern in ["**/*.txt", "**/*.md"]:
        try:
            txt_loader = DirectoryLoader(
                str(path),
                glob=pattern,
                loader_cls=TextLoader,
                loader_kwargs={"autodetect_encoding": True},
                show_progress=False,
            )
            txt_docs = txt_loader.load()
            for doc in txt_docs:
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 0
            documents.extend(txt_docs)
        except Exception as e:
            print(f"[Warning] Text loader for '{pattern}' encountered an error: {e}")

    # Standardize source path strings
    for doc in documents:
        if "source" in doc.metadata:
            doc.metadata["source"] = Path(doc.metadata["source"]).name

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Splits documents into overlapping chunks using a hierarchical header separator strategy."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        keep_separator=True,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def get_azure_embeddings() -> AzureOpenAIEmbeddings:
    """Configures and returns the Azure OpenAI Embeddings client for text-embedding-3-small.

    Returns:
        AzureOpenAIEmbeddings: Initialized embeddings client.

    Raises:
        ValueError: If AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY is not configured.
    """
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise ValueError(
            "Azure OpenAI credentials missing. Define AZURE_OPENAI_ENDPOINT and "
            "AZURE_OPENAI_API_KEY in your .env file."
        )

    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def get_azure_search_vectorstore(embeddings: AzureOpenAIEmbeddings) -> AzureSearch:
    """Initializes Azure AI Search vector store instance for index provisioning and insertion.

    Args:
        embeddings (AzureOpenAIEmbeddings): Embeddings client used for vector calculations.

    Returns:
        AzureSearch: Initialized Azure AI Search vector store client.

    Raises:
        ValueError: If AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_KEY is not configured.
    """
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
        raise ValueError(
            "Azure AI Search credentials missing. Define AZURE_SEARCH_ENDPOINT and "
            "AZURE_SEARCH_KEY in your .env file."
        )

    return AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query,
    )


def index_documents(docs_dir: Optional[Union[str, Path]] = None) -> int:
    """End-to-end RAG ingestion pipeline:

    Validate docs -> Load -> Chunk -> Embed -> Upload to Azure AI Search.

    Args:
        docs_dir (Optional[Union[str, Path]]): Custom directory path for documents. Defaults to DOCS_DIR.

    Returns:
        int: Total number of chunks successfully indexed.

    Raises:
        FileNotFoundError: If documentation directory is missing.
        ValueError: If credentials or documents are invalid.
    """
    target_dir = Path(docs_dir or DOCS_DIR)
    print("=" * 60)
    print(" STARTING AZURE AI SEARCH RAG INGESTION PIPELINE")
    print("=" * 60)
    print(f"Target Policy Directory : {target_dir}")

    # Ingest multi-format policy manuals from disk
    raw_docs = load_policy_documents(target_dir)
    print(f"Loaded Raw Pages/Files  : {len(raw_docs)} items")

    # 2. Chunk documents
    chunks = split_documents(raw_docs, chunk_size=500, chunk_overlap=50)
    print(f"Created Semantic Chunks : {len(chunks)} chunks (size=500, overlap=50)")

    # 3. Initialize Azure Embeddings & Vector Store
    embeddings = get_azure_embeddings()
    vector_store = get_azure_search_vectorstore(embeddings)

    # 4. Upload & provision index
    print(f"Uploading to Azure Search Index: '{AZURE_SEARCH_INDEX_NAME}'...")
    vector_store.add_documents(chunks)
    print(f"Successfully indexed {len(chunks)} policy chunks in Azure AI Search.")
    print("=" * 60)

    return len(chunks)


def is_index_populated() -> bool:
    """Checks whether the configured Azure AI Search index exists and contains documents.

    Returns:
        bool: True if index exists and contains documents; False otherwise.
    """
    from src.config import validate_azure_credentials

    if not validate_azure_credentials()["valid"]:
        return False

    try:
        embeddings = get_azure_embeddings()
        vector_store = get_azure_search_vectorstore(embeddings)
        probe = vector_store.similarity_search("retention discount policy", k=1)
        return len(probe) > 0
    except Exception:
        return False


def ensure_index_ready(docs_dir: Optional[Union[str, Path]] = None) -> bool:
    """Smart, idempotent RAG index initializer for local or clean-clone deployments.

    - If Azure credentials are not configured: skips gracefully without crashing.
    - If Azure AI Search index already contains documents: skips indexing (0 duplicates, 0 extra tokens).
    - If Azure AI Search index is empty or unprovisioned: loads, chunks, embeds, and indexes policy documents.

    Args:
        docs_dir (Optional[Union[str, Path]]): Document directory. Defaults to DOCS_DIR.

    Returns:
        bool: True if index is ready or successfully populated; False if credentials missing or indexing failed.
    """
    from src.config import validate_azure_credentials

    creds = validate_azure_credentials()
    if not creds["valid"]:
        return False

    try:
        if is_index_populated():
            return True

        print(f"[Info] Initializing unpopulated Azure Search index '{AZURE_SEARCH_INDEX_NAME}'...")
        count = index_documents(docs_dir=docs_dir)
        return count > 0
    except Exception as err:
        print(f"[Warning] RAG index auto-initialization skipped: {err}")
        return False


if __name__ == "__main__":
    index_documents()
