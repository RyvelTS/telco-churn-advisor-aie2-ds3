from typing import Any, List, Optional
from langchain_core.documents import Document

from src.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
)
from src.rag.indexer import get_azure_embeddings, get_azure_search_vectorstore


class AzurePolicyRetriever:
    """Service layer wrapping Azure AI Search vector similarity retrieval

    into a robust interface for the AI Retention Advisor Agent.
    """

    def __init__(self, top_k: int = 3, vector_store: Optional[Any] = None):
        self.top_k = top_k
        self._vector_store = vector_store

    @property
    def vector_store(self) -> Any:
        """Lazy-initializes the Azure AI Search vector store connection if not injected.

        Returns:
            Any: Connected Azure AI Search vector store client.

        Raises:
            ValueError: If Azure AI Search or Azure OpenAI credentials are not configured.
        """
        if self._vector_store is None:
            if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
                raise ValueError(
                    "Azure AI Search credentials not configured. Define "
                    "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY in your .env file."
                )
            if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
                raise ValueError(
                    "Azure OpenAI credentials not configured. Define "
                    "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file."
                )

            embeddings = get_azure_embeddings()
            self._vector_store = get_azure_search_vectorstore(embeddings)
        return self._vector_store

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieves raw LangChain Document excerpts matching the query.

        Args:
            query (str): The search phrase or topic.

        Returns:
            List[Document]: Excerpt documents from the index.

        Raises:
            Exception: If remote vector search encounters a transport or service error.
        """
        if not query or not query.strip():
            return []
        try:
            return self.vector_store.similarity_search(query.strip(), k=self.top_k)
        except Exception as e:
            print(f"[AzurePolicyRetriever Error] Retrieval failed: {e}")
            raise e

    def search(self, query: str) -> str:
        """Executes vector similarity search and formats results into a clean, numbered context string.

        Args:
            query (str): The search query regarding retention policies or plan terms.

        Returns:
            str: Human-readable policy excerpts formatted for LLM consumption, or an informative fallback message.
        """
        if not query or not query.strip():
            return "No matching company policy or retention terms found."

        try:
            docs = self.get_relevant_documents(query)
            if not docs:
                return "No matching company policy or retention terms found."

            formatted_excerpts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown Policy Document")
                page = doc.metadata.get("page", 1)
                content = doc.page_content.strip()

                header = f"[Policy Excerpt {i} (Source: {source}, Page: {page})]"
                formatted_excerpts.append(f"{header}:\n{content}")

            return "\n\n---\n\n".join(formatted_excerpts)

        except Exception as err:
            return (
                f"Policy retrieval currently unavailable due to remote connection issue: {str(err)}. "
                "Proceed with standard company retention guidelines."
            )
