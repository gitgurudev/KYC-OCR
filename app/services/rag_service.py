"""
RAG Service — builds and queries a ChromaDB vector store of document schemas.

How it works:
  1. On startup, each document template is embedded and stored in ChromaDB.
  2. When the OCR pipeline identifies a document type (or needs disambiguation),
     the RAG service retrieves the best-matching schema via semantic similarity.
  3. The retrieved schema provides a precise extraction prompt + field list,
     so GPT-4o knows exactly which fields to pull for that document.
"""

import os
import json
import logging
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from app.data.document_templates import DOCUMENT_TEMPLATES

logger = logging.getLogger(__name__)

CHROMA_PATH = Path("chroma_db")


class RAGService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.vectorstore: Chroma | None = None
        self._build_vectorstore()

    def _build_vectorstore(self) -> None:
        """Embed all document templates and persist to ChromaDB."""
        documents = []
        for template in DOCUMENT_TEMPLATES:
            # Rich text content for embedding — combines description + keywords + fields
            field_names = ", ".join(template["fields"].keys())
            content = (
                f"Document Type: {template['display_name']}\n"
                f"Category: {template['category']}\n"
                f"Description: {template['description']}\n"
                f"Visual Keywords: {', '.join(template['visual_keywords'])}\n"
                f"Fields to Extract: {field_names}\n"
                f"Extraction Instructions: {template['extraction_prompt']}"
            )
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "type": template["type"],
                        "display_name": template["display_name"],
                        "category": template["category"],
                        "fields_json": json.dumps(template["fields"]),
                        "extraction_prompt": template["extraction_prompt"],
                    },
                )
            )

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_PATH),
            collection_name="kyc_document_schemas",
        )
        logger.info("RAG vector store built with %d document templates.", len(documents))

    def retrieve_schema(self, query: str, k: int = 1) -> dict:
        """
        Given a free-text description of a detected document, retrieve the
        best-matching document schema from ChromaDB.

        Returns a dict with keys: type, display_name, fields, extraction_prompt, score.
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialised.")

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return {}

        doc, score = results[0]
        return {
            "type": doc.metadata["type"],
            "display_name": doc.metadata["display_name"],
            "category": doc.metadata["category"],
            "fields": json.loads(doc.metadata["fields_json"]),
            "extraction_prompt": doc.metadata["extraction_prompt"],
            "similarity_score": round(float(score), 4),
        }

    def retrieve_top_schemas(self, query: str, k: int = 3) -> list[dict]:
        """Return top-k schemas for disambiguation display."""
        if self.vectorstore is None:
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        schemas = []
        for doc, score in results:
            schemas.append({
                "type": doc.metadata["type"],
                "display_name": doc.metadata["display_name"],
                "similarity_score": round(float(score), 4),
            })
        return schemas

    def get_schema_by_type(self, doc_type: str) -> dict:
        """Direct lookup by known document type string."""
        for template in DOCUMENT_TEMPLATES:
            if template["type"] == doc_type:
                return {
                    "type": template["type"],
                    "display_name": template["display_name"],
                    "category": template["category"],
                    "fields": template["fields"],
                    "extraction_prompt": template["extraction_prompt"],
                    "similarity_score": 1.0,
                }
        return {}


# Singleton — initialised once when the module is first imported
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
