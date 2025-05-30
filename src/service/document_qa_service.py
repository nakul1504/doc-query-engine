import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from config import INACTIVITY_THRESHOLD_HOURS, EMBEDDING_MODEL, QA_PIPELINE_TASK, QA_PIPELINE_MODEL
from src.models.entities.document_entity import Document as DocumentModel
from langchain.docstore.document import Document as LangchainDoc
from sqlalchemy import select
from transformers import pipeline
from src.core.database import SessionLocal
import spacy

from src.models.request.qa_request import QARequest
from src.util.error_utils import raise_http_exception
from src.util.logging_utils import get_logger

logger = get_logger("document_qa_service")
logger.setLevel(logging.INFO)


class DocumentQAService:
    """
    DocumentQAService provides methods for handling document-based question answering
    using a retrieval-augmented generation (RAG) approach. It manages the creation,
    storage, and retrieval of FAISS indexes for document embeddings, validates QA
    requests, and generates answers by processing document content. The service
    utilizes language models and embeddings to extract factual information from
    documents and maintains metadata for index management.
    """
    INDEX_DIR = Path("./faiss_indexes")
    METADATA_FILE = INDEX_DIR / "index_metadata.json"

    nlp = spacy.load("en_core_web_sm")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    qa_pipeline = pipeline(task=QA_PIPELINE_TASK, model=QA_PIPELINE_MODEL)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    prompt_template = PromptTemplate.from_template(
        "You are a helpful assistant. Based on the context provided below, extract factual information as accurately as possible. "
        "Only use the context to answer. If the answer is not explicitly present, respond with 'The document does not contain relevant information.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    @staticmethod
    async def validate_qa_request(qa_request: QARequest):
        if not qa_request.question:
            logger.error("Question cannot be empty")
            raise_http_exception(status_code=400, message="Question cannot be empty")
        if not qa_request.document_id:
            logger.error("Document ID cannot be empty")
            raise_http_exception(status_code=400, message="Document ID is required")

    @staticmethod
    async def generate_answer_by_id(question: str, document_id: str):
        async with SessionLocal() as session:
            result = await session.execute(
                select(DocumentModel).where(DocumentModel.id == document_id)
            )
            doc = result.scalar_one_or_none()
            if not doc:
                return {"answer": f"No document found with id {document_id}"}

            doc_nlp = DocumentQAService.nlp(doc.content)
            sentences = [sent.text.strip() for sent in doc_nlp.sents]

            chunks, chunk, length = [], [], 0
            for sentence in sentences:
                if length + len(sentence) > 1600:
                    chunks.append(" ".join(chunk))
                    chunk, length = [], 0
                chunk.append(sentence)
                length += len(sentence)
            if chunk:
                chunks.append(" ".join(chunk))

            langchain_docs = [
                LangchainDoc(page_content=chunk, metadata={"document_id": doc.id})
                for chunk in chunks
            ]

            vectorstore = DocumentQAService.load_faiss_index(document_id)
            if not vectorstore:
                vectorstore = FAISS.from_documents(
                    langchain_docs, DocumentQAService.embedding_model
                )
                DocumentQAService.save_faiss_index(vectorstore, document_id)

            retriever = vectorstore.as_retriever()

            rag_chain = RetrievalQA.from_chain_type(
                llm=DocumentQAService.llm,
                retriever=retriever,
                chain_type_kwargs={"prompt": DocumentQAService.prompt_template},
            )

            return {"answer": rag_chain.invoke(question)}

    @staticmethod
    def save_faiss_index(vectorstore, document_id):
        DocumentQAService.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        index_path = DocumentQAService.INDEX_DIR / f"{document_id}.faiss"
        vectorstore.save_local(str(index_path))

        metadata = DocumentQAService.load_metadata()
        metadata[document_id] = {"last_access": datetime.now().isoformat()}
        DocumentQAService.save_metadata(metadata)

    @staticmethod
    def load_faiss_index(document_id):
        index_path = DocumentQAService.INDEX_DIR / f"{document_id}.faiss"
        if not index_path.exists():
            return None

        metadata = DocumentQAService.load_metadata()
        metadata[document_id] = {"last_access": datetime.now().isoformat()}
        DocumentQAService.save_metadata(metadata)

        return FAISS.load_local(
            str(index_path),
            DocumentQAService.embedding_model,
            allow_dangerous_deserialization=True,
        )

    @staticmethod
    def load_metadata():
        if DocumentQAService.METADATA_FILE.exists():
            with open(DocumentQAService.METADATA_FILE, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def save_metadata(metadata):
        with open(DocumentQAService.METADATA_FILE, "w") as f:
            json.dump(metadata, f)

    @staticmethod
    def clean_old_indexes():
        now = datetime.now()
        metadata = DocumentQAService.load_metadata()
        updated_metadata = {}

        for document_id, info in metadata.items():
            last_access = datetime.fromisoformat(info["last_access"])
            if now - last_access > timedelta(hours=INACTIVITY_THRESHOLD_HOURS):
                index_path = DocumentQAService.INDEX_DIR / f"{document_id}.faiss"
                if index_path.exists():
                    os.remove(index_path)
            else:
                updated_metadata[document_id] = info

        DocumentQAService.save_metadata(updated_metadata)
        return updated_metadata
