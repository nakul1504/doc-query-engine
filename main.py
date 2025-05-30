from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api import ingestion_endpoint, document_qa_endpoint, document_endpoint, database_events, user_endpoint
from src.exception.exception_handler import (
    validation_exception_handler,
    custom_http_exception_handler,
    general_exception_handler,
)
from src.middleware.request_middleware import RequestIDMiddleware

app = FastAPI(
    title="Doc Query Engine",
    description="An API for document ingestion and RAG-based Q&A.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestIDMiddleware)

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, custom_http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.include_router(user_endpoint.router)
app.include_router(ingestion_endpoint.router)
app.include_router(document_qa_endpoint.router)
app.include_router(document_endpoint.router)
app.include_router(database_events.router)
