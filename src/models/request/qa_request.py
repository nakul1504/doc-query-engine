from pydantic import BaseModel


class QARequest(BaseModel):
    question: str
    document_id: str
