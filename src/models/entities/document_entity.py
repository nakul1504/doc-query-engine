from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Column, String, Text, ForeignKey
from sqlalchemy.orm import relationship

from src.core.database import Base
from src.models.entities.user import User

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(VECTOR(1024))

    owner_id = Column(String, ForeignKey("users.id"), nullable=False)

    owner = relationship("User", backref="documents")
