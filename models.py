from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from config import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    messages = relationship("Message", back_populates="owner")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    request = Column(String, index=True)
    response = Column(String, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="messages")