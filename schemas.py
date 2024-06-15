from typing import List

from pydantic import BaseModel


class MessageBase(BaseModel):
    request: str
    response: str

class MessageCreate(BaseModel):
    request: str

class Message(MessageBase):
    id: int
    response: str
    owner_id: int   

    class Config:
        orm_mode = True

class UserBase(BaseModel):
    pass

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    messages: List[Message] = []

    class Config:
        orm_mode = True