import shutil
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.orm import Session

import models
import schemas


def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User()
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_message(db: Session, message: schemas.MessageCreate, user_id: int):
    res = get_response(request=message.request)
    
    db_message = models.Message(request=message.request, response=res, owner_id=user_id)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def get_response(request: str):
    
    return