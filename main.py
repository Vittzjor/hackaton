import faiss
import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from transformers import AutoModelForCausalLM, AutoTokenizer

# Импорты модулей
import crud
import models
import schemas
from config import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency для получения сессии базы данных
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Загрузка данных из Excel
def load_excel_data(excel_path):
    # Загружаем данные из Excel файла
    df = pd.read_excel(excel_path)
    # Переименовываем столбцы для удобства
    df.columns = [
        "Код", "Категория", "Время регистрации", "Рабочая группа", 
        "Краткое описание", "Описание", "Решение", "Аналитика 1", 
        "Аналитика 2", "Аналитика 3"
    ]
    # Конвертируем DataFrame в список словарей
    return df.to_dict(orient='records')

# Создание эмбеддингов
def create_embeddings(model, texts):
    return model.encode(texts)

file_path = './data.xlsx'


excel_data = load_excel_data('data.xlsx')

# Создание модели эмбеддингов

# Создание эмбеддингов для Excel данных

# Создание векторного индекса

# Загрузка модели LLaMA


# Функция генерации ответа
def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Модель для запроса
class ChatRequest(BaseModel):
    query: str

@app.get("/data")
def read_data():
    return excel_data.to_dict(orient='records')

@app.get("/questions")
def get_questions():
    questions = [item['Краткое описание'] for item in excel_data if 'Краткое описание' in item]
    return questions
# Маршрут для чата
@app.post("/chat", response_model=schemas.Message)
def chat(user: schemas.UserCreate, message: schemas.MessageCreate, db: Session = Depends(get_db)):
    user_query = message.request
    
    db_user = crud.create_user(db=db,user=user)
    # Преобразование запроса в вектор
    query_embedding = embedding_model.encode([user_query])
    
    # Поиск в индексе
    
    # Получение наиболее релевантного ответа из Excel данных
    
    # Генерация ответа с использованием модели LLaMA
    
    
    # Создание сообщения в базе данных
    db_message = crud.create_message(db, message, user_id=db_user.id)
    db_message.response = "фывапфыапф"
    db.commit()
    db.refresh(db_message)
    
    return db_message
