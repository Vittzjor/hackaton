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
    df = pd.read_excel(excel_path)
    return df

# Создание эмбеддингов
def create_embeddings(model, texts):
    return model.encode(texts)

# Загрузка и подготовка данных
excel_data = load_excel_data('./data.xlsx')
questions = [item['question'] for item in excel_data]

# Создание модели эмбеддингов
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Создание эмбеддингов для Excel данных
excel_embeddings = create_embeddings(embedding_model, questions)

# Создание векторного индекса
index = faiss.IndexFlatL2(excel_embeddings.shape[1])
index.add(excel_embeddings)

# Загрузка модели LLaMA
model_name = 'open-llama'
model = AutoModelForCausalLM.from_pretrained('model_name')
tokenizer = AutoTokenizer.from_pretrained('model_name')

# Функция генерации ответа
def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Модель для запроса
class ChatRequest(BaseModel):
    query: str

# Маршрут для чата
@app.post("/chat", response_model=schemas.Message)
def chat(user: schemas.UserCreate, message: schemas.MessageCreate, db: Session = Depends(get_db)):
    user_query = message.request
    
    db_user = crud.create_user(db=db,user=user)
    # Преобразование запроса в вектор
    query_embedding = embedding_model.encode([user_query])
    
    # Поиск в индексе
    D, I = index.search(query_embedding, k=1)
    
    # Получение наиболее релевантного ответа из Excel данных
    relevant_document = excel_data[I[0][0]]['answer']
    
    # Генерация ответа с использованием модели LLaMA
    prompt = f"User question: {user_query}\nRelevant document: {relevant_document}\nAnswer:"
    response_text = generate_response(prompt, model, tokenizer)
    
    # Создание сообщения в базе данных
    db_message = crud.create_message(db, message, user_id=db_user.id)
    db_message.response = response_text
    db.commit()
    db.refresh(db_message)
    
    return db_message
