import sys
from typing import List

import faiss
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from transformers import LlamaForCausalLM, LlamaTokenizer

from . import crud, models, schemas
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Извлечение текста из PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Загрузка данных из Excel
def load_excel_data(excel_path):
    df = pd.read_excel(excel_path)
    return df.to_dict(orient='records')

# Создание эмбеддингов
def create_embeddings(model, texts):
    return model.encode(texts)

# Загрузка и подготовка данных
pdf_text = extract_text_from_pdf('document.pdf')
excel_data = load_excel_data('data.xlsx')
questions = [item['question'] for item in excel_data]

# Создание модели эмбеддингов
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Создание эмбеддингов для PDF и Excel данных
pdf_embeddings = create_embeddings(embedding_model, [pdf_text])
excel_embeddings = create_embeddings(embedding_model, questions)

# Комбинирование всех эмбеддингов
all_embeddings = np.vstack((pdf_embeddings, excel_embeddings))

# Создание векторного индекса
index = faiss.IndexFlatL2(all_embeddings.shape[1])
index.add(all_embeddings)

# Загрузка модели LLaMA
model_name = 'open-llama'
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

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
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    user_query = request.query
    
    # Преобразование запроса в вектор
    query_embedding = embedding_model.encode([user_query])
    
    # Поиск в индексе
    D, I = index.search(query_embedding, k=1)
    
    # Получение наиболее релевантного документа
    if I[0][0] < len(pdf_embeddings):
        relevant_document = pdf_text
    else:
        relevant_document = excel_data[I[0][0] - len(pdf_embeddings)]['answer']
    
    # Генерация ответа с использованием модели LLaMA
    prompt = f"User question: {user_query}\nRelevant document: {relevant_document}\nAnswer:"
    response_text = generate_response(prompt, model, tokenizer)
    
    # Создание сообщения в базе данных
    message_create = schemas.MessageCreate(request=user_query)
    db_message = crud.create_message(db, message_create, user_id=1)  # Здесь user_id должен быть актуальным
    db_message.response = response_text
    db.commit()
    db.refresh(db_message)
    
    return db_message
