FROM python:3.11-slim

WORKDIR /app

# Зависимости отдельным слоем для кэширования
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Предзагрузка embedding-модели в образ
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY app/ ./app/
COPY datasets/ ./datasets/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
