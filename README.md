# Avito Service Splitter

Автоматическое выделение самостоятельных услуг из объявлений и генерация черновиков.

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Настройка .env

Скопируй `.env.example` в `.env` и заполни ключ:

```bash
cp .env.example .env
```

Отредактируй `.env`:
```
LLM_API_KEY=your_actual_key_here
```

## Запуск

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Документация API доступна по адресу: http://localhost:8000/docs

## Endpoints

### GET /health
Статус сервиса и количество загруженных микрокатегорий.

### POST /analyze
Анализирует объявление и возвращает детектированные микрокатегории, решение о сплите и черновики.

```json
{
  "itemId": 5001,
  "mcId": 201,
  "mcTitle": "Ремонт квартир под ключ",
  "description": "Текст объявления..."
}
```

### POST /evaluate
Запускает пайплайн на `limit` записях датасета и считает метрики (precision, recall, F1, accuracy по shouldSplit).

```json
{"limit": 50}
```

## Архитектура

1. **Шаг 1 — Keyword + Embedding matching** (`matching.py`): ищет кандидатов микрокатегорий через ключевые фразы и косинусное сходство embeddings (paraphrase-multilingual-MiniLM-L12-v2).
2. **Шаг 2 — LLM классификация** (`llm.py`): определяет, какие микрокатегории упомянуты самостоятельно, через Yandex Cloud LLM.
3. **Шаг 3 — Генерация черновиков** (`llm.py`): создаёт тексты для каждой самостоятельной микрокатегории (только если `shouldSplit=true`).
