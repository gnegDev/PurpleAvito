# Avito Service Splitter

Сервис для автоматического выявления самостоятельных услуг в объявлениях и генерации черновиков.
Решение для хакатона **IT Purple × Авито 2025**, категория «Ремонт и отделка».

> [!NOTE]
> Результаты тестов (`rnc_test.csv`) находяться в папке `target`.

> [!WARNING]  
> В связи с тем, что решение полагается на Yandex Cloud AI Assistants с уже настроенными агентами, для запуска вам потребуется определенный API-ключ (`YANDEX_CLOUD_AI_STUDIOS_KEY`). Свяжитесь с нами для его получения.

<img alt="PurpleAvito" src="https://github.com/user-attachments/assets/625f3d7d-5a60-4a74-b358-635a80e02851" />

---

## Системные требования

- Python 3.11+ (разработано и протестировано на 3.14)
- Или Docker
- ~500 МБ свободного места (embedding-модель ~120 МБ)
- Доступ к Yandex Cloud LLM API

---

## Быстрый старт

### Без Docker

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Создать .env
cp .env.example .env
# Отредактировать .env — добавить LLM_API_KEY

# 3. Запустить
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### С Docker

```bash
docker build -t avito-splitter .
docker run --env-file .env -p 8080:8080 avito-splitter
```

### Настройка `.env`

```
LLM_API_KEY=your_key_here
```

### Запуск тестов (опционально)

```
python run_test.py
```

---

## Интерфейсы

| Адрес | Описание |
|---|---|
| `http://localhost:8080/` | Веб-интерфейс для ручного тестирования |
| `http://localhost:8080/docs` | Swagger UI (автодокументация API) |

---

## Логика пайплайна

```
Входное объявление
        │
        ▼
┌───────────────────────────────┐
│  Шаг 1 — Поиск кандидатов     │  (без LLM)
│                               │
│  • Keyword matching:          │
│    ищем ключевые фразы        │
│    каждой микрокатегории      │
│    в тексте объявления        │
│                               │
│  • Embedding similarity:      │
│    paraphrase-multilingual-   │
│    MiniLM-L12-v2              │
│    (предвычисляется при       │
│    старте сервиса)            │
│                               │
│  Кандидат включается если:    │
│  keyword_score > 0            │
│  ИЛИ embedding_sim > 0.38     │
│                               │
│  → Топ-10 кандидатов          │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Шаг 2 — LLM классификация    │
│                               │
│  Yandex Cloud LLM определяет: │
│  • detectedMcIds — все        │
│    упомянутые микрокатегории  │
│  • shouldSplit — нужен ли     │
│    сплит                      │
│  • independentMcIds —         │
│    самостоятельные услуги     │
│                               │
│  При недоступности LLM →      │
│  fallback: shouldSplit=false  │
└───────────────┬───────────────┘
                │ (только если shouldSplit=true)
                ▼
┌───────────────────────────────┐
│  Шаг 3 — Генерация черновиков │
│                               │
│  LLM создаёт текст 50–100     │
│  слов для каждой              │
│  независимой услуги           │
└───────────────────────────────┘
```

---

## API Endpoints

### `GET /health`
Статус сервиса и количество загруженных микрокатегорий.

### `POST /analyze`
Анализирует объявление.

**Запрос:**
```json
{
  "itemId": 5001,
  "mcId": 101,
  "mcTitle": "Ремонт квартир и домов под ключ",
  "description": "Текст объявления..."
}
```

**Ответ:**
```json
{
  "itemId": 5001,
  "detectedMcIds": [102, 103],
  "shouldSplit": true,
  "drafts": [
    {"mcId": 102, "mcTitle": "Сантехника", "text": "..."}
  ],
  "reasoning": "...",
  "debug": {"candidatesFound": 4, "independentMcIds": [102, 103]}
}
```

### `POST /evaluate`
Запускает пайплайн на выборке из датасета и считает метрики.

**Запрос:** `{"limit": 50}`

**Ответ:**
```json
{
  "precision": 0.82,
  "recall": 0.75,
  "f1": 0.78,
  "shouldSplitAccuracy": 0.88,
  "totalItems": 50,
  "details": [...]
}
```

**Метрики:**
- **Precision / Recall / F1** — micro-average по `independentMcIds` vs `targetSplitMcIds`
- **shouldSplitAccuracy** — доля верно предсказанных `shouldSplit`
- Исходная микрокатегория объявления в расчёт не включается

---

## Структура проекта

```
├── app/
│   ├── main.py        — FastAPI: роуты, lifespan, раздача фронтенда
│   ├── pipeline.py    — оркестрация шагов 1 → 2 → 3
│   ├── matching.py    — keyword + embedding matching (класс Matcher)
│   ├── llm.py         — вызовы LLM (classify, generate_drafts)
│   ├── models.py      — Pydantic-модели запросов и ответов
│   └── data_loader.py — загрузка справочника и датасета
├── frontend/
│   └── index.html — веб-интерфейс (vanilla JS, без зависимостей)
├── datasets/
│   ├── rnc_mic_key_phrases.csv — справочник 11 микрокатегорий
│   └── rnc_dataset_markup.json — 2480 объявлений с эталонной разметкой
├── target/
│   └── rnc_test.csv — 159 тестовых запросов и ответов сервера 
├── Dockerfile
├── .env.example
└── requirements.txt
```
