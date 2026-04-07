import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import pipeline
from app.data_loader import load_dataset, load_microcategories
from app.matching import Matcher
from app.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    CandidateMicrocategory,
    DebugInfo,
    DraftItem,
    EvaluateDetail,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Глобальное состояние
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск сервиса: загрузка данных и моделей...")

    microcategories = load_microcategories()
    _state["microcategories"] = microcategories
    _state["mc_lookup"] = {mc["mcId"]: mc for mc in microcategories}

    logger.info("Загрузка sentence-transformer модели...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    _state["matcher"] = Matcher(microcategories, model)

    logger.info("Сервис готов к работе. Микрокатегорий: %d", len(microcategories))
    yield
    logger.info("Сервис остановлен")


app = FastAPI(
    title="Avito Service Splitter",
    description="Автоматическое выделение самостоятельных услуг и генерация черновиков",
    version="1.0.0",
    lifespan=lifespan,
)

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(_FRONTEND_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        microcategoriesLoaded=len(_state.get("microcategories", [])),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    matcher: Matcher = _state.get("matcher")
    mc_lookup: dict = _state.get("mc_lookup", {})

    if not matcher:
        raise HTTPException(status_code=503, detail="Сервис ещё не готов")

    result = await pipeline.analyze(
        item_id=request.itemId,
        mc_id=request.mcId,
        mc_title=request.mcTitle,
        description=request.description,
        matcher=matcher,
        mc_lookup=mc_lookup,
    )

    return AnalyzeResponse(
        itemId=result["itemId"],
        detectedMcIds=result["detectedMcIds"],
        shouldSplit=result["shouldSplit"],
        drafts=[DraftItem(**d) for d in result["drafts"]],
        reasoning=result["reasoning"],
        debug=DebugInfo(**result["debug"]),
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    matcher: Matcher = _state.get("matcher")
    mc_lookup: dict = _state.get("mc_lookup", {})

    if not matcher:
        raise HTTPException(status_code=503, detail="Сервис ещё не готов")

    dataset = load_dataset()
    items = dataset[: request.limit]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    split_correct = 0
    details = []

    for item in items:
        result = await pipeline.analyze(
            item_id=item["itemId"],
            mc_id=item["sourceMcId"],
            mc_title=item["sourceMcTitle"],
            description=item["description"],
            matcher=matcher,
            mc_lookup=mc_lookup,
        )

        predicted_ids = set(result["debug"]["independentMcIds"])
        actual_ids = set(item["targetSplitMcIds"])

        # Исключаем исходную микрокатегорию
        predicted_ids.discard(item["sourceMcId"])
        actual_ids.discard(item["sourceMcId"])

        tp = len(predicted_ids & actual_ids)
        fp = len(predicted_ids - actual_ids)
        fn = len(actual_ids - predicted_ids)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if result["shouldSplit"] == item["shouldSplit"]:
            split_correct += 1

        details.append(
            EvaluateDetail(
                itemId=str(item["itemId"]),
                predicted_independentMcIds=sorted(predicted_ids),
                actual_splitMcIds=sorted(actual_ids),
                predicted_shouldSplit=result["shouldSplit"],
                actual_shouldSplit=item["shouldSplit"],
                tp=tp,
                fp=fp,
                fn=fn,
            )
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    should_split_accuracy = split_correct / len(items) if items else 0.0

    logger.info(
        "Оценка завершена: P=%.3f R=%.3f F1=%.3f Acc=%.3f (n=%d)",
        precision, recall, f1, should_split_accuracy, len(items),
    )

    return EvaluateResponse(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        shouldSplitAccuracy=round(should_split_accuracy, 4),
        totalItems=len(items),
        details=details,
    )
