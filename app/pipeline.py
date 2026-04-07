import asyncio
import logging
from functools import partial

from app import llm, matching

logger = logging.getLogger(__name__)


async def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args))


async def analyze(
    item_id: int,
    mc_id: int,
    mc_title: str,
    description: str,
    matcher: matching.Matcher,
    mc_lookup: dict,
) -> dict:
    """
    Полный пайплайн: шаг 1 (кандидаты) → шаг 2 (LLM классификация) → шаг 3 (черновики).
    """
    # --- Шаг 1: поиск кандидатов ---
    candidates = await run_in_executor(matcher.find_candidates, description, mc_id)
    logger.info("[itemId=%s] Шаг 1 завершён: кандидатов=%d", item_id, len(candidates))

    if not candidates:
        logger.info("[itemId=%s] Кандидатов нет, пропускаем LLM", item_id)
        return {
            "itemId": item_id,
            "detectedMcIds": [],
            "shouldSplit": False,
            "drafts": [],
            "reasoning": "Кандидаты не найдены на шаге keyword/embedding matching",
            "debug": {"candidatesFound": 0, "independentMcIds": []},
        }

    # --- Шаг 2: LLM классификация ---
    try:
        classify_result = await run_in_executor(
            llm.classify, description, mc_id, mc_title, candidates
        )
        logger.info(
            "[itemId=%s] Шаг 2 завершён: shouldSplit=%s, independentMcIds=%s",
            item_id,
            classify_result["shouldSplit"],
            classify_result["independentMcIds"],
        )
    except Exception as exc:
        logger.error("[itemId=%s] LLM классификация недоступна: %s", item_id, exc)
        # Fallback: возвращаем результат шага 1 с shouldSplit=false
        detected = [c["mcId"] for c in candidates]
        return {
            "itemId": item_id,
            "detectedMcIds": detected,
            "shouldSplit": False,
            "drafts": [],
            "reasoning": f"LLM недоступна: {exc}",
            "debug": {"candidatesFound": len(candidates), "independentMcIds": []},
        }

    detected_mc_ids = classify_result["detectedMcIds"]
    should_split = classify_result["shouldSplit"]
    independent_mc_ids = classify_result["independentMcIds"]
    reasoning = classify_result["reasoning"]

    # --- Шаг 3: генерация черновиков (только если shouldSplit=true) ---
    drafts = []
    if should_split and independent_mc_ids:
        try:
            drafts = await run_in_executor(
                llm.generate_drafts, description, mc_title, independent_mc_ids, mc_lookup
            )
            logger.info("[itemId=%s] Шаг 3 завершён: черновиков=%d", item_id, len(drafts))
        except Exception as exc:
            logger.error("[itemId=%s] LLM генерация черновиков недоступна: %s", item_id, exc)

    return {
        "itemId": item_id,
        "detectedMcIds": detected_mc_ids,
        "shouldSplit": should_split,
        "drafts": drafts,
        "reasoning": reasoning,
        "debug": {
            "candidatesFound": len(candidates),
            "independentMcIds": independent_mc_ids,
        },
    }
