import json
import logging
import os
import re

import openai

logger = logging.getLogger(__name__)

_CLASSIFY_PROMPT_ID = "fvti1hi7d9a0u505o33v"
_DRAFT_PROMPT_ID = "fvt5q3qemedkvtd6i4c2"
_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
_PROJECT = "b1gnpsrg9bte58p5mgf1"


def _get_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=_BASE_URL,
        default_headers={"x-folder-id": _PROJECT},
    )


def _extract_text(response) -> str:
    """Универсально извлекает текст из ответа OpenAI Responses API."""
    if hasattr(response, "output_text"):
        return response.output_text
    if hasattr(response, "output"):
        for block in response.output:
            if hasattr(block, "content"):
                for part in block.content:
                    if hasattr(part, "text"):
                        return part.text
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    return str(response)


def _parse_json(text: str) -> dict:
    """Парсит JSON из текста, убирая возможный markdown-код."""
    # Убираем markdown ```json ... ```
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Ищем первый JSON-объект
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def classify(
    description: str,
    mc_id: int,
    mc_title: str,
    candidates: list[dict],
) -> dict:
    """
    Шаг 2: LLM классификация кандидатов.
    Возвращает dict с ключами: detectedMcIds, shouldSplit, independentMcIds, reasoning.
    """
    candidates_json = json.dumps(
        [{"mcId": c["mcId"], "mcTitle": c["mcTitle"], "matchedPhrases": c["matchedPhrases"]} for c in candidates],
        ensure_ascii=False,
        indent=2,
    )
    user_message = (
        f"Текст объявления: {description}\n"
        f"Исходная микрокатегория: {mc_title} (mcId: {mc_id})\n\n"
        f"Кандидаты на микрокатегории:\n{candidates_json}\n\n"
        "Верни JSON без markdown и пояснений:\n"
        "{\n"
        '  "detectedMcIds": [список mcId услуг упомянутых в тексте],\n'
        '  "shouldSplit": true/false,\n'
        '  "independentMcIds": [список mcId только самостоятельных услуг],\n'
        '  "reasoning": "краткое обоснование"\n'
        "}"
    )

    logger.info("Вызываю LLM классификацию для mcId=%d, кандидатов=%d", mc_id, len(candidates))

    client = _get_client()
    response = client.responses.create(
        prompt={"id": _CLASSIFY_PROMPT_ID},
        input=user_message,
    )
    raw_text = _extract_text(response)
    logger.info("LLM классификация получена, длина ответа=%d", len(raw_text))

    result = _parse_json(raw_text)
    return {
        "detectedMcIds": [int(x) for x in result.get("detectedMcIds", [])],
        "shouldSplit": bool(result.get("shouldSplit", False)),
        "independentMcIds": [int(x) for x in result.get("independentMcIds", [])],
        "reasoning": str(result.get("reasoning", "")),
    }


def generate_drafts(
    description: str,
    mc_title: str,
    independent_mc_ids: list[int],
    mc_lookup: dict[int, dict],
) -> list[dict]:
    """
    Шаг 3: генерация черновиков объявлений для каждого independentMcId.
    Возвращает список dict с ключами: mcId, mcTitle, text.
    """
    targets = [
        {"mcId": mc_id, "mcTitle": mc_lookup[mc_id]["mcTitle"]}
        for mc_id in independent_mc_ids
        if mc_id in mc_lookup
    ]
    if not targets:
        return []

    targets_json = json.dumps(targets, ensure_ascii=False, indent=2)
    user_message = (
        f"Исходное объявление: {description}\n"
        f"Исходная микрокатегория: {mc_title}\n\n"
        f"Создай черновики для микрокатегорий:\n{targets_json}\n\n"
        "Верни JSON без markdown и пояснений:\n"
        "{\n"
        '  "drafts": [\n'
        '    {"mcId": 101, "mcTitle": "Название", "text": "текст 50-100 слов"}\n'
        "  ]\n"
        "}"
    )

    logger.info("Вызываю LLM генерацию черновиков для %d микрокатегорий", len(targets))

    client = _get_client()
    response = client.responses.create(
        prompt={"id": _DRAFT_PROMPT_ID},
        input=user_message,
    )
    raw_text = _extract_text(response)
    logger.info("LLM черновики получены, длина ответа=%d", len(raw_text))

    result = _parse_json(raw_text)
    drafts = result.get("drafts", [])
    return [
        {
            "mcId": int(d["mcId"]),
            "mcTitle": str(d["mcTitle"]),
            "text": str(d["text"]),
        }
        for d in drafts
    ]
