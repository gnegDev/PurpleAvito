import json
import ast
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def load_microcategories() -> list[dict]:
    """Загружает справочник микрокатегорий из CSV."""
    path = DATASETS_DIR / "rnc_mic_key_phrases.csv"
    df = pd.read_csv(path, sep=",")
    categories = []
    for _, row in df.iterrows():
        raw_phrases = str(row["keyPhrases"]) if pd.notna(row["keyPhrases"]) else ""
        phrases = [p.strip() for p in raw_phrases.split(";") if p.strip()]
        categories.append(
            {
                "mcId": int(row["mcId"]),
                "mcTitle": str(row["mcTitle"]),
                "keyPhrases": phrases,
                "description": str(row["description"]) if pd.notna(row.get("description")) else "",
            }
        )
    logger.info("Загружено %d микрокатегорий из %s", len(categories), path)
    return categories


def load_dataset() -> list[dict]:
    """Загружает датасет с эталонной разметкой."""
    path = DATASETS_DIR / "rnc_dataset_markup.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result = []
    for item in data:
        def _parse_ids(val):
            if not val or val in ("[]", "", None):
                return []
            if isinstance(val, list):
                return [int(x) for x in val]
            try:
                parsed = ast.literal_eval(str(val))
                return [int(x) for x in parsed] if parsed else []
            except Exception:
                return []

        result.append(
            {
                "itemId": str(item["itemId"]),
                "sourceMcId": int(item["sourceMcId"]),
                "sourceMcTitle": str(item["sourceMcTitle"]),
                "description": str(item["description"]),
                "targetDetectedMcIds": _parse_ids(item.get("targetDetectedMcIds")),
                "targetSplitMcIds": _parse_ids(item.get("targetSplitMcIds")),
                "shouldSplit": bool(item.get("shouldSplit", False)),
                "caseType": str(item.get("caseType", "")),
            }
        )
    logger.info("Загружено %d записей датасета из %s", len(result), path)
    return result
