import logging
import re

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Пороги
KEYWORD_THRESHOLD = 0  # любое совпадение ключевой фразы
EMBEDDING_THRESHOLD = 0.38
TOP_K = 10


def _normalize(text: str) -> str:
    return text.lower()


def keyword_match(description: str, key_phrases: list[str]) -> list[str]:
    """Возвращает список совпавших ключевых фраз."""
    desc_lower = _normalize(description)
    matched = []
    for phrase in key_phrases:
        if phrase and re.search(re.escape(_normalize(phrase)), desc_lower):
            matched.append(phrase)
    return matched


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class Matcher:
    def __init__(self, microcategories: list[dict], model: SentenceTransformer):
        self.microcategories = microcategories
        self.model = model
        self._mc_embeddings: dict[int, np.ndarray] = {}
        self._precompute_embeddings()

    def _mc_text(self, mc: dict) -> str:
        """Текст микрокатегории для embedding."""
        phrases = "; ".join(mc["keyPhrases"][:30])  # берём первые 30 фраз
        return f"{mc['mcTitle']}: {phrases}"

    def _precompute_embeddings(self):
        logger.info("Вычисляю embeddings для %d микрокатегорий...", len(self.microcategories))
        texts = [self._mc_text(mc) for mc in self.microcategories]
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        for mc, emb in zip(self.microcategories, embeddings):
            self._mc_embeddings[mc["mcId"]] = emb
        logger.info("Embeddings готовы")

    def find_candidates(
        self, description: str, source_mc_id: int
    ) -> list[dict]:
        """
        Шаг 1: находит кандидатов через keyword matching + embedding similarity.
        Исключает исходную микрокатегорию. Возвращает топ-10 кандидатов.
        """
        desc_embedding = self.model.encode(description, show_progress_bar=False, convert_to_numpy=True)

        scored = []
        for mc in self.microcategories:
            if mc["mcId"] == source_mc_id:
                continue

            matched_phrases = keyword_match(description, mc["keyPhrases"])
            keyword_score = len(matched_phrases) / max(len(mc["keyPhrases"]), 1)

            emb_score = cosine_similarity(desc_embedding, self._mc_embeddings[mc["mcId"]])

            if len(matched_phrases) > KEYWORD_THRESHOLD or emb_score > EMBEDDING_THRESHOLD:
                scored.append(
                    {
                        "mcId": mc["mcId"],
                        "mcTitle": mc["mcTitle"],
                        "matchedPhrases": matched_phrases,
                        "keywordScore": round(keyword_score, 4),
                        "embeddingScore": round(emb_score, 4),
                    }
                )

        # Сортируем: сначала по keyword, потом по embedding
        scored.sort(key=lambda x: (x["keywordScore"], x["embeddingScore"]), reverse=True)
        top = scored[:TOP_K]

        logger.info(
            "Найдено %d кандидатов для source_mc_id=%d (топ-%d возвращено)",
            len(scored),
            source_mc_id,
            len(top),
        )
        return top
