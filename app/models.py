from pydantic import BaseModel
from typing import Optional


class AnalyzeRequest(BaseModel):
    itemId: int
    mcId: int
    mcTitle: str
    description: str


class DraftItem(BaseModel):
    mcId: int
    mcTitle: str
    text: str


class DebugInfo(BaseModel):
    candidatesFound: int
    independentMcIds: list[int]


class AnalyzeResponse(BaseModel):
    itemId: int
    detectedMcIds: list[int]
    shouldSplit: bool
    drafts: list[DraftItem]
    reasoning: str
    debug: DebugInfo


class EvaluateRequest(BaseModel):
    limit: int = 50


class EvaluateDetail(BaseModel):
    itemId: str
    predicted_independentMcIds: list[int]
    actual_splitMcIds: list[int]
    predicted_shouldSplit: bool
    actual_shouldSplit: bool
    tp: int
    fp: int
    fn: int


class EvaluateResponse(BaseModel):
    precision: float
    recall: float
    f1: float
    shouldSplitAccuracy: float
    totalItems: int
    details: list[EvaluateDetail]


class HealthResponse(BaseModel):
    status: str
    microcategoriesLoaded: int


class CandidateMicrocategory(BaseModel):
    mcId: int
    mcTitle: str
    matchedPhrases: list[str]
    keywordScore: float
    embeddingScore: float
