from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


@dataclass(frozen=True)
class RetrievalResult:
    question: str
    answer: str
    score: float


class FAQRetriever:
    def __init__(self) -> None:
        self.faqs = self._load_faqs()
        self._model = None
        self._embeddings = None
        self._load_model()

    def search(self, query: str) -> RetrievalResult:
        if not self.faqs:
            return RetrievalResult(
                question="",
                answer="I do not have FAQ data loaded yet. Please add entries to backend/data/faq.json.",
                score=0.0,
            )

        if self._model is not None and self._embeddings is not None:
            try:
                query_embedding = self._model.encode([query], normalize_embeddings=True)[0]
                scores = self._embeddings @ query_embedding
                best_index = int(scores.argmax())
                best = self.faqs[best_index]
                return RetrievalResult(
                    question=best["question"],
                    answer=best["answer"],
                    score=float(scores[best_index]),
                )
            except Exception:
                pass

        best = max(
            self.faqs,
            key=lambda item: self._keyword_score(query, item["search_text"]),
        )
        return RetrievalResult(
            question=best["question"],
            answer=best["answer"],
            score=self._keyword_score(query, best["search_text"]),
        )

    def _load_model(self) -> None:
        use_minilm = os.getenv("CHATBOT_USE_MINILM") == "1"
        allow_downloads = os.getenv("CHATBOT_ALLOW_MODEL_DOWNLOADS") == "1"
        if not use_minilm:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                local_files_only=not allow_downloads,
            )
            searchable_texts = [item["search_text"] for item in self.faqs]
            self._embeddings = self._model.encode(
                searchable_texts,
                normalize_embeddings=True,
            )
        except BaseException:
            self._model = None
            self._embeddings = None

    def _load_faqs(self) -> list[dict[str, str]]:
        data = self._load_json(DATA_DIR / "faq.json", [])
        if isinstance(data, dict):
            data = data.get("faqs", [])

        faqs: list[dict[str, str]] = []
        for item in data:
            question = item.get("question")
            answer = item.get("answer")
            if question and answer:
                faqs.append(
                    {
                        "question": str(question),
                        "answer": str(answer),
                        "search_text": f"{question} {answer}",
                    }
                )

        return faqs

    @staticmethod
    def _keyword_score(query: str, text: str) -> float:
        query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
        text_words = set(re.findall(r"[a-z0-9]+", text.lower()))
        if not query_words or not text_words:
            return 0.0

        intersection = len(query_words & text_words)
        denominator = math.sqrt(len(query_words) * len(text_words))
        return intersection / denominator

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default

        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
