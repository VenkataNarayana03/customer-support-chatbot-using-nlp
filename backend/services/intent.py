from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "bert_model"


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float


class IntentClassifier:
    def __init__(self) -> None:
        self.responses = self._load_json(DATA_DIR / "responses.json", {})
        self.intent_patterns = self._load_intent_patterns()
        self._classifier = None
        self._labels = self._load_labels()
        self._load_model()

    def predict(self, text: str) -> IntentResult:
        if self.is_order_id(text):
            return IntentResult(intent="order_status", confidence=0.95)

        pattern_result = self._predict_from_patterns(text)
        if pattern_result.confidence >= 0.9:
            return pattern_result

        if self._classifier is not None:
            try:
                prediction = self._classifier(text, truncation=True)[0]
                predicted_intent = str(prediction["label"])
                confidence = float(prediction["score"])
                if (
                    predicted_intent.startswith("faq_")
                    and pattern_result.intent != predicted_intent
                    and pattern_result.confidence < 0.75
                ):
                    return pattern_result

                return IntentResult(
                    intent=predicted_intent,
                    confidence=confidence,
                )
            except Exception:
                pass

        return pattern_result

    def _predict_from_patterns(self, text: str) -> IntentResult:
        normalized = text.lower()
        best_intent = "faq"
        best_score = 0.0

        for intent, patterns in self.intent_patterns.items():
            score = self._score_patterns(normalized, patterns)
            if score > best_score:
                best_intent = intent
                best_score = score

        if best_score == 0.0:
            return IntentResult(intent="faq", confidence=0.55)

        return IntentResult(intent=best_intent, confidence=min(0.95, best_score))

    def get_response(self, intent: str, message: str = "") -> str:
        if intent == "order_status":
            order_id = self.extract_order_id(message)
            if order_id:
                template = self.responses.get(
                    "order_status_tracking",
                    "Order {message} is being processed and should be updated soon.",
                )
                return template.format(message=order_id)

            return self.responses.get(
                "order_status_prompt",
                "Sure, I can help with that. Please share your order ID.",
            )

        return self.responses.get(
            intent,
            self.responses.get(
                "fallback",
                "I can help with that. Could you share a little more detail?",
            ),
        )

    def _load_model(self) -> None:
        if not (MODEL_DIR / "config.json").exists():
            return

        try:
            from transformers import pipeline

            self._classifier = pipeline(
                "text-classification",
                model=str(MODEL_DIR),
                tokenizer=str(MODEL_DIR),
            )
        except Exception:
            self._classifier = None

    def _load_labels(self) -> list[str]:
        config_path = MODEL_DIR / "config.json"
        config = self._load_json(config_path, {})
        id2label = config.get("id2label", {})
        if isinstance(id2label, dict):
            return [id2label[key] for key in sorted(id2label, key=lambda item: int(item))]
        return []

    def _load_intent_patterns(self) -> dict[str, list[str]]:
        data = self._load_json(DATA_DIR / "intents.json", {})
        patterns: dict[str, list[str]] = {}

        if isinstance(data, dict) and isinstance(data.get("intents"), list):
            for item in data["intents"]:
                tag = item.get("tag")
                samples = item.get("patterns", [])
                if tag and isinstance(samples, list):
                    patterns[tag] = [str(sample) for sample in samples]

        return patterns

    @staticmethod
    def _score_patterns(text: str, patterns: list[str]) -> float:
        text_words = set(re.findall(r"[a-z0-9]+", text))
        best_score = 0.0

        for pattern in patterns:
            pattern_words = set(re.findall(r"[a-z0-9]+", pattern.lower()))
            if not pattern_words:
                continue
            overlap = len(text_words & pattern_words) / len(pattern_words)
            substring_bonus = 0.25 if pattern.lower() in text else 0.0
            best_score = max(best_score, overlap + substring_bonus)

        return best_score

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def is_order_id(text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False

        return bool(
            re.fullmatch(r"(?:order\s*)?#?[A-Z0-9][A-Z0-9-]{2,24}", normalized, re.IGNORECASE)
            and re.search(r"\d", normalized)
        )

    @staticmethod
    def has_order_id(text: str) -> bool:
        return IntentClassifier.extract_order_id(text) is not None

    @staticmethod
    def extract_order_id(text: str) -> str | None:
        match = re.search(
            r"\b(?:order\s*)?(#?[A-Z0-9-]*\d[A-Z0-9-]*)\b",
            text,
            re.IGNORECASE,
        )
        if match is None:
            return None

        return match.group(1)
