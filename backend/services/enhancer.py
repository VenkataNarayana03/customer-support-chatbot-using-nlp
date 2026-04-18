from __future__ import annotations

import os

from services.context import ChatTurn


class ResponseEnhancer:
    def __init__(self) -> None:
        self._generator = None
        self._load_model()

    def enhance(self, user_message: str, answer: str, history: list[ChatTurn]) -> str:
        if self._generator is not None:
            try:
                prompt = self._build_prompt(user_message, answer, history)
                generated = self._generator(
                    prompt,
                    max_length=96,
                    num_beams=4,
                    do_sample=False,
                )[0]["generated_text"]
                cleaned = generated.strip()
                if cleaned:
                    return cleaned
            except Exception:
                pass

        return self._polish(answer)

    def _load_model(self) -> None:
        use_t5 = os.getenv("CHATBOT_USE_T5") == "1"
        allow_downloads = os.getenv("CHATBOT_ALLOW_MODEL_DOWNLOADS") == "1"
        if not use_t5:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(
                "t5-small",
                local_files_only=not allow_downloads,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "t5-small",
                local_files_only=not allow_downloads,
            )
            self._generator = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        except Exception:
            self._generator = None

    @staticmethod
    def _build_prompt(user_message: str, answer: str, history: list[ChatTurn]) -> str:
        history_text = " ".join(
            f"Customer: {turn.user} Assistant: {turn.bot}" for turn in history[-2:]
        )
        return (
            "Rewrite this support answer to be clear, polite, and concise. "
            f"Conversation: {history_text} "
            f"Customer: {user_message} "
            f"Answer: {answer}"
        )

    @staticmethod
    def _polish(answer: str) -> str:
        cleaned = answer.strip()
        if not cleaned:
            return "I can help with that. Could you share a little more detail?"
        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned
