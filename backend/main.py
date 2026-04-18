from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.context import ConversationMemory
from services.dialogue import DialogueStateManager
from services.enhancer import ResponseEnhancer
from services.intent import IntentClassifier
from services.retrieval import FAQRetriever


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    source: str


app = FastAPI(title="Customer Support Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

intent_classifier = IntentClassifier()
faq_retriever = FAQRetriever()
response_enhancer = ResponseEnhancer()
conversation_memory = ConversationMemory()
dialogue_state = DialogueStateManager()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    user_message = payload.message.strip()
    history = conversation_memory.get_history(payload.session_id)

    dialogue_result = dialogue_state.handle_active_flow(
        payload.session_id,
        user_message,
        intent_classifier,
    )
    if dialogue_result is not None:
        return _finalize_response(
            session_id=payload.session_id,
            user_message=user_message,
            base_response=dialogue_result.response,
            intent=dialogue_result.intent,
            confidence=dialogue_result.confidence,
            source="dialogue_state",
            history=history,
        )

    intent_result = intent_classifier.predict(user_message)

    if (
        intent_classifier.is_order_id(user_message)
        and history
        and "order id" in history[-1].bot.lower()
    ):
        intent_result = type(intent_result)(intent="order_status", confidence=0.99)

    if _is_ambiguous_order_issue(user_message, intent_result.confidence):
        dialogue_result = dialogue_state.start_order_issue_clarification(
            payload.session_id,
            intent_classifier,
        )
        return _finalize_response(
            session_id=payload.session_id,
            user_message=user_message,
            base_response=dialogue_result.response,
            intent=dialogue_result.intent,
            confidence=dialogue_result.confidence,
            source="confidence_routing",
            history=history,
        )

    if _needs_low_confidence_clarification(intent_result.intent, intent_result.confidence):
        return _finalize_response(
            session_id=payload.session_id,
            user_message=user_message,
            base_response=intent_classifier.responses.get(
                "low_confidence_clarification",
                "I want to make sure I help correctly. Is this about an order, payment, refund, return, account, or product issue?",
            ),
            intent="clarification",
            confidence=intent_result.confidence,
            source="confidence_routing",
            history=history,
        )

    if intent_result.intent == "order_status" and not intent_classifier.has_order_id(user_message):
        dialogue_result = dialogue_state.start_order_status(
            payload.session_id,
            intent_classifier,
        )
        return _finalize_response(
            session_id=payload.session_id,
            user_message=user_message,
            base_response=dialogue_result.response,
            intent=dialogue_result.intent,
            confidence=dialogue_result.confidence,
            source="dialogue_state",
            history=history,
        )

    if intent_result.intent == "wrong_or_expired_product":
        dialogue_result = dialogue_state.start_product_issue(
            payload.session_id,
            user_message,
            intent_classifier,
        )
        return _finalize_response(
            session_id=payload.session_id,
            user_message=user_message,
            base_response=dialogue_result.response,
            intent=dialogue_result.intent,
            confidence=dialogue_result.confidence,
            source="dialogue_state",
            history=history,
        )

    if intent_result.intent == "faq":
        retrieval_result = faq_retriever.search(user_message)
        base_response = retrieval_result.answer
        source = "faq_retrieval"
    else:
        base_response = intent_classifier.get_response(intent_result.intent, user_message)
        source = "fixed_response"

    return _finalize_response(
        session_id=payload.session_id,
        user_message=user_message,
        base_response=base_response,
        intent=intent_result.intent,
        confidence=intent_result.confidence,
        source=source,
        history=history,
    )


def _finalize_response(
    session_id: str,
    user_message: str,
    base_response: str,
    intent: str,
    confidence: float,
    source: str,
    history: list,
) -> ChatResponse:
    final_response = response_enhancer.enhance(
        user_message=user_message,
        answer=base_response,
        history=history,
    )

    conversation_memory.add_exchange(
        session_id=session_id,
        user_message=user_message,
        bot_message=final_response,
    )

    return ChatResponse(
        response=final_response,
        intent=intent,
        confidence=confidence,
        source=source,
    )


def _is_ambiguous_order_issue(message: str, confidence: float) -> bool:
    normalized = message.lower()
    if confidence >= 0.75:
        return False
    return "order" in normalized and any(
        word in normalized
        for word in ("issue", "problem", "help", "concern", "trouble")
    )


def _needs_low_confidence_clarification(intent: str, confidence: float) -> bool:
    if intent == "faq":
        return False
    return confidence < 0.45


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
