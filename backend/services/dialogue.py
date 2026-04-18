from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.intent import IntentClassifier


@dataclass
class DialogueState:
    active_flow: str
    required_slots: list[str]
    filled_slots: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DialogueResult:
    response: str
    intent: str
    confidence: float = 0.99


class DialogueStateManager:
    def __init__(self) -> None:
        self._states: dict[str, DialogueState] = {}

    def handle_active_flow(
        self,
        session_id: str,
        message: str,
        intent_classifier: IntentClassifier,
    ) -> DialogueResult | None:
        state = self._states.get(session_id)
        if state is None:
            return None

        if state.active_flow == "order_status":
            return self._continue_order_status(session_id, message, intent_classifier)

        if state.active_flow == "product_issue":
            return self._continue_product_issue(session_id, message, intent_classifier)

        if state.active_flow == "order_issue_clarification":
            return self._continue_order_issue_clarification(
                session_id,
                message,
                intent_classifier,
            )

        self.clear(session_id)
        return None

    def start_order_status(
        self,
        session_id: str,
        intent_classifier: IntentClassifier,
    ) -> DialogueResult:
        self._states[session_id] = DialogueState(
            active_flow="order_status",
            required_slots=["order_id"],
        )
        return DialogueResult(
            response=self._response(
                intent_classifier,
                "ask_order_id",
                "Sure, please share your order ID.",
            ),
            intent="order_status",
        )

    def start_product_issue(
        self,
        session_id: str,
        message: str,
        intent_classifier: IntentClassifier,
    ) -> DialogueResult:
        order_id = intent_classifier.extract_order_id(message)
        state = DialogueState(
            active_flow="product_issue",
            required_slots=["order_id", "issue_description"],
            filled_slots={"issue_type": self._detect_issue_type(message)},
        )

        if order_id:
            state.filled_slots["order_id"] = order_id

        self._states[session_id] = state

        if "order_id" not in state.filled_slots:
            return DialogueResult(
                response=self._response(
                    intent_classifier,
                    "ask_order_id",
                    "Please share your order ID so I can create the request.",
                ),
                intent="wrong_or_expired_product",
            )

        return DialogueResult(
            response=self._response(
                intent_classifier,
                "ask_product_issue_details",
                "Thanks. Please upload a photo or describe what was wrong with the product.",
            ),
            intent="wrong_or_expired_product",
        )

    def start_order_issue_clarification(
        self,
        session_id: str,
        intent_classifier: IntentClassifier,
    ) -> DialogueResult:
        self._states[session_id] = DialogueState(
            active_flow="order_issue_clarification",
            required_slots=["order_issue_choice"],
        )
        return DialogueResult(
            response=self._response(
                intent_classifier,
                "clarify_order_issue",
                "Do you want to track, cancel, return, or report a problem with your order?",
            ),
            intent="clarify_order_issue",
            confidence=0.6,
        )

    def clear(self, session_id: str) -> None:
        self._states.pop(session_id, None)

    def _continue_order_status(
        self,
        session_id: str,
        message: str,
        intent_classifier: IntentClassifier,
    ) -> DialogueResult:
        order_id = intent_classifier.extract_order_id(message)
        if order_id is None:
            return DialogueResult(
                response=self._response(
                    intent_classifier,
                    "ask_order_id",
                    "Please share a valid order ID so I can check the status.",
                ),
                intent="order_status",
                confidence=0.85,
            )

        self.clear(session_id)
        return DialogueResult(
            response=intent_classifier.get_response("order_status", order_id),
            intent="order_status",
        )

    def _continue_product_issue(
        self,
        session_id: str,
        message: str,
        intent_classifier: IntentClassifier,
    ) -> DialogueResult:
        state = self._states[session_id]
        order_id = intent_classifier.extract_order_id(message)

        if "order_id" not in state.filled_slots:
            if order_id is None:
                return DialogueResult(
                    response=self._response(
                        intent_classifier,
                        "ask_order_id",
                        "Please share your order ID so I can create the request.",
                    ),
                    intent="wrong_or_expired_product",
                    confidence=0.85,
                )

            state.filled_slots["order_id"] = order_id
            return DialogueResult(
                response=self._response(
                    intent_classifier,
                    "ask_product_issue_details",
                    "Thanks. Please upload a photo or describe what was wrong with the product.",
                ),
                intent="wrong_or_expired_product",
            )

        if not intent_classifier.is_order_id(message):
            state.filled_slots["issue_description"] = message.strip()

        if "issue_description" not in state.filled_slots:
            return DialogueResult(
                response=self._response(
                    intent_classifier,
                    "ask_product_issue_details",
                    "Thanks. Please upload a photo or describe what was wrong with the product.",
                ),
                intent="wrong_or_expired_product",
                confidence=0.9,
            )

        response = self._response(
            intent_classifier,
            "product_issue_ticket_created",
            "Thanks. I have created a replacement or refund request for order {order_id}. Our team will review the details and contact you soon.",
        ).format(**state.filled_slots)
        self.clear(session_id)
        return DialogueResult(response=response, intent="wrong_or_expired_product")

    def _continue_order_issue_clarification(
        self,
        session_id: str,
        message: str,
        intent_classifier: IntentClassifier,
    ) -> DialogueResult:
        normalized = message.lower()

        if any(word in normalized for word in ("track", "status", "where", "shipping")):
            return self.start_order_status(session_id, intent_classifier)

        self.clear(session_id)
        if "cancel" in normalized:
            return DialogueResult(
                response=intent_classifier.get_response("cancel_order", message),
                intent="cancel_order",
            )

        if any(word in normalized for word in ("return", "refund", "exchange")):
            return DialogueResult(
                response=intent_classifier.responses.get(
                    "faq_return_a_product",
                    "Go to your Orders page, choose the item, and select Return or Replace if the product is still within the return window.",
                ),
                intent="faq_return_a_product",
            )

        if any(
            word in normalized
            for word in ("report", "wrong", "expired", "damaged", "missing", "problem")
        ):
            return self.start_product_issue(session_id, message, intent_classifier)

        self._states[session_id] = DialogueState(
            active_flow="order_issue_clarification",
            required_slots=["order_issue_choice"],
        )
        return DialogueResult(
            response=self._response(
                intent_classifier,
                "clarify_order_issue",
                "Do you want to track, cancel, return, or report a problem with your order?",
            ),
            intent="clarify_order_issue",
            confidence=0.5,
        )

    @staticmethod
    def _detect_issue_type(message: str) -> str:
        normalized = message.lower()
        if "expired" in normalized or "expiry" in normalized:
            return "expired_product"
        if "wrong" in normalized or "different" in normalized:
            return "wrong_product"
        if "damaged" in normalized or "broken" in normalized:
            return "damaged_product"
        if "missing" in normalized:
            return "missing_item"
        return "product_issue"

    @staticmethod
    def _response(
        intent_classifier: IntentClassifier,
        key: str,
        default: str,
    ) -> str:
        responses: Any = intent_classifier.responses
        if isinstance(responses, dict):
            return str(responses.get(key, default))
        return default
