from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTurn:
    user: str
    bot: str


class ConversationMemory:
    def __init__(self, max_turns: int = 6) -> None:
        self._max_turns = max_turns
        self._sessions: dict[str, deque[ChatTurn]] = defaultdict(
            lambda: deque(maxlen=self._max_turns)
        )

    def add_exchange(self, session_id: str, user_message: str, bot_message: str) -> None:
        self._sessions[session_id].append(ChatTurn(user=user_message, bot=bot_message))

    def get_history(self, session_id: str) -> list[ChatTurn]:
        return list(self._sessions[session_id])

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
