from enum import Enum


class CharacterState(Enum):
    IDLE = "idle"
    ENGAGED = "engaged"
    LISTENING = "listening"
    OBSERVING = "observing"
    THINKING = "thinking"
    RESPONDING = "responding"
    ACTING = "acting"
