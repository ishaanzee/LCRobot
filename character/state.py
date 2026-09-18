from enum import Enum

from character.audio import CharacterAudio


class CharacterState(Enum):
    IDLE = "idle"
    ENGAGED = "engaged"
    LISTENING = "listening"
    OBSERVING = "observing"
    THINKING = "thinking"
    RESPONDING = "responding"
    ACTING = "acting"


class CharacterStateMachine:
    def __init__(self, audio=None):
        self.current = CharacterState.IDLE
        self.audio = audio or CharacterAudio()

    def transition_to(self, new_state):
        if isinstance(new_state, str):
            new_state = CharacterState(new_state)
        if new_state == self.current:
            return False

        previous = self.current
        self.current = new_state
        print(f"Character state: {previous.value} -> {new_state.value}")
        self.audio.play_transition(previous.value, new_state.value)
        return True
