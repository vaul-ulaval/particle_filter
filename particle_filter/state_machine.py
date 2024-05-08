from enum import Enum, auto


class StateMachine:
    def __init__(self, initial_state, transitions):
        self.state = initial_state
        self.transitions = transitions
        self.on_enter_state()

    def trigger(self, event):
        if event in self.transitions[self.state]:
            self.state = self.transitions[self.state][event]
            self.on_enter_state()

    def on_enter_state(self):
        print(f"Entering state {self.state}")

class State(Enum):
    WAITING_INIT = auto()
    UNCERTAIN = auto()
    LOCALIZE = auto()
    STOPPED = auto()
    RECOVERY = auto()
    FAILED_TO_LOCALIZE = auto()
    FALLBACK = auto()

class Event(Enum):
    START_LOCALIZATION = auto()
    STOP_LOCALIZATION = auto()
    INITIALIZE_GLOBAL = auto()
    INITIALIZE_MANUAL = auto()
    ERROR_ENCOUNTERED = auto()
    STOP_RECORDING_POSE = auto()
    ENABLE_RECOVERY = auto()
    RECOVERED = auto()
    LOCALIZATION_FAILED = auto()
    BEHIND_VEHICULE_RESPAWN = auto()
    LOCALIZED = auto()
    OFF = auto()
    ON = auto()