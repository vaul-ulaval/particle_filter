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
    RECOVERED = auto()
    LOCALIZATION_FAILED = auto()
    BEHIND_VEHICULE_RESPAWN = auto()
    LOCALIZED = auto()
    OFF = auto()
    ON = auto()

transitions = {
    State.FALLBACK: {
        Event.ON: State.WAITING_INIT
    },
    State.WAITING_INIT: {
        Event.START_LOCALIZATION: State.UNCERTAIN,
        Event.INITIALIZE_GLOBAL: State.UNCERTAIN,
        Event.INITIALIZE_MANUAL: State.LOCALIZE,
        Event.OFF: State.FALLBACK
    },
    State.UNCERTAIN: {
        Event.LOCALIZED: State.LOCALIZE,
        Event.INITIALIZE_MANUAL: State.LOCALIZE,
        Event.STOP_LOCALIZATION: State.WAITING_INIT,
        Event.ERROR_ENCOUNTERED: State.RECOVERY,
        Event.OFF: State.FALLBACK
    },
    State.LOCALIZE: {
        Event.STOP_LOCALIZATION: State.STOPPED,
        Event.ERROR_ENCOUNTERED: State.RECOVERY,
        Event.LOCALIZATION_FAILED: State.FAILED_TO_LOCALIZE,
        Event.OFF: State.FALLBACK
    },
    State.STOPPED: {
        Event.START_LOCALIZATION: State.LOCALIZE,
        Event.ERROR_ENCOUNTERED: State.RECOVERY,
        Event.LOCALIZATION_FAILED: State.FAILED_TO_LOCALIZE,
        Event.OFF: State.FALLBACK
    },
    State.RECOVERY: {
        Event.RECOVERED: State.LOCALIZE,
        Event.LOCALIZATION_FAILED: State.FAILED_TO_LOCALIZE,
        Event.INITIALIZE_GLOBAL: State.UNCERTAIN,
        Event.INITIALIZE_MANUAL: State.LOCALIZE,
        Event.BEHIND_VEHICULE_RESPAWN: State.LOCALIZE,
        Event.OFF: State.FALLBACK
    },
    State.FAILED_TO_LOCALIZE: {
        Event.RECOVERED: State.LOCALIZE,
        Event.INITIALIZE_GLOBAL: State.UNCERTAIN,
        Event.INITIALIZE_MANUAL: State.LOCALIZE,
        Event.BEHIND_VEHICULE_RESPAWN: State.LOCALIZE,
        Event.OFF: State.FALLBACK
    }
}

# Scenario normal
state_machine = StateMachine(State.FALLBACK, transitions)
state_machine.trigger(Event.ON)
state_machine.trigger(Event.INITIALIZE_MANUAL)
state_machine.trigger(Event.STOP_LOCALIZATION)
state_machine.trigger(Event.START_LOCALIZATION)
state_machine.trigger(Event.OFF)