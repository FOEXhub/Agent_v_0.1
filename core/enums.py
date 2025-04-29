from enum import Enum, auto

class AgentState(Enum):
    INIT = auto()
    REQUIREMENTS_WRITTEN = auto()
    REQUIREMENTS_APPROVED = auto()
    CODE_WRITTEN = auto()
    CODE_APPROVED = auto()
    FINISHED = auto()
    ERROR = auto()