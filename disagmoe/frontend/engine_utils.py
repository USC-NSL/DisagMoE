import enum
from disagmoe.config import EngineConfig

class EngineType(enum.Enum):
    ATTENTION = enum.auto()
    EXPERT = enum.auto()
    HYBRID = enum.auto()
    
global_engine_config: EngineConfig = None

def set_global_engine_config(engine_config: EngineConfig):
    global global_engine_config
    global_engine_config = engine_config
    
def get_global_engine_config() -> EngineConfig:
    global global_engine_config
    return global_engine_config