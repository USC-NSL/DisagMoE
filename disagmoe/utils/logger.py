import logging

from time import time
from logging import getLogger, Formatter, StreamHandler

class SimFormatter(Formatter):
    
    def __init__(self, fmt: str | None = None, datefmt: str | None = None, style = "%", validate: bool = True, *, defaults = None) -> None:
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self._launch_time = time()
    
    def format(self, record):
        log_message = super().format(record)
        return f"{(time() - self._launch_time):.4f} - {log_message}"


_formatter = SimFormatter(f"[%(levelname)s] [%(filename)s:%(lineno)d] <%(name)s>: %(message)s")
# _formatter = Formatter(f"{sim_clock()} - [%(levelname)s]: %(message)s")
_handler = StreamHandler()
_handler.setFormatter(_formatter)


def get_logger(name, level=logging.INFO):
    logger = getLogger(name)
    logger.setLevel(level)
    logger.addHandler(_handler)
    return logger