"""The `logger` routine for writing a log string to stdout."""
import datetime
from contextlib import nullcontext, AbstractContextManager
from typing import Final, Callable

#: the "now" function
__DTN: Final[Callable[[], datetime.datetime]] = datetime.datetime.now


def logger(message: str,
           note: str = "",
           lock: AbstractContextManager = nullcontext()) -> None:
    """
    Write a message to the log.

    :param message: the message
    :param note: a note to put between the time and the message
    :param lock: the lock to prevent multiple threads to write log
        output at the same time
    """
    text: Final[str] = f"{__DTN()}{note}: {message}"
    with lock:
        print(text, flush=True)
