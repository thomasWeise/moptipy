"""The `logger` routine for writing a log string to stdout."""
import datetime
from contextlib import nullcontext
from typing import Final, ContextManager


def logger(message: str,
           note: str = "",
           lock: ContextManager = nullcontext()) -> None:
    """
    Write a message to the log.

    :param message: the message
    :param note: a note to put between the time and the message
    :param lock: the lock to prevent multiple threads to write log
        output at the same time
    """
    text: Final[str] = f"{datetime.datetime.now()}{note}: {message}"
    with lock:
        print(text, flush=True)
