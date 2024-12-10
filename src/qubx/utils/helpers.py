from functools import wraps
from threading import Lock


def synchronized(func):
    """Decorator that ensures only one thread can execute the decorated function at a time."""
    lock = Lock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)

    return wrapper
