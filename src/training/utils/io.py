import time
from functools import wraps


def use_backoff(max_retries=10, initial_delay=1, backoff_factor=2):
    """
    Universal retry wrapper with exponential backoff for any function, but primarily for loading
    and storing HuggingFace datasets and objects.

    Example usage:

    >>> @use_backoff(max_retries=10, delay=1, backoff_factor=2)
    >>> def important_io_operation(x):
    >>>     return x + 1

    Args:
        fn: Function to execute
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1)
        backoff_factor: Multiplier for delay between retries (default: 2)

    Returns:
        A wrapper function that will retry the function fn up to max_retries times with exponential backoff

    Raises:
        Exception: If all retries fail
    """

    def _decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            current_delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        time.sleep(current_delay)
                        current_delay *= backoff_factor

            raise Exception(
                f"IO Operation failed after {max_retries} attempts: {str(last_exception)}"
            )

        return wrapper

    return _decorator
