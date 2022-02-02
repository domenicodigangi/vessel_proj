import functools

def catch_all_and_log(f, logger=None):
    """
    A function wrapper for catching all exceptions and logging them
    """
    @functools.wraps(f)
    def inner(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        try:
            return f(*args, **kwargs)
        except Exception as ex:
            if logger is None:
                print(ex)
            else:
                logger.error(ex)

    return inner


