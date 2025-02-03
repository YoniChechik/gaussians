import time
from datetime import datetime


def now_str(is_human_readable: bool = False) -> str:
    if is_human_readable:
        pattern = "%Y-%m-%d %H:%M:%S"
    else:
        pattern = "%Y%m%d_%H%M%S"
    return datetime.now().strftime(pattern)


def dt_str(dt_sec: float):
    SECONDS_IN_HOUR = 3600
    SECONDES_IN_MINUTE = 60
    hours, residual_secondes = divmod(dt_sec, SECONDS_IN_HOUR)
    minutes, seconds = divmod(residual_secondes, SECONDES_IN_MINUTE)
    return f"{hours:.0f}hrs, {minutes:.0f}mins, {seconds:.2f}secs"


class Timer:
    def __enter__(self):
        self._start_time_sec = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.dt_sec: float = time.perf_counter() - self._start_time_sec
