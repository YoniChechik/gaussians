import functools
from typing import Callable

from pipe.params import params
from utils.multiprocess import run_iters
from utils.random import seed_everything


def run_iters_pipe(
    func: Callable,
    iters_args: list[tuple],
    max_parallelism: int | None = None,
) -> list:
    child_wrapped_func = _child_process_setup_wrapper(func)
    params_as_dict = params.model_dump_serializable()
    iters_args = [(args, params_as_dict, params.is_reproducible) for args in iters_args]

    return run_iters(
        child_wrapped_func,
        iters_args,
        max_parallelism=max_parallelism,
        is_multiprocess=params.is_multiprocess,
    )


class _child_process_setup_wrapper:
    """
    why we wrap a function like this? because pickle problem on multiprocess:
    https://stackoverflow.com/a/27642041/4879610
    """

    def __init__(self, func: Callable):
        self.func = func

        functools.update_wrapper(self, func)

    def __call__(self, *args):
        original_args, params_as_dict, is_reproducible = args
        if is_reproducible:
            seed_everything()

        params.override_from_dict(params_as_dict)

        return self.func(*original_args)
