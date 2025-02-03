import inspect
import os
import sys
from enum import IntEnum
from importlib.metadata import distributions
from multiprocessing import set_start_method
from pathlib import Path
from typing import Callable

from loguru import logger
from parametric import Override

from pipe_utils.def_params import DefaultParams
from utils.random import seed_everything
from utils.time import now_str


def script_runner(params: DefaultParams, specific_runner_script: Callable | None = None):
    """a wrapper that performs all required preparations for running the main functionality as a script.

    parameters:
        params: the parameters Scheme of the running script
    """

    def decorate(func):
        is_main = _check_if_caller_is_main()
        if not is_main:
            # func is already called from a decorated script - avoid additional decoration
            return func

        def wrapper(*args, **kwargs):
            # === for multiprocessing (if enabled)
            # https://github.com/isl-org/Open3D/issues/4923
            # NOTE this call should be the first function before anything else is initialized so that the multiprocess
            # start method will affect all new object (e.g. loguru)
            set_start_method("spawn")

            # === parse params
            dev_override_params_path = Path("override_params.yaml")
            if dev_override_params_path.is_file():
                params.override_from_yaml_path(dev_override_params_path)
            _parse_params_from_envs(params)

            now = now_str()
            # ==== save_dir_path
            if params.save_dir_path is None:
                func_name = func.__name__
                relative_dir_structure = Path(f"{func_name}_res") / now
                if params.git_sha is not None:
                    save_dir_path = Path("/dl_db", relative_dir_structure)
                else:
                    save_dir_path = Path("/tmp", relative_dir_structure)

                save_dir_path.mkdir(parents=True, exist_ok=True)
                with Override():
                    params.save_dir_path = save_dir_path
            else:
                logger.opt(colors=True).info(f"<magenta>Running from existing save_dir_path: {params.save_dir_path}</>")

            # ==== set logger level
            if params.is_debug:
                logger_level = "DEBUG"
            else:
                logger_level = "INFO"

            # set console level
            logger.remove()
            logger.add(sys.stderr, level=logger_level)

            # create log dir
            curr_logs_dir = params.save_dir_path / f"logs/{now}"
            curr_logs_dir.mkdir(parents=True, exist_ok=True)
            # enqueue=True for process safe writing
            logger.add(curr_logs_dir / "log.log", enqueue=True, level=logger_level)
            # add logger files for warnings and errors
            logger.add(curr_logs_dir / "problems.log", enqueue=True, level="WARNING")

            # ======= avoid randomness due to random number generators state
            if params.is_reproducible:
                seed_everything()

            # ==== run specific
            if specific_runner_script is not None:
                specific_runner_script()

            # === write params
            params.save_yaml(curr_logs_dir / "params.yaml")
            # === write packages
            with open(curr_logs_dir / "pip_freeze.txt", "w") as f:
                f.write(_get_pip_freeze_str())

            logger.opt(colors=True).info(f"<magenta>save dir path: {params.save_dir_path}</>")

            # ======= execution
            if params.git_sha is None:
                # not inside a production docker -> debug mode
                func(*args, **kwargs)
            else:
                # inside a production/ci-cd docker -> production mode -> log exception and exit
                exit_code = _ExitCode.SUCCESS
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logger.exception(e)
                    logger.error("Exception occurred, exiting...")
                    exit_code = _ExitCode.FAILURE

                logger.info(f"End; Save dir path: {params.save_dir_path}")
                exit(exit_code)  # Zero exit code indicates success

        return wrapper

    return decorate


class _ExitCode(IntEnum):
    SUCCESS = 0
    FAILURE = 1


def _get_pip_freeze_str() -> str:
    installed_packages = distributions()
    return "\n".join(f"{dist.metadata['Name']}=={dist.version}" for dist in installed_packages)


def _check_if_caller_is_main(stack_caller_idx: int = 2) -> bool:
    # Get the current call stack
    stack = inspect.stack()

    # Check if the stack is large enough to have a caller
    if len(stack) > 1:
        # Get the caller's frame. since this is an nested auxiliary function, we bactrack 2 stacks instead of 1
        caller_frame = stack[stack_caller_idx].frame

        # Get the module name of the caller
        caller_module = inspect.getmodule(caller_frame)

        # If the module is None, check based on the filename instead
        if caller_module is None and caller_frame.f_globals["__name__"] == "__main__":
            return True
        elif caller_module and caller_module.__name__ == "__main__":
            return True
        else:
            return False
    else:
        # No caller (or it's not possible to determine), assume not __main__
        return False


def _parse_params_from_envs(params: DefaultParams):
    # Filter out keys that start with the specified prefix and remove the prefix from the key
    env_prefix = "_param_"

    to_override = {}
    for key, value in os.environ.items():
        if not key.startswith(env_prefix):
            continue
        to_override[key[len(env_prefix) :]] = value

    params.override_from_dict(to_override)
