from pathlib import Path

from parametric import BaseParams


class DefaultParams(BaseParams):
    save_dir_path: Path | None = None
    git_sha: str | None = None
    is_multiprocess: bool = True
    is_debug: bool = False
    is_reproducible: bool = True  # whether to use a constant seed for all random number generators
