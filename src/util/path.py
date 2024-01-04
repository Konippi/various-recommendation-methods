import os
from typing import TypeAlias

Path: TypeAlias = str


def get_root_dir() -> Path:
    """
    Get the root directory of the project.

    Parameters
    ----------
        None

    Returns
    -------
        root_dir: Path
            The root directory of the project
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
    return root_dir + os.path.sep
