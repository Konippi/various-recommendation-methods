import os

from src.recommend.util import path


def test_get_root_dir():
    ROOT_DIR_NAME = "various-recommendation-methods"
    assert path.get_root_dir().endswith(ROOT_DIR_NAME + os.path.sep)
