import os

from src.recommend.util.path import Path


class TestPath:
    __path = Path()

    def test_get_root_dir(self):
        ROOT_DIR_NAME = "various-recommendation-methods"
        assert self.__path.get_root().endswith(ROOT_DIR_NAME + os.path.sep)
