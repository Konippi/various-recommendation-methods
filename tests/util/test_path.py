import os

from src.utils.path import Path


class TestPath:
    __path = Path()

    def test_get_root_dir(self) -> None:
        ROOT_DIR_NAME = "various_recommendation_methods"
        assert self.__path.get_root_dir().endswith(ROOT_DIR_NAME + os.path.sep)
