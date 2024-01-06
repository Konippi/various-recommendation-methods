import os
import sys

sys.path.insert(0, os.getcwd())

from src.recommend.base_recommend import BaseRecommend


class Random(BaseRecommend):
    def __init__(self, user_nums: int | None = None, test_size: float = 0.3):
        super().__init__("dataset/movielens-10m", user_nums, test_size)

    def __recommend(self, dataset, **kwargs):
        pass

    def run(self):
        train, test = super().get_dataset()
