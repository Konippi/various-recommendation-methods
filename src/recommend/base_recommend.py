import os
import sys

sys.path.insert(0, os.getcwd())

from abc import abstractclassmethod
from typing import Tuple

import pandas as pd

from src.recommend.util.dataset import Dataset
from src.recommend.util.path import Path


class BaseRecommend:
    def __init__(self, dataset_dir: str, user_nums: int | None = None, test_size: float = 0.3):
        self.path = Path()
        self.dataset = Dataset(self.path.get_local_path(dataset_dir), user_nums)
        self.user_nums = user_nums
        self.test_size = test_size

    def get_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        movies, ratings = self.dataset.load()
        train, test = self.dataset.split_ratings(ratings, self.test_size)
        return train, test

    @abstractclassmethod
    def run(self):
        pass
