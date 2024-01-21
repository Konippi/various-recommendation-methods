from abc import abstractmethod
from typing import Optional

import pandas as pd

from src.dataset import Dataset
from src.utils.path import Path


class BaseRecommend:
    def __init__(self, dataset_dir: str, user_nums: Optional[int], test_size: float = 0.3) -> None:
        self.path = Path()
        self.dataset = Dataset(self.path.get_local_path(dataset_dir), user_nums)
        self.user_nums = user_nums
        self.test_size = test_size

    def get_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        movies, ratings = self.dataset.load()
        train, test = self.dataset.split_ratings(ratings, self.test_size)
        return train, test

    def output(self, **kwargs: float) -> None:
        print(kwargs)

    @abstractmethod
    def run(self) -> None:
        pass
