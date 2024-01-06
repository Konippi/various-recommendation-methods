import os
import sys

sys.path.insert(0, os.getcwd())

from src.recommend.util.dataset import Dataset
from src.recommend.util.path import Path

# class Random:
#     def recommend(self, dataset, **kwargs):
#         unique_user_ids = sorted(dataset.train.user_id.unique())

#     def run(self):
#         self.recommend(dataset)


if __name__ == "__main__":
    path = Path()
    dataset_loader = Dataset(path.get_local_path("dataset/movielens-10m/"), 1000)
    dataset_loader.load()
    # model = Random()
    # model.run()
