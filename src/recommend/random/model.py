import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))))

from src.recommend.util.dataset_loader import DatasetLoader
from src.recommend.util.path import Path

# class Random:
#     def recommend(self, dataset, **kwargs):
#         unique_user_ids = sorted(dataset.train.user_id.unique())

#     def run(self):
#         self.recommend(dataset)


if __name__ == "__main__":
    path = Path()
    dataset_loader = DatasetLoader(path.get_local_path("dataset/movielens-10m/"))
    dataset_loader.load()
    # model = Random()
    # model.run()
