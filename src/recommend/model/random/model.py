import os
import sys

sys.path.insert(0, os.getcwd())

from collections import defaultdict

import numpy as np

from src.recommend.base_recommend import BaseRecommend


class Random(BaseRecommend):
    __MOVIELENS_MIN_RATING = 0.5
    __MOVIELENS_MAX_RATING = 5.0
    __RECOMMENDATION_NUM = 10

    def __init__(self, user_nums: int | None = None, test_size: float = 0.3):
        super().__init__("dataset/movielens-10m", user_nums, test_size)

    def __recommend(self, train, test):
        # sorted unique user and movie ids
        unique_user_ids = sorted(train.user_id.unique())
        unique_movie_ids = sorted(train.movie_id.unique())

        # associate user and movie ids with index
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        # create prediction matrix (predicted ratings: random value between 0.5 and 5.0)
        prediction_matrix = np.random.uniform(
            self.__MOVIELENS_MIN_RATING, self.__MOVIELENS_MAX_RATING, (len(unique_user_ids), len(unique_movie_ids))
        )

        expected_results = test.copy()
        predicted_results = []

        for i, row in test.iterrows():
            user_id = row["user_id"]
            movie_id = row["movie_id"]

            # if movie_id is not in the training dataset, predict random value
            if movie_id not in movie_id2index:
                predicted_results.append(np.random.uniform(self.__MOVIELENS_MIN_RATING, self.__MOVIELENS_MAX_RATING))
                continue

            user_index = user_id2index[user_id]
            movie_index = movie_id2index[movie_id]

            predicted_score = prediction_matrix[user_index, movie_index]

            predicted_results.append(predicted_score)

        expected_results["predicted_ratings"] = predicted_results

        # list of recommended movies for each user
        user2recommended_movies = defaultdict(list)

        user_evaluated_movies = train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for user_id in unique_user_ids:
            user_index = user_id2index[user_id]
            movie_indexes = np.argsort(-prediction_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = unique_movie_ids[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    user2recommended_movies[user_id].append(movie_id)
                if len(user2recommended_movies[user_id]) == self.__RECOMMENDATION_NUM:
                    break

        print(expected_results.predicted_ratings, user2recommended_movies)

    def run(self):
        train, test = super().get_dataset()
        self.__recommend(train, test)
