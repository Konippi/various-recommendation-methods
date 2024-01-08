import os
import sys

sys.path.insert(0, os.getcwd())

from collections import defaultdict

import numpy as np
import pandas as pd

from src.recommend.base_recommend import BaseRecommend
from src.recommend.evaluation import Evaluation


class Random(BaseRecommend):
    __USER_NUMS: int = 1000
    __TEST_SIZE: float = 0.2
    __MOVIELENS_MIN_RATING: float = 0.5
    __MOVIELENS_MAX_RATING: float = 5.0
    __RECOMMEND_MOVIES_NUM: int = 10
    __EVALUATE_MIN_RATING: float = 4.0

    __evaluation = Evaluation()

    def __init__(self):
        super().__init__("dataset/movielens-10m", self.__USER_NUMS, self.__TEST_SIZE)

    def __recommend(self, train, test) -> tuple[pd.DataFrame, dict[int, list[int]]]:
        """
        Recommend movies for each user

        Parameters
        ----------
        train : pd.DataFrame
            Training dataset
        test : pd.DataFrame
            Test dataset

        Returns
        -------
        predicted_ratings : list[float]
            list of predicted ratings
        user_id2recommended_movie_ids : dict[int, list[int]]
            combination of user id and recommended movie ids
        """
        # sorted unique user and movie ids
        sorted_unique_user_ids = sorted(train.user_id.unique())
        sorted_unique_movie_ids = sorted(train.movie_id.unique())

        # associate user and movie ids with index
        user_id2index = dict(zip(sorted_unique_user_ids, range(len(sorted_unique_user_ids))))
        movie_id2index = dict(zip(sorted_unique_movie_ids, range(len(sorted_unique_movie_ids))))

        # create prediction matrix (predicted ratings: random value between 0.5 and 5.0)
        prediction_matrix = np.random.uniform(
            self.__MOVIELENS_MIN_RATING,
            self.__MOVIELENS_MAX_RATING,
            (len(sorted_unique_user_ids), len(sorted_unique_movie_ids)),
        )

        expected_results = test.copy()
        predicted_ratings = []

        for i, row in test.iterrows():
            user_id = row["user_id"]
            movie_id = row["movie_id"]

            # if movie_id is not in the training dataset, predict random value
            if movie_id not in movie_id2index:
                predicted_ratings.append(np.random.uniform(self.__MOVIELENS_MIN_RATING, self.__MOVIELENS_MAX_RATING))
                continue

            user_index = user_id2index[user_id]
            movie_index = movie_id2index[movie_id]

            predicted_score = prediction_matrix[user_index, movie_index]

            predicted_ratings.append(predicted_score)

        expected_results["predicted_ratings"] = predicted_ratings

        # list of recommended movies for each user
        user_id2recommended_movie_ids = defaultdict(list)

        user_id2evaluated_movie_ids = train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        for user_id in sorted_unique_user_ids:
            user_index = user_id2index[user_id]

            # sort movies by predicted ratings
            movie_indexes = np.argsort(-prediction_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = sorted_unique_movie_ids[movie_index]
                if movie_id not in user_id2evaluated_movie_ids[user_id]:
                    user_id2recommended_movie_ids[user_id].append(movie_id)
                if len(user_id2recommended_movie_ids[user_id]) == self.__RECOMMEND_MOVIES_NUM:
                    break

        return expected_results.predicted_ratings, dict(user_id2recommended_movie_ids)

    def __evaluate(
        self, test: pd.DataFrame, predicted_ratings: list[float], user_id2recommended_movie_ids: dict[int, list[int]]
    ) -> tuple[float, float, float]:
        """
        Evaluate the recommendation model

        Parameters
        ----------
        test : pd.DataFrame
            Test dataset
        predicted_ratings : list[float]
            list of predicted ratings
        user_id2recommended_movie_ids : dict[int, list[int]]
            combination of user id and recommended movie ids

        Returns
        -------
        rmse : float
            RMSE
        recall_at_k : float
            Recall@k
        precision_at_k : float
            Precision@k
        """
        test_user_id2movie_ids = (
            test[test.rating > self.__EVALUATE_MIN_RATING]
            .groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        # rmse
        rmse = self.__evaluation.calc_rmse(list(test.rating), predicted_ratings)

        # recall@k
        recall_at_k = self.__evaluation.calc_recall_at_k(
            test_user_id2movie_ids, user_id2recommended_movie_ids, self.__RECOMMEND_MOVIES_NUM
        )

        # precision@k
        precision_at_k = self.__evaluation.calc_precision_at_k(
            test_user_id2movie_ids, user_id2recommended_movie_ids, self.__RECOMMEND_MOVIES_NUM
        )

        return rmse, recall_at_k, precision_at_k

    def run(self):
        """
        Run the recommendation model
        """
        train, test = super().get_dataset()
        predicted_ratings, user_id2recommended_movie_ids = self.__recommend(train, test)
        rmse, recall_at_k, precision_at_k = self.__evaluate(
            test, list(predicted_ratings), user_id2recommended_movie_ids
        )
        super().output(rmse=rmse, recall_at_k=recall_at_k, precision_at_k=precision_at_k)
