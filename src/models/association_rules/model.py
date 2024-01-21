from collections import Counter, defaultdict

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from src.base_recommend import BaseRecommend
from src.evaluation import Evaluation


class AssociationRules(BaseRecommend):
    __USER_NUMS: int = 1000
    __TEST_SIZE: float = 0.2
    __EVALUATE_MIN_RATING: float = 4.0
    __APRIORI_SUPPORT_THRESHOLD: float = 0.05
    __APRIORI_LIFT_THRESHOLD: float = 1.0
    __INPUT_MAX_MOVIE_NUMS: int = 5
    __RECOMMEND_MOVIE_NUMS: int = 10

    __evaluation = Evaluation()

    def __init__(self) -> None:
        super().__init__(dataset_dir="dataset/movielens-10m", user_nums=self.__USER_NUMS, test_size=self.__TEST_SIZE)

    def __recommend(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, list[int]]]:
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
        # create user-movie matrix(value=rating)
        user_movie_matrix = train.pivot(index="user_id", columns="movie_id", values="rating")

        # convert ratings to 0 and 1 based on threshold for using apriori algorithm
        # apriori algorithm: https://docs.oracle.com/cd/E16338_01/datamine.112/e48231/algo_apriori.htm
        user_movie_matrix[user_movie_matrix < self.__EVALUATE_MIN_RATING] = 0.0
        user_movie_matrix[user_movie_matrix.isnull()] = 0.0
        user_movie_matrix[user_movie_matrix >= self.__EVALUATE_MIN_RATING] = 1.0

        # extract movies that have high support by using apriori algorithm
        frequent_movies = apriori(
            df=user_movie_matrix.astype("bool"),
            min_support=self.__APRIORI_SUPPORT_THRESHOLD,
            use_colnames=True,
        )

        # association rules
        rules = association_rules(
            df=frequent_movies,
            metric="lift",
            min_threshold=self.__APRIORI_LIFT_THRESHOLD,
        )

        # list of recommended movies for each user
        user_id2recommended_movie_ids = defaultdict(list)

        user_id2evaluated_movie_ids = train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()

        train_filtered_high_rating = train[train.rating >= self.__EVALUATE_MIN_RATING]
        for user_id, movies in train_filtered_high_rating.groupby("user_id"):
            newest_rated_movie_ids = movies.sort_values("timestamp")["movie_id"].to_list()[
                -self.__INPUT_MAX_MOVIE_NUMS :
            ]

            # extract association rules that contain at least one in antecedents
            matched_rules = rules.antecedents.apply(lambda x: len(set(newest_rated_movie_ids) & x)) >= 1

            consequent_movie_ids = []
            for _, row in rules[matched_rules].sort_values("lift", ascending=False).iterrows():
                consequent_movie_ids.extend(row.consequents)

            counter = Counter(consequent_movie_ids)
            for movie_id, _ in counter.most_common():
                # recommend movies that have not been previously rated
                if movie_id not in user_id2evaluated_movie_ids[user_id]:
                    user_id2recommended_movie_ids[user_id].append(movie_id)

                if len(user_id2recommended_movie_ids[user_id]) == self.__RECOMMEND_MOVIE_NUMS:
                    break

        # since the RMSE is not calculated, return the ratings of test as predicted ratings
        return test.rating, dict(user_id2recommended_movie_ids)

    def __evaluate(
        self, test: pd.DataFrame, predicted_ratings: list[float], user_id2recommended_movie_ids: dict[int, list[int]]
    ) -> tuple[float, float]:
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
        recall_at_k : float
            Recall@k
        precision_at_k : float
            Precision@k
        """
        test_user_id2movie_ids = (
            test[test.rating >= self.__EVALUATE_MIN_RATING]
            .groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )

        # recall@k
        recall_at_k = self.__evaluation.calc_recall_at_k(
            test_user_id2movie_ids,
            user_id2recommended_movie_ids,
            self.__RECOMMEND_MOVIE_NUMS,
        )

        # precision@k
        precision_at_k = self.__evaluation.calc_precision_at_k(
            test_user_id2movie_ids,
            user_id2recommended_movie_ids,
            self.__RECOMMEND_MOVIE_NUMS,
        )

        return recall_at_k, precision_at_k

    def run(self) -> None:
        """
        Run the recommendation model
        """
        train, test = super().get_dataset()
        predicted_ratings, user_id2recommended_movie_ids = self.__recommend(train, test)

        # RMSE is not calculated due to the difficulty in predicting the evaluation value
        recall_at_k, precision_at_k = self.__evaluate(test, list(predicted_ratings), user_id2recommended_movie_ids)
        super().output(recall_at_k=recall_at_k, precision_at_k=precision_at_k)
