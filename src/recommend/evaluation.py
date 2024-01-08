import numpy as np
from sklearn.metrics import mean_squared_error as mse


class Evaluation:
    def __init__(self):
        pass

    def calc_recall_at_k(
        self,
        true_user_id2movie_ids: dict[int, list[int]],
        predicted_user_id2movie_ids: dict[int, list[int]],
        k: int = 10,
    ) -> float:
        """
        Calculate recall@k

        Parameters
        ----------
        true_user2movies : dict[int, list[int]]
            combination of user id and truly favorite movies
        predicted_user2movies : dict[int, list[int]]
            combination of user id and predicted favorite movies

        Returns
        -------
        mean_scores : float
            average of recall@k for all users
        """
        scores = [
            self.__recall_at_k(true_user_id2movie_ids[user_id], predicted_user_id2movie_ids[user_id], k)
            for user_id in true_user_id2movie_ids.keys()
        ]

        return float(np.mean(scores))

    def __recall_at_k(self, true_movie_ids: list[int], predicted_movie_ids: list[int], k: int) -> float:
        """
        Calculate recall@k

        Parameters
        ----------
        true_movie_ids : list[int]
            list of truly favorite movie ids
        predicted_movie_ids : list[int]
            list of predicted favorite movie ids

        Returns
        -------
        score: float
            recall@k
        """
        # if true_movie_ids is empty or k is 0, impossible to calculate recall@k
        if len(true_movie_ids) == 0 or k == 0:
            return 0.0

        return len(set(true_movie_ids) & set(predicted_movie_ids[:k])) / len(true_movie_ids)

    def calc_precision_at_k(
        self,
        true_user_id2movie_ids: dict[int, list[int]],
        predicted_user_id2movie_ids: dict[int, list[int]],
        k: int = 10,
    ) -> float:
        """
        Calculate precision@k

        Parameters
        ----------
        true_user2movies : dict[int, list[int]]
            combination of user id and truly favorite movies
        predicted_user2movies : dict[int, list[int]]
            combination of user id and predicted favorite movies

        Returns
        -------
        mean_scores: float
            average of precision@k for all users
        """
        scores = [
            self.__precision_at_k(true_user_id2movie_ids[user_id], predicted_user_id2movie_ids[user_id], k)
            for user_id in true_user_id2movie_ids.keys()
        ]

        return float(np.mean(scores))

    def __precision_at_k(self, true_movie_ids: list[int], predicted_movie_ids: list[int], k: int) -> float:
        """
        Calculate precision@k

        Parameters
        ----------
        true_movie_ids : list[int]
            list of truly favorite movie ids
        predicted_movie_ids : list[int]
            list of predicted favorite movie ids

        Returns
        -------
        score: float
            precision@k
        """
        # if k is 0, impossible to calculate precision@k
        if k == 0:
            return 0.0

        return len(set(true_movie_ids) & set(predicted_movie_ids[:k])) / k

    def calc_rmse(self, true_ratings: list[float], predicted_ratings: list[float]) -> float:
        """
        Calculate RMSE

        Parameters
        ----------
        true_ratings : list[float]
            list of truly rating
        predicted_ratings : list[float]
            list of predicted rating

        Returns
        -------
        score: float
            RMSE
        """
        return np.sqrt(mse(true_ratings, predicted_ratings))
