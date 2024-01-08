import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pytest

from src.recommend.evaluation import Evaluation


class Test_Evaluation:
    __evaluation = Evaluation()

    @pytest.mark.parametrize(
        [
            "true_user_id2movie_ids",
            "predicted_user_id2movie_ids",
            "k",
            "expected",
        ],
        [
            pytest.param(
                {1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]},
                {1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]},
                3,
                1.000,
            ),
            pytest.param(
                {1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]},
                {1: [1, 2, 3, 4], 2: [1, 2, 3], 3: [1, 2, 3, 4, 5]},
                3,
                1.000,
            ),
            pytest.param(
                {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4]},
                {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 4, 5]},
                4,
                0.916,
            ),
        ],
    )
    def test_calc_recall_at_k(
        self,
        true_user_id2movie_ids: dict[int, list[int]],
        predicted_user_id2movie_ids: dict[int, list[int]],
        k: int,
        expected: float,
    ):
        actual = self.__evaluation.calc_recall_at_k(true_user_id2movie_ids, predicted_user_id2movie_ids, k)
        assert np.isclose(actual, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        [
            "true_user_id2movie_ids",
            "predicted_user_id2movie_ids",
            "k",
            "expected",
        ],
        [
            pytest.param(
                {1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]},
                {1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]},
                3,
                1.000,
            ),
            pytest.param(
                {1: [1, 2, 3], 2: [1, 2, 3], 3: [1, 2, 3]},
                {1: [1, 2, 3, 4], 2: [1, 2, 3], 3: [1, 2, 3, 4, 5]},
                3,
                1.000,
            ),
            pytest.param(
                {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4]},
                {1: [1, 2, 3, 4], 2: [1, 2, 3, 4], 3: [1, 2, 4, 5]},
                4,
                0.916,
            ),
        ],
    )
    def test_calc_precision_at_k(
        self,
        true_user_id2movie_ids: dict[int, list[int]],
        predicted_user_id2movie_ids: dict[int, list[int]],
        k: int,
        expected: float,
    ):
        actual = self.__evaluation.calc_precision_at_k(true_user_id2movie_ids, predicted_user_id2movie_ids, k)
        assert np.isclose(actual, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        [
            "true_ratings",
            "predicted_ratings",
            "expected",
        ],
        [
            pytest.param(
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                0.000,
            ),
            pytest.param(
                [1.1, 1.2, 1.3, 1.4, 1.5],
                [1.6, 1.7, 1.8, 1.9, 2.0],
                0.500,
            ),
            pytest.param(
                [0.5, 1.5, 2.5, 3.5],
                [1.5, 2.5, 3.5, 4.5],
                1.000,
            ),
        ],
    )
    def test_calc_rmse(self, true_ratings: list[float], predicted_ratings: list[float], expected: float):
        actual = self.__evaluation.calc_rmse(true_ratings, predicted_ratings)
        assert np.isclose(actual, expected, atol=1e-3, rtol=1e-3)
