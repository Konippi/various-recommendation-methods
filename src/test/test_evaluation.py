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
        self, true_user_id2movie_ids: dict, predicted_user_id2movie_ids: dict, k: int, expected: float
    ):
        actual = self.__evaluation.calc_recall_at_k(true_user_id2movie_ids, predicted_user_id2movie_ids, k)
        assert np.isclose(actual, expected, atol=1e-3, rtol=1e-3)
