import unittest

import numpy as np
import pandas as pd

from src.evaluate.ensemble_xlmr_knn import blend_scores, predictions_from_scores, select_alpha
from src.evaluate.knn_baseline import KNNConfig, score_knn


class EnsembleXlmrKnnTests(unittest.TestCase):
    def test_knn_scores_are_probability_distributions(self):
        train = pd.DataFrame({"titulo": ["Python guide", "French cooking"], "cdd": ["005.13", "641.5"]})
        query = pd.DataFrame({"titulo": ["Python tutorial", "French recipes"], "cdd": ["005.13", "641.5"]})
        scores = score_knn(train, query, KNNConfig("word_char", 1), ["005.13", "641.5"])
        self.assertTrue(np.allclose(scores.sum(axis=1), 1.0))
        self.assertEqual(scores.argmax(axis=1).tolist(), [0, 1])

    def test_blend_and_selection_choose_best_validation_weight(self):
        val = pd.DataFrame({"titulo": ["A", "B"], "cdd": ["001", "002"]})
        labels = ["001", "002"]
        xlmr = np.array([[0.9, 0.1], [0.2, 0.8]])
        knn = np.array([[0.2, 0.8], [0.8, 0.2]])
        self.assertTrue(np.allclose(blend_scores(xlmr, knn, 1.0), xlmr))
        alpha, table = select_alpha(val, labels, xlmr, knn, [0.0, 0.5, 1.0])
        self.assertEqual(alpha, 1.0)
        prediction = predictions_from_scores(val, labels, xlmr, knn, alpha)
        self.assertEqual(prediction["cdd_pred"].tolist(), ["001", "002"])
        self.assertEqual(table.iloc[0]["full_accuracy"], 1.0)
