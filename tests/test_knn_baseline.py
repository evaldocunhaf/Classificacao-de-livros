import unittest

import pandas as pd

from src.evaluate.knn_baseline import KNNConfig, predict_knn, select_on_validation


class KNNBaselineTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "titulo": ["Python for beginners", "French cooking recipes", "Advanced Python"],
                "cdd": ["005.13", "641.5", "005.13"],
            }
        )
        self.val = pd.DataFrame(
            {"titulo": ["Advanced Python guide", "French cuisine"], "cdd": ["005.13", "641.5"]}
        )

    def test_knn_predicts_from_titles_only(self):
        predictions = predict_knn(self.train, self.val, KNNConfig(feature="word_char", k=1))
        self.assertEqual(predictions["cdd_pred"].tolist(), ["005.13", "641.5"])
        self.assertTrue((predictions["method"] == "knn_tfidf_word_char").all())

    def test_selection_returns_the_first_ranked_validation_configuration(self):
        configs = [KNNConfig(feature="word", k=1), KNNConfig(feature="word_char", k=1)]
        selected, table = select_on_validation(self.train, self.val, configs)
        self.assertEqual(selected, KNNConfig(feature=table.iloc[0]["feature"], k=int(table.iloc[0]["k"])))
        self.assertEqual(len(table), len(configs))
        self.assertTrue(table["full_accuracy"].is_monotonic_decreasing)
