import unittest

import pandas as pd

from src.evaluate.logreg_baseline import fit_predict, prediction_frame, select_c


class LogRegBaselineTests(unittest.TestCase):
    def test_logreg_predicts_and_selects_from_validation_only(self):
        train = pd.DataFrame(
            {
                "texto": ["python language", "python code", "french cooking", "food recipes"],
                "cdd": ["005.13", "005.13", "641.5", "641.5"],
                "titulo": ["a", "b", "c", "d"],
            }
        )
        val = pd.DataFrame(
            {
                "texto": ["python tutorial", "french recipes"],
                "cdd": ["005.13", "641.5"],
                "titulo": ["e", "f"],
            }
        )

        predictions = fit_predict(train, val, c_value=1.0)
        frame = prediction_frame(val, predictions)
        selected, table = select_c(train, val, [0.1, 1.0], "texto")

        self.assertEqual(predictions.tolist(), ["005.13", "641.5"])
        self.assertEqual(frame["method"].tolist(), ["tfidf_logreg", "tfidf_logreg"])
        self.assertIn(selected, {0.1, 1.0})
        self.assertEqual(len(table), 2)
