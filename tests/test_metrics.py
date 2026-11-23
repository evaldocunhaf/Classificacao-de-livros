import unittest

import pandas as pd

from src.evaluate.metrics import canonical_cdd, evaluate


class MetricsTests(unittest.TestCase):
    def test_canonical_cdd_limits_precision_and_restores_leading_zero(self):
        self.assertEqual(canonical_cdd("025.4678"), "025.46")
        self.assertEqual(canonical_cdd("25.4678"), "025.46")
        self.assertEqual(canonical_cdd(""), "")


    def test_metrics_use_only_gold_labels_and_separate_fallback(self):
        result = pd.DataFrame(
            {
                "cdd_true": ["005.1234", "006.31", "001.42"],
                "cdd_pred": ["005.1299", "999.999", ""],
                "method": ["llm", "knn", "none"],
            }
        )

        report = evaluate(result)

        self.assertEqual(report["accuracy_por_nivel"]["full"], {"accuracy": 0.3333, "hits": 1, "total": 3})
        self.assertEqual(report["accuracy_por_nivel"]["section"]["hits"], 1)
        self.assertEqual(report["labels_gabarito"], ["001.42", "005.12", "006.31"])
        self.assertEqual(report["predicoes_fora_do_gabarito"], 2)
        self.assertEqual(set(report["por_origem_predicao"]), {"knn", "llm", "none"})
        self.assertNotIn("999.99", report["labels_gabarito"])
