import json
import unittest

import pandas as pd

from src.evaluate.compare_llm_conditions import compare


class CompareLlmConditionsTests(unittest.TestCase):
    def test_reports_paired_changes_and_keyword_audit(self):
        l0 = pd.DataFrame({"edition_id": ["a", "b", "c"], "cdd_true": ["001.42"] * 3, "cdd_pred": ["001.42", "001.43", "001.43"], "keywords": ["[]"] * 3})
        l1 = pd.DataFrame({"edition_id": ["a", "b", "c"], "cdd_true": ["001.42"] * 3, "cdd_pred": ["001.43", "001.42", "001.43"], "keywords": [json.dumps(["redes", "dados"]), "[]", json.dumps(["redes"]) ]})

        report, pairs, terms = compare(l0, l1)

        self.assertEqual(report["pares"], {"ambos_acertam": 0, "somente_l0_acerta": 1, "somente_l1_acerta": 1, "ambos_erram": 1})
        self.assertEqual(report["keywords"]["linhas_sem_keywords"], 1)
        self.assertEqual(terms.iloc[0].to_dict(), {"keyword": "redes", "frequencia": 2})
        self.assertEqual(len(pairs), 3)
