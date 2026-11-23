import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.audit_descriptive_corpus import audit_descriptive_corpus


class DescriptiveCorpusAuditTests(unittest.TestCase):
    def test_audit_counts_only_nonempty_summaries_per_cdd(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.csv"
            pd.DataFrame(
                {
                    "titulo": ["A", "B", "C", "D"],
                    "cdd": ["001.42", "001.42", "005.13", "005.13"],
                    "resumo": ["um", "dois", "", "três"],
                }
            ).to_csv(source, index=False)
            report, counts = audit_descriptive_corpus(source)

        self.assertEqual(report["registros_com_resumo"], 3)
        self.assertEqual(report["limiares_por_classe"]["2"], {"classes": 1, "registros": 2})
        self.assertEqual(counts.iloc[0].to_dict(), {"cdd": "001.42", "registros_com_resumo": 2})
