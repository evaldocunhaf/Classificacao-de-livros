import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.audit_dataset import audit


class DatasetAuditTests(unittest.TestCase):
    def test_audit_reports_work_overlap_and_duplicate_editions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                {
                    "edition_id": ["e1", "e2"],
                    "work_id": ["w1", "w2"],
                    "cdd": ["001.42", "005.13"],
                    "resumo": ["texto", ""],
                }
            ).to_csv(root / "train.csv", index=False)
            pd.DataFrame(
                {"edition_id": ["e3"], "work_id": ["w3"], "cdd": ["001.42"], "resumo": [""]}
            ).to_csv(root / "val.csv", index=False)
            pd.DataFrame(
                {"edition_id": ["e2"], "work_id": ["w4"], "cdd": ["005.13"], "resumo": [""]}
            ).to_csv(root / "test.csv", index=False)

            report, distribution = audit(root)

        self.assertEqual(report["intersecao_work_id"], {"train_val": 0, "train_test": 0, "val_test": 0})
        self.assertEqual(report["duplicatas"]["edicoes_repetidas"], 1)
        self.assertEqual(report["total"]["com_resumo"], 1)
        self.assertEqual(len(distribution), 4)
