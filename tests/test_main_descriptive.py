import unittest

import pandas as pd

from main import only_descriptive_rows


class DescriptiveSelectionTests(unittest.TestCase):
    def test_keeps_only_rows_with_descriptive_text(self):
        data = pd.DataFrame(
            {"titulo": ["A", "B", "C"], "resumo": ["subject", "  ", None]}
        )
        self.assertEqual(only_descriptive_rows(data)["titulo"].tolist(), ["A"])

    def test_requires_summary_column(self):
        with self.assertRaisesRegex(ValueError, "resumo"):
            only_descriptive_rows(pd.DataFrame({"titulo": ["A"]}))
