import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.cdd import normalize_cdd, truncate_cdd
from src.data.extract_openlibrary_editions import extract
from src.data.prepare_dataset import prepare_dataset
from src.data.inspect_openlibrary_cdd import inspect
from src.extractors.tfidf import extract_keywords_tfidf
from src.data.enrich_openlibrary_works import work_summary


class OpenLibraryExtractionTests(unittest.TestCase):
    def test_normalize_common_dewey_forms(self):
        self.assertEqual(normalize_cdd("028/.9"), "028.9")
        self.assertEqual(normalize_cdd("303.48/33"), "303.4833")
        self.assertEqual(normalize_cdd("005.133 Manual"), "005.133")
        self.assertEqual(normalize_cdd("sem classificação"), "")

    def test_truncate_cdd_to_five_digits(self):
        self.assertEqual(truncate_cdd("303.48/33"), "303.48")
        self.assertEqual(truncate_cdd("005.133 Manual"), "005.13")
        self.assertEqual(truncate_cdd("025"), "025")

    def test_tfidf_accepts_title_without_valid_terms(self):
        self.assertEqual(extract_keywords_tfidf("de em o"), [])

    def test_work_summary_prefers_description_then_excerpts(self):
        self.assertEqual(work_summary({"description": {"value": "Resumo."}}), ("Resumo.", "openlibrary_work_description"))
        self.assertEqual(
            work_summary({"excerpts": [{"excerpt": "Primeiro trecho."}, {"excerpt": "Segundo trecho."}]}),
            ("Primeiro trecho.\nSegundo trecho.", "openlibrary_work_excerpts"),
        )

    def test_extracts_rows_without_loading_the_dump(self):
        records = [
            ["/type/edition", "/books/OL1M", "1", "2026-01-01", {"key": "/books/OL1M", "title": "Livro A", "dewey_decimal_class": ["028/.9"], "works": [{"key": "/works/OL1W"}], "languages": [{"key": "/languages/por"}]}],
            ["/type/edition", "/books/OL2M", "1", "2026-01-01", {"key": "/books/OL2M", "title": "Livro B", "dewey_decimal_class": ["004.1", "005.1"]}],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dump = root / "editions.txt"
            with dump.open("w", encoding="utf-8") as file:
                for kind, key, revision, date, record in records:
                    file.write("\t".join((kind, key, revision, date, json.dumps(record))) + "\n")
            output = root / "dataset.csv"
            report = extract(dump, output, prefixes=("004",), max_per_cdd=1)
            with output.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(report["records"], 1)
        self.assertEqual(rows[0]["cdd"], "004.1")
        self.assertEqual(rows[0]["source"], "openlibrary_editions_2026-08-31")

    def test_extract_uses_one_cdd_per_edition(self):
        record = {
            "key": "/books/OL1M",
            "title": "Livro com múltiplas CDDs",
            "dewey_decimal_class": ["004.1", "005.1"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dump = root / "editions.txt"
            dump.write_text("\t".join(("/type/edition", "/books/OL1M", "1", "2026-01-01", json.dumps(record))) + "\n", encoding="utf-8")
            output = root / "dataset.csv"
            extract(dump, output, prefixes=("004", "005"))
            with output.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cdd"], "004.1")
        self.assertEqual(rows[0]["cdd_raw"], "004.1 | 005.1")

    def test_prepare_can_reuse_a_reference_label_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            reference = root / "reference.csv"
            pd.DataFrame(
                {
                    "titulo": ["A", "B", "C", "D", "E", "F"],
                    "cdd": ["001.42", "001.42", "001.42", "005.13", "005.13", "005.13"],
                }
            ).to_csv(source, index=False)
            pd.DataFrame({"cdd": ["001.42"]}).to_csv(reference, index=False)

            clean, _, _, _, report = prepare_dataset(
                source, root / "prepared", min_per_class=3, allowed_cdds_path=reference
            )

        self.assertEqual(clean["cdd"].unique().tolist(), ["001.42"])
        self.assertEqual(report["allowed_cdds_path"], str(reference))

    def test_inspection_counts_truncated_target_cdds(self):
        records = [
            ["/type/edition", "/books/OL1M", "1", "2026-01-01", {"dewey_decimal_class": ["005.133", "005.1339", "004.123"]}],
            ["/type/edition", "/books/OL2M", "1", "2026-01-01", {"dewey_decimal_class": ["025.049"]}],
        ]
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory) / "editions.txt"
            with dump.open("w", encoding="utf-8") as file:
                for kind, key, revision, date, record in records:
                    file.write("\t".join((kind, key, revision, date, json.dumps(record))) + "\n")
            report = inspect(dump, prefixes=("004", "005", "025"), max_digits=5)

        self.assertEqual(report["linhas_extraiveis_total"], 3)
        self.assertEqual(report["por_faixa"]["005"]["top_cdds"], [("005.13", 1)])
        self.assertEqual(report["por_faixa"]["025"]["top_cdds"], [("025.04", 1)])
