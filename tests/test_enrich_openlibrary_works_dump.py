import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.data.enrich_openlibrary_works_dump import enrich_from_dump


class EnrichOpenLibraryWorksDumpTests(unittest.TestCase):
    def test_selects_requested_work_and_uses_subjects_as_descriptive_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            output = root / "output.csv"
            dump = root / "works.txt.gz"
            with source.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["titulo", "cdd", "work_id", "resumo"])
                writer.writeheader()
                writer.writerow({"titulo": "A", "cdd": "005.1", "work_id": "/works/OL1W", "resumo": ""})
                writer.writerow({"titulo": "B", "cdd": "005.2", "work_id": "/works/OL2W", "resumo": "existente"})
            records = [
                "/type/work\t/works/OL1W\t1\t2026-01-01\t" + json.dumps({"subjects": ["Computing", "computing", {"name": "Algorithms"}]}),
                "/type/work\t/works/OTHER\t1\t2026-01-01\t" + json.dumps({"subjects": ["Ignored"]}),
            ]
            with gzip.open(dump, "wt", encoding="utf-8") as file:
                file.write("\n".join(records) + "\n")

            report = enrich_from_dump(source, dump, output)
            with output.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(report["obras_encontradas"], 1)
        self.assertEqual(report["linhas_enriquecidas"], 1)
        self.assertEqual(rows[0]["resumo"], "Computing\nAlgorithms")
        self.assertEqual(json.loads(rows[0]["subjects"]), ["Computing", "Algorithms"])
        self.assertEqual(rows[0]["resumo_source"], "openlibrary_work_subjects")
        self.assertEqual(rows[1]["resumo"], "existente")
