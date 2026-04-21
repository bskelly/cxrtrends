from __future__ import annotations

import unittest

from cik_benchmark.data.cxr_trends import ValidationError, normalize_workbook
from tests.helpers import TempDirTestCaseMixin, workbook_rows, write_workbook


class TestIngestion(TempDirTestCaseMixin, unittest.TestCase):
    def test_normalize_workbook_parses_and_reports_quality_checks(self):
        workbook_path = write_workbook(self.tmp_path / "sample.xlsx", workbook_rows())

        result = normalize_workbook(workbook_path)

        self.assertEqual(result.metadata["normalized_rows"], 2)
        self.assertEqual(result.metadata["dropped_rows"], 1)
        self.assertEqual(
            result.metadata["dropped_rows_by_reason"]["missing_imaged_ts"], 1
        )
        self.assertEqual(result.dataframe["exam_id"].tolist(), ["ACC-001", "ACC-002"])
        self.assertEqual(str(result.dataframe.loc[1, "imaged_ts"]), "2025-09-02 11:30:00")

        checks = result.metadata["quality_checks"]
        self.assertEqual(checks["malformed_facility_rows"], 1)
        self.assertEqual(checks["negative_age_rows"], 1)
        self.assertEqual(checks["admission_after_imaged_rows"], 1)
        self.assertIn("negative_age_rows=1", checks["warnings"])

    def test_normalize_workbook_rejects_duplicate_exam_ids(self):
        rows = workbook_rows()
        rows[1]["Accession Number"] = "ACC-001"
        rows[1]["Imaged DTTM"] = "2025-09-02 12:00:00"
        workbook_path = write_workbook(self.tmp_path / "duplicates.xlsx", rows[:2])

        with self.assertRaises(ValidationError):
            normalize_workbook(workbook_path)

    def test_normalize_workbook_rejects_missing_required_columns(self):
        rows = workbook_rows()
        del rows[0]["Exam Reason"]
        del rows[1]["Exam Reason"]
        del rows[2]["Exam Reason"]
        workbook_path = write_workbook(self.tmp_path / "missing_column.xlsx", rows)

        with self.assertRaises(ValidationError):
            normalize_workbook(workbook_path)


if __name__ == "__main__":
    unittest.main()
