from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RSV_CONTEXT = (
    "Ireland offers a free RSV immunisation program (Nirsevimab) to protect infants, "
    "focusing on newborns and high-risk babies during winter. The program is provided "
    "in maternity hospitals for babies born from 1 September 2025 to 28 February 2026, "
    "with catch-up options for those born from 1 March 2025."
)


def workbook_rows():
    return [
        {
            "Facility": "CHI",
            "Accession Number": "ACC-001",
            "Exam Description": "XR CHEST",
            "Imaged DTTM": "2025-09-01 10:00:00",
            "Age": 0.2,
            "Age Category": "Paeds",
            "Patient Class": "IPUB",
            "Order Location": "LOC-A",
            "Ref Office Code": "REF-A",
            "Admission DTTM": "2025-09-01 12:00:00",
            "Discharge DTTM": "2025-09-02 10:00:00",
            "Exam Reason": "cough",
            "Location Type": "IP",
            "Attending Specialty": "PAEDIATRICS",
            "PACS Work Group": "WG-A",
            "Imaged Day of Week": "Mon",
            "Imaged Hour of Day": 10,
            "9-5 Flag": 1,
            "Prelim Texts": "portable chest",
        },
        {
            "Facility": "CHI",
            "Accession Number": "ACC-002",
            "Exam Description": "XR CHEST",
            "Imaged DTTM": "02/09/2025 11:30",
            "Age": 0.4,
            "Age Category": "Paeds",
            "Patient Class": "OPUB",
            "Order Location": "LOC-B",
            "Ref Office Code": "REF-B",
            "Admission DTTM": "02/09/2025 11:00",
            "Discharge DTTM": "02/09/2025 12:00",
            "Exam Reason": "wheeze",
            "Location Type": "OP",
            "Attending Specialty": "RESP",
            "PACS Work Group": "WG-B",
            "Imaged Day of Week": "Tue",
            "Imaged Hour of Day": 11,
            "9-5 Flag": 1,
            "Prelim Texts": "portable chest follow-up",
        },
        {
            "Facility": "*",
            "Accession Number": "BROKEN-ROW",
            "Exam Description": "XR CHEST",
            "Imaged DTTM": None,
            "Age": -1,
            "Age Category": "Paeds",
            "Patient Class": "IPUB",
            "Order Location": "LOC-BAD",
            "Ref Office Code": None,
            "Admission DTTM": None,
            "Discharge DTTM": None,
            "Exam Reason": None,
            "Location Type": "IP",
            "Attending Specialty": "RESP",
            "PACS Work Group": "WG-B",
            "Imaged Day of Week": None,
            "Imaged Hour of Day": None,
            "9-5 Flag": None,
            "Prelim Texts": None,
        },
    ]


def write_workbook(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    return path


def build_normalized_events(days: int = 60) -> pd.DataFrame:
    start = pd.Timestamp("2025-09-01")
    rows = []
    exam_counter = 1
    for day_offset in range(days):
        current_day = start + pd.Timedelta(days=day_offset)
        base_count = 3 + (day_offset % 4)
        for facility_id, age_years in [("CHI", 0.25), ("CHI", 1.5), ("OTHER", 0.3)]:
            count = base_count if facility_id == "CHI" else 1
            for sample_idx in range(count):
                imaged_ts = current_day + pd.Timedelta(hours=9 + sample_idx)
                rows.append(
                    {
                        "source_workbook": "synthetic.xlsx",
                        "source_sheet": "Sheet1",
                        "exam_id": f"EXAM-{exam_counter:05d}",
                        "facility_id": facility_id,
                        "exam_description": "XR CHEST",
                        "imaged_ts": imaged_ts,
                        "imaged_date": imaged_ts.floor("D"),
                        "imaged_day_of_week": imaged_ts.day_name()[:3],
                        "imaged_hour_of_day": imaged_ts.hour,
                        "is_business_hours_derived": True,
                        "admission_ts": imaged_ts - pd.Timedelta(hours=1),
                        "discharge_ts": imaged_ts + pd.Timedelta(hours=4),
                        "age_years": age_years,
                        "age_category": "Paeds",
                        "patient_class": "IPUB",
                        "order_location": "LOC-A" if facility_id == "CHI" else "LOC-C",
                        "ref_office_code": "REF",
                        "location_type": "IP",
                        "attending_specialty": "PAEDIATRICS",
                        "pacs_work_group": "WG-A" if facility_id == "CHI" else "WG-C",
                        "exam_reason_text": "respiratory distress",
                        "prelim_text": "portable chest",
                        "imaged_day_of_week_source": imaged_ts.day_name()[:3],
                        "imaged_hour_of_day_source": float(imaged_ts.hour),
                        "business_hours_flag_source": 1.0,
                    }
                )
                exam_counter += 1
    return pd.DataFrame(rows)


def write_normalized_csv(path: Path, days: int = 60) -> Path:
    df = build_normalized_events(days=days)
    df.to_csv(path, index=False)
    return path


class TempDirTestCaseMixin:
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()
