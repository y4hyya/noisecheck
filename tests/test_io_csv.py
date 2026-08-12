from __future__ import annotations

from pathlib import Path

import pytest

from noisecheck import LoadError, read_csv


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(text, encoding="utf-8")
    return path


HEADER = "example_id,variant,metric,value,cluster_id,run_id\n"


class TestReadCsv:
    def test_reads_records_with_optional_columns(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            HEADER + "a,baseline,accuracy,1,conv1,r1\na,candidate,accuracy,0,,\n",
        )
        ds = read_csv(path)
        assert len(ds) == 2
        first, second = ds.records
        assert first.cluster_id == "conv1"
        assert first.run_id == "r1"
        assert second.cluster_id is None
        assert second.run_id is None

    def test_optional_columns_may_be_absent(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "example_id,variant,metric,value\na,baseline,accuracy,0.5\n",
        )
        ds = read_csv(path)
        assert ds.records[0].value == 0.5

    def test_missing_required_column_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, "example_id,variant,metric\na,baseline,accuracy\n")
        with pytest.raises(LoadError, match="value"):
            read_csv(path)

    def test_unexpected_column_rejected(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            "example_id,variant,metric,value,score\na,baseline,accuracy,1,9\n",
        )
        with pytest.raises(LoadError, match=r"unexpected.*score"):
            read_csv(path)

    def test_bad_number_reports_line_number(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + "a,baseline,accuracy,1,,\nb,baseline,accuracy,fast,,\n")
        with pytest.raises(LoadError, match=r"line 3.*fast"):
            read_csv(path)

    def test_ragged_row_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + "a,baseline,accuracy,1,conv1,r1,surprise\n")
        with pytest.raises(LoadError, match="line 2"):
            read_csv(path)

    def test_empty_file_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, "")
        with pytest.raises(LoadError, match="empty"):
            read_csv(path)

    def test_header_without_rows_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER)
        with pytest.raises(LoadError, match="no records"):
            read_csv(path)

    def test_missing_file_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(LoadError):
            read_csv(tmp_path / "missing.csv")

    def test_short_row_reports_missing_value(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + "a,baseline\n")
        with pytest.raises(LoadError, match=r"line 2.*value is missing"):
            read_csv(path)

    def test_blank_field_reports_line_number_and_field(self, tmp_path: Path) -> None:
        path = write(tmp_path, HEADER + ",baseline,accuracy,1,,\n")
        with pytest.raises(LoadError, match=r"line 2.*example_id"):
            read_csv(path)
