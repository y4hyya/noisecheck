from __future__ import annotations

from pathlib import Path

import pytest

from noisecheck import LoadError, read_jsonl


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "data.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


VALID_LINE = '{"example_id": "a", "variant": "baseline", "metric": "accuracy", "value": 1}\n'


class TestReadJsonl:
    def test_reads_records_and_skips_blank_lines(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            VALID_LINE
            + "\n"
            + '{"example_id": "a", "variant": "candidate", "metric": "accuracy", "value": 0.5}\n',
        )
        ds = read_jsonl(path)
        assert len(ds) == 2
        assert ds.variants() == {"baseline", "candidate"}

    def test_invalid_json_reports_line_number(self, tmp_path: Path) -> None:
        path = write(tmp_path, VALID_LINE + "{not json}\n")
        with pytest.raises(LoadError, match="line 2"):
            read_jsonl(path)

    def test_invalid_record_reports_line_number_and_field(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            VALID_LINE + '{"example_id": "b", "variant": "baseline", "metric": "accuracy"}\n',
        )
        with pytest.raises(LoadError, match=r"line 2.*value"):
            read_jsonl(path)

    def test_non_object_line_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, "[1, 2]\n")
        with pytest.raises(LoadError, match="object"):
            read_jsonl(path)

    def test_empty_file_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, "\n")
        with pytest.raises(LoadError, match="no records"):
            read_jsonl(path)

    def test_missing_file_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(LoadError):
            read_jsonl(tmp_path / "missing.jsonl")

    def test_duplicate_rows_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, VALID_LINE + VALID_LINE)
        with pytest.raises(LoadError, match="duplicate"):
            read_jsonl(path)
