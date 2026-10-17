import json
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from diabetes_screening.cli import main
from diabetes_screening.pipeline import run_analysis


def test_four_stages_end_to_end(sample, tmp_path):
    source = tmp_path / "synthetic.csv"
    sample.to_csv(source, index=False)
    output = tmp_path / "results"
    result = subprocess.run([sys.executable, "-m", "diabetes_screening", "run", "--input", str(source),
                             "--output", str(output), "--synthetic-rows", "160"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "25 tests executed" in result.stdout
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["analysis_rows"] == 160 and manifest["tests_skipped"] == 0
    assert manifest["validation_mode"] == "synthetic/test only"
    cleaned = pd.read_csv(output / "cleaned.csv")
    pd.testing.assert_frame_equal(cleaned, sample)
    hypotheses = pd.read_csv(output / "hypotheses.csv")
    assert len(hypotheses) == 25 and hypotheses.p_adjusted_bh.notna().all()
    assert len(pd.read_csv(output / "data_dictionary.csv")) == 22
    assert "25.00%" in (output / "eda" / "observations.md").read_text()
    with pytest.raises(FileExistsError):
        run_analysis(source, output, expected_rows=160)


def test_cli_errors_are_nonzero_and_leave_no_success_marker(sample, tmp_path, capsys):
    source = tmp_path / "invalid.csv"
    sample.to_csv(source, index=False)
    output = tmp_path / "failed"
    assert main(["run", "--input", str(source), "--output", str(output)]) == 1
    assert "253,680" in capsys.readouterr().err
    assert not (output / "manifest.json").exists()
    assert main(["run", "--input", str(source), "--output", str(output), "--synthetic-rows", "0"]) == 1
    assert main(["run", "--input", str(tmp_path / "missing.csv"), "--output", str(tmp_path / "other")]) == 1


def test_cli_download_delegates_without_live_network(tmp_path, capsys):
    destination = tmp_path / "source.csv"
    with patch("diabetes_screening.cli.download_source", return_value=destination) as fetch:
        assert main(["download", "--output", str(destination)]) == 0
    fetch.assert_called_once_with(destination)
    assert "Saved" in capsys.readouterr().out


def test_cleaning_failure_preserves_quality_report(sample, tmp_path):
    sample.loc[0, "Age"] = 99
    source = tmp_path / "invalid.csv"
    sample.to_csv(source, index=False)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="Invalid"):
        run_analysis(source, output, expected_rows=160)
    assert (output / "quality.json").exists()
    assert not (output / "cleaned.csv").exists()
    assert not (output / "manifest.json").exists()


def test_in_process_cli_run_preserves_explicit_duplicate_policy(sample, tmp_path, capsys):
    source = tmp_path / "synthetic.csv"
    pd.concat([sample, sample.iloc[:1]]).to_csv(source, index=False)
    output = tmp_path / "sensitivity"
    assert main(["run", "--input", str(source), "--output", str(output),
                 "--synthetic-rows", "161", "--duplicates", "drop"]) == 0
    assert "160 rows" in capsys.readouterr().out
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["duplicate_policy"] == "drop"
    assert manifest["validation_mode"] == "synthetic/test only"
