import hashlib
import io
import json
import zipfile

import pytest

from diabetes_screening.preparation import THREE_CLASS_FILE, prepare_binary
from diabetes_screening.pipeline import run_analysis


def three_class_csv(sample):
    frame = sample.rename(columns={"Diabetes_binary": "Diabetes_012"})
    frame["Diabetes_012"] = [0, 1, 2, 0] * 40
    return frame


@pytest.mark.parametrize("endpoint,positives", [("combined", 80), ("diabetes-only", 40)])
def test_prepare_preserves_rows_and_explicit_mapping(sample, tmp_path, endpoint, positives):
    path = tmp_path / "upload.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(THREE_CLASS_FILE, three_class_csv(sample).to_csv(index=False))
    output = tmp_path / "derived.csv"
    metadata = prepare_binary(path, output, endpoint, expected_rows=160)
    assert metadata["derived_target_counts"]["1"] == positives
    assert metadata["rows"] == 160
    assert metadata["original_target_counts"] == {"0": 80, "1": 40, "2": 40}
    assert metadata["derived_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert not metadata["is_official_binary_file"]
    with pytest.raises(FileExistsError):
        prepare_binary(path, output, endpoint, expected_rows=160)


def test_bad_targets_and_wrong_variant_are_rejected(sample, tmp_path):
    path = tmp_path / "source.csv"
    sample.to_csv(path, index=False)
    with pytest.raises(ValueError, match="three-class schema"):
        prepare_binary(path, tmp_path / "out.csv", "combined", expected_rows=160)
    frame = three_class_csv(sample)
    frame.loc[0, "Diabetes_012"] = 9
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid labels"):
        prepare_binary(path, tmp_path / "out.csv", "combined", expected_rows=160)
    with pytest.raises(ValueError, match="Endpoint"):
        prepare_binary(path, tmp_path / "out.csv", "unknown")
    with zipfile.ZipFile(tmp_path / "wrong.zip", "w") as archive:
        archive.writestr("wrong.csv", "hello")
    with pytest.raises(ValueError, match="exactly one"):
        prepare_binary(tmp_path / "wrong.zip", tmp_path / "out.csv", "combined")


def test_provenance_is_checked_before_analysis(sample, tmp_path):
    source = tmp_path / "source.csv"
    sample.to_csv(source, index=False)
    sidecar = tmp_path / "provenance.json"
    sidecar.write_text(json.dumps({"derived_sha256": "wrong"}))
    with pytest.raises(ValueError, match="hash"):
        run_analysis(source, tmp_path / "bad", expected_rows=160, provenance=sidecar)


def test_diabetes_only_analysis_labels_endpoint(sample, tmp_path):
    source = tmp_path / "three.csv"
    three_class_csv(sample).to_csv(source, index=False)
    prepared = tmp_path / "derived.csv"
    prepare_binary(source, prepared, "diabetes-only", expected_rows=160)
    result = run_analysis(prepared, tmp_path / "results", expected_rows=160,
                          provenance=prepared.with_suffix(".provenance.json"))
    assert result["endpoint"] == "diabetes-only"
    assert "diabetes-only" in (tmp_path / "results/eda/observations.md").read_text()
