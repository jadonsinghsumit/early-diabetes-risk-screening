import pytest
from diabetes_screening.ingestion import DataQualityError, audit, load_source
from diabetes_screening.schema import COLUMNS, data_dictionary


def test_load_records_source_and_schema(sample, tmp_path):
    path = tmp_path / "source.csv"
    sample.to_csv(path, index=False)
    frame, report = load_source(path, expected_rows=160)
    assert frame.shape == (160, 22)
    assert len(report["source"]["sha256"]) == 64
    assert report["invalid_values"] == {}
    assert [row["name"] for row in data_dictionary()] == COLUMNS


def test_wrong_variant_rejected(sample, tmp_path):
    path = tmp_path / "source.csv"
    sample.to_csv(path, index=False)
    with pytest.raises(DataQualityError, match="253,680"):
        load_source(path)
    sample.drop(columns="BMI").to_csv(path, index=False)
    with pytest.raises(DataQualityError, match="22 columns"):
        load_source(path, expected_rows=160)


def test_invalid_codes_and_missingness_are_visible(sample):
    sample.loc[0, "BMI"] = 500
    sample.loc[1, "HighBP"] = 2
    sample.loc[2, "Age"] = float("nan")
    report = audit(sample)
    assert report["invalid_values"] == {"HighBP": 1, "BMI": 1}
    assert report["missing"]["Age"] == 1
