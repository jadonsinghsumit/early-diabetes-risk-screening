import pandas as pd
import pytest
from diabetes_screening.cleaning import clean_data
from diabetes_screening.ingestion import DataQualityError


def test_identical_profiles_are_retained_by_default(sample):
    frame = pd.concat([sample, sample.iloc[:1]], ignore_index=True)
    cleaned, report = clean_data(frame)
    assert len(cleaned) == 161
    assert report["duplicate_rows"] == 1
    assert cleaned["BMI"].tolist() == frame["BMI"].tolist()
    deduplicated, _ = clean_data(frame, "drop")
    assert len(deduplicated) == 160


@pytest.mark.parametrize("column,value", [("Age", 14), ("BMI", 10), ("PhysHlth", 31),
                                          ("Sex", 2), ("Income", -1), ("BMI", 22.5),
                                          ("BMI", float("inf"))])
def test_invalid_values_stop_analysis(sample, column, value):
    sample[column] = sample[column].astype(float)
    sample.loc[0, column] = value
    with pytest.raises(DataQualityError, match="require review"):
        clean_data(sample)


def test_missingness_and_policy_require_explicit_decisions(sample):
    sample.loc[0, "BMI"] = float("nan")
    with pytest.raises(DataQualityError, match="Missing values"):
        clean_data(sample)
    with pytest.raises(ValueError, match="duplicate_policy"):
        clean_data(sample, "guess")
    with pytest.raises(DataQualityError, match="empty"):
        clean_data(sample.iloc[:0])
