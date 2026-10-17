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


def test_review_flags_count_unusual_but_valid_responses_without_changing_data(sample):
    sample["HighChol"], sample["CholCheck"] = 0, 1          # remove the random baseline first
    sample.loc[:2, ["HighChol", "CholCheck"]] = [[1, 0], [1, 0], [0, 0]]
    sample.loc[:1, "BMI"] = [14, 62]
    sample.loc[0, "PhysHlth"] = 30
    before = sample.copy()
    cleaned, report = clean_data(sample)
    flags = report["review_flags"]
    assert flags["HighChol=1 but CholCheck=0 (told cholesterol high without a check in five years)"] == 2
    assert flags["BMI below 15"] >= 1 and flags["BMI 60 or above"] >= 1
    assert flags["PhysHlth == 30 (heaping at the top code)"] >= 1
    pd.testing.assert_frame_equal(cleaned, before.astype("int16"))
