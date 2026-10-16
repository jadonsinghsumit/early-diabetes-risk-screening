import numpy as np
import pandas as pd
import pytest
from diabetes_screening.schema import COLUMNS, FIELDS, TARGET


@pytest.fixture
def sample():
    rng = np.random.default_rng(19)
    frame = pd.DataFrame({field.name: rng.integers(field.minimum, field.maximum + 1, 160)
                          for field in FIELDS})[COLUMNS]
    frame[TARGET] = np.tile([0, 0, 0, 1], 40)
    return frame
