"""BRFSS-derived codebook; categories retain their original survey codes."""
from dataclasses import asdict, dataclass

TARGET = "Diabetes_binary"
EXPECTED_ROWS = 253_680
FILENAME = "diabetes_binary_health_indicators_BRFSS2015.csv"


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    minimum: int
    maximum: int
    description: str


_BINARY = {
    TARGET: "0: no diabetes; 1: self-reported prediabetes or diabetes",
    "HighBP": "Ever told blood pressure was high",
    "HighChol": "Ever told cholesterol was high",
    "CholCheck": "Cholesterol checked in the past five years",
    "Smoker": "Smoked at least 100 cigarettes in lifetime",
    "Stroke": "Ever told of a stroke",
    "HeartDiseaseorAttack": "Coronary heart disease or myocardial infarction",
    "PhysActivity": "Physical activity in past 30 days, excluding job",
    "Fruits": "Consumes fruit at least once daily",
    "Veggies": "Consumes vegetables at least once daily",
    "HvyAlcoholConsump": "Heavy drinking under source sex-specific thresholds",
    "AnyHealthcare": "Has healthcare coverage",
    "NoDocbcCost": "Could not see a doctor because of cost in past year",
    "DiffWalk": "Serious difficulty walking or climbing stairs",
    "Sex": "Source coding: 0 female; 1 male",
}
FIELDS = [Field(name, "binary", 0, 1, description) for name, description in _BINARY.items()]
FIELDS += [
    Field("BMI", "numeric", 12, 98, "Body mass index in kg/m²; source-derived observed bounds"),
    Field("GenHlth", "ordinal", 1, 5, "General health: 1 excellent through 5 poor"),
    Field("MentHlth", "numeric", 0, 30, "Days of poor mental health in past 30 days"),
    Field("PhysHlth", "numeric", 0, 30, "Days of poor physical health in past 30 days"),
    Field("Age", "ordinal", 1, 13, "Age bands: 1=18–24, 2=25–29, then five-year bands; 13=80+"),
    Field("Education", "ordinal", 1, 6, "Education: 1 never attended/kindergarten through 6 college graduate"),
    Field("Income", "ordinal", 1, 8, "Household income: 1 below $10,000 through 8 at least $75,000"),
]
COLUMNS = [TARGET, "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
           "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
           "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
           "Sex", "Age", "Education", "Income"]


def data_dictionary():
    """Return a serializable codebook in source column order."""
    fields = {field.name: field for field in FIELDS}
    return [asdict(fields[name]) for name in COLUMNS]
