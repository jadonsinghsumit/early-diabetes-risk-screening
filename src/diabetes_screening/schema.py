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
    Field("GenHlth", "ordinal", 1, 5, "General health: 1 excellent, 2 very good, 3 good, 4 fair, 5 poor"),
    Field("MentHlth", "numeric", 0, 30, "Days of poor mental health in past 30 days"),
    Field("PhysHlth", "numeric", 0, 30, "Days of poor physical health in past 30 days"),
    Field("Age", "ordinal", 1, 13, "Age bands: 1=18–24, 2=25–29, 3=30–34, 4=35–39, 5=40–44, 6=45–49, 7=50–54, 8=55–59, 9=60–64, 10=65–69, 11=70–74, 12=75–79, 13=80+"),
    Field("Education", "ordinal", 1, 6, "Education: 1 never attended/kindergarten, 2 grades 1–8, 3 grades 9–11, 4 high school/GED, 5 college 1–3 years, 6 college 4+ years"),
    Field("Income", "ordinal", 1, 8, "Household income: 1 <$10k, 2 $10k–<15k, 3 $15k–<20k, 4 $20k–<25k, 5 $25k–<35k, 6 $35k–<50k, 7 $50k–<75k, 8 ≥$75k"),
]
COLUMNS = [TARGET, "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
           "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
           "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
           "Sex", "Age", "Education", "Income"]


def data_dictionary():
    """Return a serializable codebook in source column order."""
    fields = {field.name: field for field in FIELDS}
    rows = [asdict(fields[name]) for name in COLUMNS]
    for row in rows:
        if row["kind"] == "binary" and row["name"] not in {TARGET, "Sex"}:
            row["description"] += "; 0 no, 1 yes"
    return rows
