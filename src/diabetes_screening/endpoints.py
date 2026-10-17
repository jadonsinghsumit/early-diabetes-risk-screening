"""Outcome definitions shared by preparation, validation, and reporting."""
ENDPOINTS = {
    "prediabetes": {0: 0, 1: 1, 2: None},
    "combined": {0: 0, 1: 1, 2: 1},
    "diabetes-only": {0: 0, 1: 0, 2: 1},
}
LABELS = {
    "prediabetes": ["No diabetes", "Prediabetes"],
    "combined": ["No diabetes", "Prediabetes or diabetes"],
    "diabetes-only": ["No diabetes or prediabetes", "Diabetes"],
}
DESCRIPTIONS = {
    "prediabetes": "0: no diabetes; 1: self-reported prediabetes; existing diabetes excluded",
    "combined": "0: no diabetes; 1: self-reported prediabetes or diabetes",
    "diabetes-only": "0: no diabetes or prediabetes; 1: self-reported diabetes",
}
