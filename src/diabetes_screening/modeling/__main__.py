"""Allow `python -m diabetes_screening.modeling ...` as well as `diabetes-screening-model ...`."""
from .cli import main

raise SystemExit(main())
