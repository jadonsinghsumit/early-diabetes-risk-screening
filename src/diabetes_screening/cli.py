"""Small command-line interface for download and the four-stage workflow."""
import argparse
from pathlib import Path
import sys
import urllib.error
import zipfile

from .download import download_source
from .endpoints import ENDPOINTS
from .pipeline import run_analysis
from .preparation import prepare_binary
from .schema import EXPECTED_ROWS, FILENAME


def main(argv=None):
    parser = argparse.ArgumentParser(description="BRFSS diabetes exploration: roadmap stages 1–4")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Fetch the full binary file from UCI")
    fetch.add_argument("--output", type=Path, default=Path("data/raw") / FILENAME)
    prepare = commands.add_parser("prepare", help="Derive an explicitly labelled binary CSV from three-class input")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--endpoint", required=True, choices=list(ENDPOINTS))
    run = commands.add_parser("run", help="Validate, clean, explore, and test associations")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path, help="New directory; existing paths are never overwritten")
    run.add_argument("--provenance", type=Path, help="Preparation sidecar, checked against the input hash")
    run.add_argument("--duplicates", choices=["keep", "drop"], default="keep")
    run.add_argument("--synthetic-rows", type=int, help="Test fixtures only: override source row count and label outputs synthetic")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(f"Saved {download_source(args.output)}")
        elif args.command == "prepare":
            metadata = prepare_binary(args.input, args.output, args.endpoint)
            print(f"Prepared {metadata['rows']:,} rows with explicit endpoint: {args.endpoint}")
        else:
            expected_rows = args.synthetic_rows if args.synthetic_rows is not None else EXPECTED_ROWS
            if expected_rows < 1:
                raise ValueError("Expected row count must be positive")
            manifest = run_analysis(args.input, args.output, args.duplicates, expected_rows,
                                    synthetic=args.synthetic_rows is not None, provenance=args.provenance)
            print(f"Completed stages 1–4: {manifest['analysis_rows']:,} rows; "
                  f"{manifest['tests_executed']} tests executed; {manifest['tests_skipped']} skipped")
            print(f"Validation mode: {manifest['validation_mode']}. Outputs: {args.output}")
    except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print(f"Analysis stopped: {error}", file=sys.stderr)
        return 1
    return 0
