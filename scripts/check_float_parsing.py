"""Check that a task's CSV/TSV result files parse to the floats they encode.

pandas' default float_precision='high' parser is fast but not correctly rounded,
so a few percent of cells come back 1-2 ULP away from the value written in the
file. The dashboard reads the affected tables with float_precision='round_trip';
this script re-checks that against Python's float(), which is correctly rounded.

Usage:
    python scripts/check_float_parsing.py <task_id> [<task_id> ...]
    python scripts/check_float_parsing.py --file <path> [--sep TAB|,] [--index-col N]

Exits non-zero if round_trip disagrees with float() anywhere.
"""
import argparse
import csv
import io
import sys

import numpy as np
import pandas as pd
import requests

# file -> (separator, index_col) for the result files the dashboard parses as floats
TASK_FILES = {
    "nf_output/bin_counts/binned_spectra.csv": (",", 0),
    "nf_output/output_histogram_data_directory/labels_spectra.tsv": ("\t", None),
    "nf_output/search/query_query_distances.tsv": ("\t", None),
}


def compare(text, sep, index_col, label):
    """Report how far default and round_trip parsing land from float() ground truth."""
    default = pd.read_csv(io.StringIO(text), sep=sep, index_col=index_col)
    exact = pd.read_csv(io.StringIO(text), sep=sep, index_col=index_col,
                        float_precision='round_trip')

    float_cols = [c for c in default.columns if default[c].dtype == np.float64]
    if not float_cols:
        print(f"  {label:<46} no float columns")
        return 0

    rows = list(csv.reader(io.StringIO(text), delimiter=sep))
    header = {name: i for i, name in enumerate(rows[0])}

    bad_default = bad_exact = total = 0
    for row_i, row in enumerate(rows[1:]):
        for col in float_cols:
            if col not in header:
                continue
            raw = row[header[col]]
            if raw == '':
                continue
            total += 1
            truth = float(raw)
            if default.iloc[row_i][col] != truth:
                bad_default += 1
            if exact.iloc[row_i][col] != truth:
                bad_exact += 1

    status = "OK" if bad_exact == 0 else "MISMATCH"
    print(f"  {label:<46} {total:6d} cells   default off by {bad_default:5d}   "
          f"round_trip off by {bad_exact:5d}   {status}")
    return bad_exact


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task_ids", nargs="*", help="GNPS2 task ids to check")
    parser.add_argument("--file", help="check a local file instead of a task")
    parser.add_argument("--sep", default=",", help="separator for --file (use TAB for tab)")
    parser.add_argument("--index-col", type=int, default=None)
    args = parser.parse_args()

    failures = 0

    if args.file:
        sep = "\t" if args.sep.upper() == "TAB" else args.sep
        with open(args.file) as handle:
            failures += compare(handle.read(), sep, args.index_col, args.file)
    elif args.task_ids:
        for task in args.task_ids:
            print(task)
            for path, (sep, index_col) in TASK_FILES.items():
                url = f"https://gnps2.org/resultfile?task={task}&file={path}"
                response = requests.get(url, timeout=120)
                if response.status_code != 200:
                    print(f"  {path.split('/')[-1]:<46} unavailable "
                          f"(HTTP {response.status_code})")
                    continue
                failures += compare(response.text, sep, index_col, path.split('/')[-1])
    else:
        parser.error("give at least one task id, or --file")

    if failures:
        print(f"\nFAILED: round_trip parsing disagreed with float() in {failures} cells")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
