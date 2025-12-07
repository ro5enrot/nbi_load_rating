#!/usr/bin/env python3
"""
Analyze NBI inventory rating method data for methods 3 and 4.

The script scans the `all_States_in_a_single_file_raw` directory for
`NBI_*_Delimited_AllStates.txt` files, aggregates records where
`INV_RATING_METH_065` equals 3 or 4, and produces the following outputs:

1. `rating_increase_records.csv`
   Rows where the same bridge (structure number) shows a strictly higher
   `INVENTORY_RATING_066` in a later year compared to any prior year,
   grouped by rating method.

2. `reconstructed_after_first_rating_records.csv`
   Rows where `YEAR_RECONSTRUCTED_106` is greater than the first year the
   bridge appears with the given inventory rating method.

3. `summary.json`
   Totals of affected records and structures for items (1) and (2),
   broken down by rating method.

All outputs are written to the same directory as this script. The summary
statistics are stored in `summary.yaml`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DATA_DIR_NAME = "all_States_in_a_single_file_raw"
CSV_GLOB = "NBI_*_Delimited_AllStates.txt"
TARGET_METHODS = {"3", "4"}


@dataclass
class Record:
    year: int
    rating: Optional[float]
    reconstruction_year: Optional[int]
    source_file: str


def load_records(data_dir: Path) -> Dict[str, Dict[str, List[Record]]]:
    """
    Load records per method -> structure -> list[Record].
    """
    records: Dict[str, Dict[str, List[Record]]] = {
        method: defaultdict(list) for method in TARGET_METHODS
    }

    files = sorted(data_dir.glob(CSV_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No files matching {CSV_GLOB!r} found in {data_dir}"
        )

    for path in files:
        name = path.name
        try:
            year = int(name.split("_")[1])
        except (IndexError, ValueError):
            # Skip unexpected file names.
            continue

        with path.open("r", encoding="latin-1", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                continue

            try:
                idx_method = header.index("INV_RATING_METH_065")
                idx_rating = header.index("INVENTORY_RATING_066")
                idx_structure = header.index("STRUCTURE_NUMBER_008")
                idx_reconstruct = header.index("YEAR_RECONSTRUCTED_106")
            except ValueError as exc:
                missing = exc.args[0]
                raise ValueError(
                    f"Required column missing in {name}: {missing}"
                ) from exc

            for row in reader:
                if idx_method >= len(row) or idx_structure >= len(row):
                    continue

                method = row[idx_method].strip()
                if method not in TARGET_METHODS:
                    continue

                structure = row[idx_structure].strip()
                if not structure:
                    continue

                rating = _parse_float(row, idx_rating)
                reconstruction_year = _parse_int(row, idx_reconstruct)

                records[method][structure].append(
                    Record(
                        year=year,
                        rating=rating,
                        reconstruction_year=reconstruction_year,
                        source_file=name,
                    )
                )

    return records


def _parse_float(row: List[str], idx: int) -> Optional[float]:
    if idx >= len(row):
        return None
    value = row[idx].strip()
    if not value or value.upper() == "N":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(row: List[str], idx: int) -> Optional[int]:
    if idx >= len(row):
        return None
    value = row[idx].strip()
    if not value or value.upper() == "N":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def analyze_rating_increase(
    records: Dict[str, Dict[str, List[Record]]]
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """
    Detect when subsequent years show a higher inventory rating.
    Returns (rows, counts) where rows is a list of dicts ready for CSV export
    and counts summarises per method.
    """
    csv_rows: List[Dict[str, object]] = []
    counts: Dict[str, int] = {method: 0 for method in TARGET_METHODS}
    unique_structures: Dict[str, set] = {method: set() for method in TARGET_METHODS}

    for method, struct_map in records.items():
        for structure, struct_records in struct_map.items():
            sorted_records = sorted(struct_records, key=lambda rec: rec.year)
            max_rating: Optional[float] = None
            for rec in sorted_records:
                if rec.rating is None:
                    continue

                if max_rating is not None and rec.rating > max_rating:
                    counts[method] += 1
                    unique_structures[method].add(structure)
                    csv_rows.append(
                        {
                            "method": method,
                            "structure_number": structure,
                            "year": rec.year,
                            "inventory_rating": rec.rating,
                            "previous_max_rating": max_rating,
                            "source_file": rec.source_file,
                        }
                    )

                if max_rating is None or rec.rating > max_rating:
                    max_rating = rec.rating

    per_method_counts = {
        method: {
            "records": counts[method],
            "structures": len(unique_structures[method]),
        }
        for method in TARGET_METHODS
    }
    return csv_rows, per_method_counts


def analyze_reconstruction_relative_to_first_year(
    records: Dict[str, Dict[str, List[Record]]]
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """
    Find rows where YEAR_RECONSTRUCTED_106 is greater than the first
    observation year for the bridge/method combination.
    Returns (rows, counts) where rows is a list of dicts ready for CSV export.
    """
    csv_rows: List[Dict[str, object]] = []
    counts: Dict[str, int] = {method: 0 for method in TARGET_METHODS}
    unique_structures: Dict[str, set] = {method: set() for method in TARGET_METHODS}

    for method, struct_map in records.items():
        for structure, struct_records in struct_map.items():
            sorted_records = sorted(struct_records, key=lambda rec: rec.year)
            if not sorted_records:
                continue
            first_year = sorted_records[0].year
            qualifies = False

            for rec in sorted_records:
                recon_year = rec.reconstruction_year
                if recon_year is None:
                    continue
                if recon_year > first_year:
                    counts[method] += 1
                    qualifies = True
                    csv_rows.append(
                        {
                            "method": method,
                            "structure_number": structure,
                            "first_measurement_year": first_year,
                            "observation_year": rec.year,
                            "reconstruction_year": recon_year,
                            "source_file": rec.source_file,
                        }
                    )

            if qualifies:
                unique_structures[method].add(structure)

    per_method_counts = {
        method: {
            "records": counts[method],
            "structures": len(unique_structures[method]),
        }
        for method in TARGET_METHODS
    }
    return csv_rows, per_method_counts


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / DATA_DIR_NAME

    records = load_records(data_dir)

    rating_increase_rows, rating_increase_counts = analyze_rating_increase(records)
    reconstruction_rows, reconstruction_counts = (
        analyze_reconstruction_relative_to_first_year(records)
    )

    summary = {}
    for method in TARGET_METHODS:
        summary[method] = {
            "rating_increase": rating_increase_counts[method],
            "reconstructed_after_first_rating": reconstruction_counts[method],
        }

    write_csv(script_dir / "rating_increase_records.csv", rating_increase_rows)
    write_csv(
        script_dir / "reconstructed_after_first_rating_records.csv",
        reconstruction_rows,
    )
    # Write YAML summary instead of JSON for easier downstream consumption
    yaml_lines = []
    for method in TARGET_METHODS:
        yaml_lines.append(f"{method}:")
        rating = summary[method]["rating_increase"]
        recon = summary[method]["reconstructed_after_first_rating"]
        yaml_lines.append(f"  rating_increase:")
        yaml_lines.append(f"    records: {rating['records']}")
        yaml_lines.append(f"    structures: {rating['structures']}")
        yaml_lines.append(f"  reconstructed_after_first_rating:")
        yaml_lines.append(f"    records: {recon['records']}")
        yaml_lines.append(f"    structures: {recon['structures']}")
    (script_dir / "summary.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
