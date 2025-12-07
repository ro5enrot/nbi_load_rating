# Inventory Rating Method 065 Analysis

This folder contains the code and outputs requested for analysing records
where `INV_RATING_METH_065` equals `3` or `4` across all available NBI files.

## Contents

- `inv_rating_method_analysis.py` – processing script that reads every
  `NBI_*_Delimited_AllStates.txt` file under
  `../all_States_in_a_single_file_raw` and generates the artefacts below.
- `rating_increase_records.csv` – rows where the same bridge has a higher
  `INVENTORY_RATING_066` in a later year for the same method.
- `reconstructed_after_first_rating_records.csv` – rows where
  `YEAR_RECONSTRUCTED_106` is greater than the first observation year for
  the bridge/method pair.
- `summary.yaml` – aggregate counts of affected records and structures for
  both analyses, broken down by method, stored as YAML for easier
  consumption in ML/analysis tooling.

## Running the analysis

```bash
python inv_rating_method_analysis.py
```

The script is idempotent; running it again will refresh the CSV and YAML
outputs based on the current source data.
