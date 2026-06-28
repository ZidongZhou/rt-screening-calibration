# Reproduction run report

Release configuration:

- Random baseline repeats: 50
- Full leakage audit: 100 participants × 20 repeats × 4 prefix lengths
- Item budgets: 11, 19, and 26
- Raw data: not included in the clean archive
- Verified runtime: Python 3.11.8

Download the five public CSV files from Zenodo record
`10.5281/zenodo.10423537` with `python scripts/download_data.py`, which also
verifies the published MD5 checksums.

## Clean reproduction verification

A clean Windows reproduction was run after deleting generated models, processed
caches, tables, and figures, then restoring only the checksum-verified source
CSVs:

- primary stage: 1,683.7 seconds;
- robustness stage: 1,785.4 seconds;
- additional-analysis stage: 203.5 seconds;
- selective-auditing stage: 328.5 seconds.

Publication export and both submission-output checks passed. The complete test
suite reported `6 passed`. The full leakage audit reported 8,000/8,000 unchanged
policy simulations after future-response perturbation and 8,000/8,000 after
future-RT perturbation.

The packaged manuscript outputs can be checked without raw CSVs. Full
numerical regeneration requires the downloaded CSVs and the staged commands in
the repository README.
