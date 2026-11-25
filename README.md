# RR-MERFISH-AIT21

Retrieve segmented Allen MERFISH sections with their original **AIT21 (CCN20230722)** labels and write one AnnData file per physical section.

The first dataset adapter is `MERFISH-C57BL6J-638850` (59 coronal sections, one adult mouse). Anatomical crops use CCF parcellation (`parcellation_division` by default). Expression is measured raw counts with the 50 blank barcode controls removed. Section coordinates are converted from millimeters to micrometers and stored in `obsm["spatial"]` for `snmf.pp.compress_to_grid`. This script does not bin cells, fit SNMF, normalize counts, or reassign cell types.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python retrieve_allen_merfish.py --list-regions --cache-dir ./abc_atlas_cache

python retrieve_allen_merfish.py \
    --region cerebellum \
    --list-sections \
    --cache-dir ./abc_atlas_cache

python retrieve_allen_merfish.py \
    --region cerebellum \
    --n-sections 5 \
    --min-cells-per-section 5000 \
    --reference data/reference/cerebellum_AIT21.h5ad \
    --output-dir data/merfish/cerebellum \
    --cache-dir ./abc_atlas_cache
```

`--sections` accepts official IDs such as `C57BL6J-638850.40` and overrides automatic ranking, still requiring the regional cell threshold. `--scope full-section` ranks by the regional count but exports every annotated cell on the chosen sections. `--manifest` selects the same Allen manifest used by the AIT21 reference retrieval when you need to pin it.

A selected cell with no raw-count row fails the run. `--allow-missing-expression` drops those cells and records their IDs. The published MERFISH metadata and section matrices differ by a few cells per section.

## Outputs

```text
data/merfish/cerebellum/
  C57BL6J-638850.<section>.h5ad
  section_summary.csv
  subclass_counts.csv
  retrieval_manifest.json
```

`adata.X` is sparse raw counts. `adata.obs["subclass"]` is the published AIT21 label. Provenance is in `adata.uns["retrieval_info"]`.

## Tests

```bash
pytest
```

A real cerebellar section plus `snmf.pp.compress_to_grid` is checked separately once the Allen cache is available.
