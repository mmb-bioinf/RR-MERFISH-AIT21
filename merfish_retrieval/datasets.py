"""Dataset adapters and anatomical aliases for MERFISH retrieval."""

from __future__ import annotations

from dataclasses import dataclass


# Verified aliases shared with the AIT21 branch of retrieve_allen_reference.py.
# Targets are CCF / WMB division acronyms and must exist in the loaded metadata.
REGION_ALIASES_AIT21: dict[str, str] = {
    "cerebellum": "CB",
    "cb": "CB",
    "hippocampus": "HPF",
    "hippocampal formation": "HPF",
    "hpf": "HPF",
    "cortex": "Isocortex",
    "isocortex": "Isocortex",
    "thalamus": "TH",
    "th": "TH",
    "hypothalamus": "HY",
    "hy": "HY",
    "midbrain": "MB",
    "mb": "MB",
    "striatum": "STR",
    "str": "STR",
    "pallidum": "PAL",
    "pal": "PAL",
    "olfactory areas": "OLF",
    "olfactory": "OLF",
    "olf": "OLF",
    "cortical subplate": "CTXsp",
    "ctxsp": "CTXsp",
    "pons": "P",
    "p": "P",
    "medulla": "MY",
    "my": "MY",
    "hindbrain": "MY",
}

DIVISION_DESCRIPTIONS: dict[str, str] = {
    "CB": "Cerebellum",
    "CTXsp": "Cortical subplate",
    "HPF": "Hippocampal formation",
    "HY": "Hypothalamus",
    "Isocortex": "Isocortex",
    "MB": "Midbrain",
    "MY": "Medulla",
    "OLF": "Olfactory areas",
    "P": "Pons",
    "PAL": "Pallidum",
    "STR": "Striatum",
    "TH": "Thalamus",
}

REGION_LEVELS: dict[str, str] = {
    "division": "parcellation_division",
    "structure": "parcellation_structure",
    "substructure": "parcellation_substructure",
}

# Allen documents ~200 µm between coronal sections, stored as 0.2 mm in z.
EXPECTED_Z_SPACING_MM = 0.2
Z_SPACING_TOLERANCE_MM = 0.05
MM_TO_UM = 1000.0

TAXONOMY = "AIT21"
TAXONOMY_ID = "CCN20230722"
TAXONOMY_NAME = "20230722_BICCN-Adult_Mouse_WholeBrain_10X"
TAXONOMY_DIRECTORY = "WMB-taxonomy"

COMPATIBLE_TAXONOMY_TOKENS = frozenset(
    {"AIT21", "CCN20230722", "WMB-taxonomy", TAXONOMY_NAME}
)
INCOMPATIBLE_TAXONOMY_TOKENS = frozenset(
    {
        "AIT33",
        "CCN20251031",
        "Consensus-WMB-integrated-taxonomy",
        "20251030_BICCN-Adult_Mouse_WholeBrain_consensus",
    }
)


@dataclass(frozen=True)
class MerfishDataset:
    """Where one MERFISH release stores metadata and raw counts."""

    key: str
    metadata_directories: tuple[str, ...]
    parcellation_file: str
    cluster_metadata_directory: str
    cluster_metadata_file: str
    ccf_directory: str
    sections_directory: str
    whole_brain_directory: str
    whole_brain_raw_file: str
    gene_directory: str
    gene_file: str
    ccf_ontology_directory: str
    taxonomy: str = TAXONOMY
    taxonomy_id: str = TAXONOMY_ID
    taxonomy_name: str = TAXONOMY_NAME
    taxonomy_directory: str = TAXONOMY_DIRECTORY


MERFISH_C57BL6J_638850 = MerfishDataset(
    key="MERFISH-C57BL6J-638850",
    metadata_directories=(
        "MERFISH-C57BL6J-638850",
        "MERFISH-C57BL6J-638850-CCF",
    ),
    parcellation_file="cell_metadata_with_parcellation_annotation",
    cluster_metadata_directory="MERFISH-C57BL6J-638850",
    cluster_metadata_file="cell_metadata_with_cluster_annotation",
    ccf_directory="MERFISH-C57BL6J-638850-CCF",
    sections_directory="MERFISH-C57BL6J-638850-sections",
    whole_brain_directory="MERFISH-C57BL6J-638850",
    whole_brain_raw_file="C57BL6J-638850/raw",
    gene_directory="MERFISH-C57BL6J-638850",
    gene_file="gene",
    ccf_ontology_directory="Allen-CCF-2020",
)

DATASETS: dict[str, MerfishDataset] = {
    MERFISH_C57BL6J_638850.key: MERFISH_C57BL6J_638850,
}


def get_dataset(key: str) -> MerfishDataset:
    if key not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise SystemExit(f"Unknown --dataset '{key}'. Choose from: {known}.")
    return DATASETS[key]
