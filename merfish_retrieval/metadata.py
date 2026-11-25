"""Load Allen MERFISH cell metadata without touching expression matrices."""

from __future__ import annotations

import pandas as pd

from merfish_retrieval.datasets import (
    DIVISION_DESCRIPTIONS,
    MerfishDataset,
)


def _directories(cache) -> set[str]:
    directories = cache.list_directories
    if callable(directories):
        directories = directories()
    return set(directories)


def _metadata_files(cache, directory: str) -> set[str]:
    return set(cache.list_metadata_files(directory))


def _rename_section_coordinates(cells: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "x_section" not in cells.columns and "x" in cells.columns:
        rename["x"] = "x_section"
    if "y_section" not in cells.columns and "y" in cells.columns:
        rename["y"] = "y_section"
    if "z_section" not in cells.columns and "z" in cells.columns:
        rename["z"] = "z_section"
    if rename:
        cells = cells.rename(columns=rename)
    return cells


def _finalize_metadata(cells: pd.DataFrame) -> pd.DataFrame:
    if "cell_label" in cells.columns:
        cells = cells.set_index("cell_label", verify_integrity=True)
    cells.index = cells.index.astype(str)
    cells.index.name = "cell_label"
    cells = _rename_section_coordinates(cells)
    for column in ("x_section", "y_section", "z_section"):
        if column in cells.columns:
            cells[column] = pd.to_numeric(cells[column], errors="coerce")
    required = [
        "brain_section_label",
        "class",
        "subclass",
        "supertype",
        "cluster",
        "x_section",
        "y_section",
        "z_section",
        "parcellation_division",
        "parcellation_structure",
        "parcellation_substructure",
    ]
    missing = [column for column in required if column not in cells.columns]
    if missing:
        raise RuntimeError(f"MERFISH metadata is missing required columns: {missing}")
    cells["brain_section_label"] = cells["brain_section_label"].astype(str)
    return cells


def _join_parcellation_metadata(cache, dataset: MerfishDataset) -> pd.DataFrame:
    """Reproduce the official metadata joins when the prejoined table is absent."""
    print(
        "Prejoined cell_metadata_with_parcellation_annotation is unavailable; "
        "joining cluster annotation, CCF coordinates, and parcellation membership."
    )
    cells = cache.get_metadata_dataframe(
        directory=dataset.cluster_metadata_directory,
        file_name=dataset.cluster_metadata_file,
        dtype={"cell_label": str},
        low_memory=False,
    )
    cells = _rename_section_coordinates(cells.set_index("cell_label", verify_integrity=True))

    reconstructed = cache.get_metadata_dataframe(
        directory=dataset.ccf_directory,
        file_name="reconstructed_coordinates",
        dtype={"cell_label": str},
    ).set_index("cell_label")
    reconstructed.index = reconstructed.index.astype(str)
    reconstructed = reconstructed.rename(
        columns={"x": "x_reconstructed", "y": "y_reconstructed", "z": "z_reconstructed"}
    )
    cells = cells.join(reconstructed, how="left")

    ccf = cache.get_metadata_dataframe(
        directory=dataset.ccf_directory,
        file_name="ccf_coordinates",
        dtype={"cell_label": str},
    ).set_index("cell_label")
    ccf.index = ccf.index.astype(str)
    ccf = ccf.drop(columns=["parcellation_index"], errors="ignore")
    ccf = ccf.rename(columns={"x": "x_ccf", "y": "y_ccf", "z": "z_ccf"})
    cells = cells.join(ccf, how="left")

    membership = cache.get_metadata_dataframe(
        directory=dataset.ccf_ontology_directory,
        file_name="parcellation_to_parcellation_term_membership_acronym",
    ).set_index("parcellation_index")
    membership.columns = [
        column if str(column).startswith("parcellation_") else f"parcellation_{column}"
        for column in membership.columns
    ]
    if "parcellation_index" not in cells.columns:
        raise RuntimeError("Joined CCF coordinates did not provide parcellation_index.")
    cells = cells.join(membership, on="parcellation_index", how="left")
    return _finalize_metadata(cells)


def load_metadata(cache, dataset: MerfishDataset) -> pd.DataFrame:
    """Load annotated MERFISH cells with CCF anatomical assignments.

    Uses the prejoined parcellation table when the manifest publishes it.
    The current Allen manifest stores that table under the CCF directory,
    not under the expression directory.
    """
    available = _directories(cache)
    for directory in dataset.metadata_directories:
        if directory not in available:
            continue
        files = _metadata_files(cache, directory)
        if dataset.parcellation_file not in files:
            continue
        print(f"Loading {directory}/{dataset.parcellation_file}")
        cells = cache.get_metadata_dataframe(
            directory=directory,
            file_name=dataset.parcellation_file,
            dtype={"cell_label": str},
            low_memory=False,
        )
        cells = _finalize_metadata(cells)
        print(f"Loaded MERFISH cell metadata: {len(cells):,} cells")
        return cells
    cells = _join_parcellation_metadata(cache, dataset)
    print(f"Loaded MERFISH cell metadata: {len(cells):,} cells")
    return cells


def load_gene_metadata(cache, dataset: MerfishDataset) -> pd.DataFrame:
    genes = cache.get_metadata_dataframe(
        directory=dataset.gene_directory,
        file_name=dataset.gene_file,
    )
    if "gene_identifier" not in genes.columns:
        raise RuntimeError(f"{dataset.gene_file} lacks gene_identifier.")
    genes = genes.set_index("gene_identifier")
    genes.index = genes.index.astype(str)
    return genes


def load_region_descriptions(cache, dataset: MerfishDataset) -> dict[str, str]:
    """Map anatomical acronyms to official names when the CCF term table exists."""
    descriptions = dict(DIVISION_DESCRIPTIONS)
    available = _directories(cache)
    if dataset.ccf_ontology_directory not in available:
        return descriptions
    files = _metadata_files(cache, dataset.ccf_ontology_directory)
    if "parcellation_term" not in files:
        return descriptions
    terms = cache.get_metadata_dataframe(
        directory=dataset.ccf_ontology_directory,
        file_name="parcellation_term",
    )
    if not {"acronym", "name"}.issubset(terms.columns):
        return descriptions
    grouped: dict[str, set[str]] = {}
    for acronym, name in zip(terms["acronym"], terms["name"]):
        grouped.setdefault(str(acronym), set()).add(str(name))
    for acronym, names in grouped.items():
        if len(names) == 1:
            descriptions[acronym] = next(iter(names))
    return descriptions
