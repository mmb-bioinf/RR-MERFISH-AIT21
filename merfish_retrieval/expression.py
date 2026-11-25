"""Raw MERFISH expression access. Log-normalized matrices are never used."""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from merfish_retrieval.datasets import MerfishDataset


@dataclass(frozen=True)
class ExpressionSource:
    directory: str
    file_name: str
    kind: str


class ExpressionSession:
    """Keep a whole-brain raw matrix open across sections; close section files."""

    def __init__(self, cache) -> None:
        self.cache = cache
        self._shared: dict[tuple[str, str], ad.AnnData] = {}

    def open(self, source: ExpressionSource) -> tuple[ad.AnnData, bool]:
        """Return ``(adata, caller_must_close)``."""
        key = (source.directory, source.file_name)
        if source.kind == "whole_brain_raw":
            cached = self._shared.get(key)
            if cached is None:
                cached = _read_backed(self.cache, source)
                self._shared[key] = cached
            return cached, False
        return _read_backed(self.cache, source), True

    def close(self) -> None:
        for backed in self._shared.values():
            close_backed(backed)
        self._shared.clear()


def close_backed(adata: ad.AnnData) -> None:
    file_obj = getattr(adata, "file", None)
    if file_obj is not None:
        file_obj.close()


def _read_backed(cache, source: ExpressionSource) -> ad.AnnData:
    path = cache.get_file_path(directory=source.directory, file_name=source.file_name)
    return ad.read_h5ad(path, backed="r")


def _listed_files(cache, directory: str) -> set[str]:
    files = cache.list_expression_matrix_files(directory)
    return set(files)


def resolve_raw_expression_source(
    cache,
    section_id: str,
    dataset: MerfishDataset,
) -> ExpressionSource:
    """Prefer the section raw matrix. Fall back to the whole-brain raw matrix.

    A section ``/log2`` file is never substituted for measured counts.
    """
    section_files = _listed_files(cache, dataset.sections_directory)
    raw_key = f"{section_id}/raw"
    log_key = f"{section_id}/log2"
    if raw_key in section_files:
        return ExpressionSource(
            directory=dataset.sections_directory,
            file_name=raw_key,
            kind="section_raw",
        )
    whole_files = _listed_files(cache, dataset.whole_brain_directory)
    if dataset.whole_brain_raw_file in whole_files:
        return ExpressionSource(
            directory=dataset.whole_brain_directory,
            file_name=dataset.whole_brain_raw_file,
            kind="whole_brain_raw",
        )
    if log_key in section_files:
        raise RuntimeError(
            f"Section {section_id} exposes {log_key} but no raw counts. "
            "Refusing to substitute log-normalized expression."
        )
    raise RuntimeError(
        f"No raw expression file for section {section_id}. "
        f"Looked for {dataset.sections_directory}:{raw_key} and "
        f"{dataset.whole_brain_directory}:{dataset.whole_brain_raw_file}."
    )


def subset_expression_by_cell_ids(
    adata: ad.AnnData,
    cell_ids,
    section_id: str | None = None,
    annotated_ids=None,
    allow_missing: bool = False,
) -> tuple[ad.AnnData, dict]:
    """Materialize selected cells in ``cell_ids`` order, independent of row order.

    Missing expression fails unless ``allow_missing`` is set. Allowed gaps are
    returned in the report and are not dropped silently.
    """
    wanted = pd.Index(pd.Index(cell_ids).map(str))
    if not wanted.is_unique:
        raise RuntimeError("Selected cell IDs are not unique.")
    names = pd.Index(adata.obs_names.map(str))
    if not names.is_unique:
        raise RuntimeError("Expression matrix has duplicate cell IDs.")
    missing = wanted.difference(names)
    if len(missing) and not allow_missing:
        examples = ", ".join(map(str, list(missing[:5])))
        raise RuntimeError(
            f"{len(missing)} selected cells have no expression row "
            f"(examples: {examples}). Missing expression fails retrieval. "
            "Pass --allow-missing-expression to drop and record them."
        )
    present = wanted[~wanted.isin(missing)]
    if len(present) == 0:
        raise RuntimeError("No selected cells were found in the expression matrix.")
    indexer = names.get_indexer(present)
    if np.any(indexer < 0):
        raise RuntimeError("Failed to align selected cell IDs to the expression matrix.")

    is_backed = bool(getattr(adata, "isbacked", False) or getattr(adata, "filename", None))
    subset = adata[indexer].to_memory() if is_backed else adata[indexer].copy()
    subset.obs_names = present
    if not sparse.issparse(subset.X):
        subset.X = sparse.csr_matrix(subset.X)
    else:
        subset.X = subset.X.tocsr()

    universe = wanted if annotated_ids is None else pd.Index(pd.Index(annotated_ids).map(str))
    if section_id is not None and "brain_section_label" in adata.obs.columns:
        in_section = adata.obs["brain_section_label"].astype(str).to_numpy() == str(section_id)
        section_names = names[in_section]
        expression_only = section_names.difference(universe)
        n_expression_rows = int(in_section.sum())
    else:
        expression_only = names.difference(universe)
        n_expression_rows = int(adata.n_obs)

    report = {
        "n_selected": int(len(wanted)),
        "n_missing_expression": int(len(missing)),
        "missing_expression_ids": [str(cell_id) for cell_id in missing],
        "n_expression_rows": n_expression_rows,
        "n_expression_only": int(len(expression_only)),
    }
    return subset, report


def control_feature_mask(
    var_names,
    gene_metadata: pd.DataFrame | None = None,
) -> np.ndarray:
    """Identify blank barcode controls from official gene identifiers or symbols."""
    ids = pd.Index(var_names).astype(str)
    mask = np.asarray(ids.str.startswith("Blank"), dtype=bool)
    if gene_metadata is not None and "gene_symbol" in gene_metadata.columns:
        symbols = gene_metadata["gene_symbol"].reindex(ids).astype("string").fillna("")
        mask = mask | np.asarray(symbols.str.startswith("Blank"), dtype=bool)
    return mask


def drop_control_features(
    adata: ad.AnnData,
    gene_metadata: pd.DataFrame | None = None,
) -> tuple[ad.AnnData, list[str]]:
    """Remove blank controls from ``X`` and keep every measured biological gene."""
    mask = control_feature_mask(adata.var_names, gene_metadata)
    removed = sorted(adata.var_names[mask].astype(str).tolist())
    kept = adata[:, ~mask].copy()
    if not sparse.issparse(kept.X):
        kept.X = sparse.csr_matrix(kept.X)
    else:
        kept.X = kept.X.tocsr()
    return kept, removed


def ensure_raw_counts(adata: ad.AnnData) -> None:
    """Require nonnegative integer-like values and store them as sparse int32."""
    matrix = adata.X
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()
    if matrix.nnz and matrix.data.min() < -1e-5:
        raise RuntimeError(
            "Expression contains negative values. Refusing log-normalized or imputed data."
        )
    if matrix.nnz and not np.all(np.abs(matrix.data - np.rint(matrix.data)) <= 1e-5):
        raise RuntimeError(
            "Expression values are not integer-like raw counts. "
            "Refusing log-normalized or imputed data."
        )
    if matrix.nnz:
        data = np.rint(matrix.data).astype(np.int32)
        matrix = sparse.csr_matrix(
            (data, matrix.indices.copy(), matrix.indptr.copy()),
            shape=matrix.shape,
            dtype=np.int32,
        )
    else:
        matrix = sparse.csr_matrix(matrix.shape, dtype=np.int32)
    adata.X = matrix


def load_section_expression(
    cache,
    section_id: str,
    cell_ids,
    gene_metadata: pd.DataFrame | None,
    dataset: MerfishDataset,
    session: ExpressionSession | None = None,
    annotated_ids=None,
    allow_missing: bool = False,
) -> tuple[ad.AnnData, dict]:
    """Load raw counts for one section's selected cells and drop blank controls."""
    source = resolve_raw_expression_source(cache, section_id, dataset)
    owns_session = session is None
    if session is None:
        session = ExpressionSession(cache)
    backed, caller_closes = session.open(source)
    try:
        subset, report = subset_expression_by_cell_ids(
            backed,
            cell_ids,
            section_id=section_id,
            annotated_ids=annotated_ids,
            allow_missing=allow_missing,
        )
        subset, removed = drop_control_features(subset, gene_metadata)
        ensure_raw_counts(subset)
    finally:
        if caller_closes:
            close_backed(backed)
        if owns_session:
            session.close()
    report["source"] = {
        "directory": source.directory,
        "file_name": source.file_name,
        "kind": source.kind,
    }
    report["removed_control_feature_ids"] = removed
    return subset, report
