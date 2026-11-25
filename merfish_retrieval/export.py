"""Assemble, validate, and write one AnnData file per physical section."""

from __future__ import annotations

import importlib.metadata
import json
import os
import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from merfish_retrieval.coordinates import section_coordinates_to_um
from merfish_retrieval.datasets import (
    COMPATIBLE_TAXONOMY_TOKENS,
    INCOMPATIBLE_TAXONOMY_TOKENS,
)

OBS_COLUMNS = (
    "brain_section_label",
    "donor_label",
    "donor_sex",
    "donor_genotype",
    "class",
    "subclass",
    "supertype",
    "cluster",
    "cluster_alias",
    "average_correlation_score",
    "neurotransmitter",
    "feature_matrix_label",
    "parcellation_index",
    "parcellation_organ",
    "parcellation_category",
    "parcellation_division",
    "parcellation_structure",
    "parcellation_substructure",
    "x_section",
    "y_section",
    "z_section",
    "x_reconstructed",
    "y_reconstructed",
    "z_reconstructed",
    "x_ccf",
    "y_ccf",
    "z_ccf",
)
HIERARCHY_COLUMNS = ("class", "subclass", "supertype", "cluster")
PROVENANCE_KEYS = (
    "taxonomy",
    "taxonomy_id",
    "taxonomy_name",
    "taxonomy_directory",
)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _blank_labels(series: pd.Series) -> pd.Series:
    as_str = series.astype("string").str.strip()
    return as_str.isna() | as_str.eq("") | as_str.str.lower().isin(["nan", "none", "null", "<na>"])


def attach_metadata_and_coordinates(
    expression: ad.AnnData,
    metadata: pd.DataFrame,
    coordinate_info: dict,
    retrieval_info: dict,
    gene_metadata: pd.DataFrame | None = None,
) -> ad.AnnData:
    """Align metadata by cell ID and attach section coordinates in micrometers."""
    meta = metadata.reindex(expression.obs_names)
    if meta["subclass"].isna().any():
        n_missing = int(meta["subclass"].isna().sum())
        raise RuntimeError(
            f"Metadata is missing for {n_missing} expression cells after ID alignment."
        )

    obs = pd.DataFrame(index=pd.Index(expression.obs_names.astype(str), name=None))
    for column in OBS_COLUMNS:
        if column in meta.columns:
            obs[column] = meta[column].to_numpy()
    for column in (*HIERARCHY_COLUMNS, "brain_section_label"):
        if column not in obs.columns:
            raise RuntimeError(f"Aligned metadata lacks {column}.")
        if _blank_labels(obs[column]).any():
            raise RuntimeError(f"Column {column} contains missing or blank labels.")
        obs[column] = pd.Categorical(obs[column].astype(str))

    var = expression.var.copy() if expression.var is not None else pd.DataFrame(index=expression.var_names)
    if gene_metadata is not None:
        aligned = gene_metadata.reindex(expression.var_names)
        for column in gene_metadata.columns:
            if column not in var.columns:
                var[column] = aligned[column].to_numpy()
    var["gene_identifier"] = pd.Index(expression.var_names).astype(str)
    var.index = pd.Index(expression.var_names.astype(str), name=None)

    matrix = expression.X
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    else:
        matrix = matrix.tocsr()

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obs_names = pd.Index(expression.obs_names.astype(str), name=None)
    adata.var_names = pd.Index(expression.var_names.astype(str), name=None)
    adata.obsm["spatial"] = section_coordinates_to_um(
        meta,
        conversion_factor=float(coordinate_info["conversion_factor_to_um"]),
    )
    info = json_safe(retrieval_info)
    info["coordinate_provenance"] = json_safe(coordinate_info)
    adata.uns["retrieval_info"] = info
    return adata


def _taxonomy_tokens(info: dict) -> set[str]:
    tokens: set[str] = set()
    for key in PROVENANCE_KEYS:
        value = info.get(key)
        if value is not None and str(value).strip():
            tokens.add(str(value).strip())
    return tokens


def _incompatible(tokens: set[str]) -> bool:
    if tokens & INCOMPATIBLE_TAXONOMY_TOKENS:
        return True
    return any("AIT33" in token or "CCN20251031" in token for token in tokens)


def _compatible(tokens: set[str]) -> bool:
    if tokens & COMPATIBLE_TAXONOMY_TOKENS:
        return True
    return any(
        "AIT21" in token or "CCN20230722" in token or token == "WMB-taxonomy"
        for token in tokens
    )


def report_reference_compatibility(
    adata: ad.AnnData,
    reference: str | Path | ad.AnnData,
    label_key: str = "subclass",
    coverage_warn_threshold: float = 0.95,
    accept_unverified: bool = False,
) -> dict:
    """Compare published labels with an AIT21 reference without relabeling cells.

    Cells outside the reference are retained. Coverage is a report, not a filter.
    """
    owns_file = False
    if isinstance(reference, ad.AnnData):
        ref = reference
    else:
        ref = ad.read_h5ad(reference, backed="r")
        owns_file = True
    try:
        info = {}
        if "reference_info" in ref.uns:
            info = dict(ref.uns["reference_info"])
        tokens = _taxonomy_tokens(info)
        if _incompatible(tokens):
            raise SystemExit(
                "Reference taxonomy is incompatible with MERFISH AIT21 labels "
                f"({', '.join(sorted(tokens))}). Provide an AIT21 / CCN20230722 reference."
            )
        if not tokens:
            if not accept_unverified:
                raise SystemExit(
                    "Reference has no taxonomy provenance in uns['reference_info']. "
                    "Refusing to assume AIT21. Confirm the file uses original WMB "
                    "AIT21 (CCN20230722) labels and pass --accept-unverified-reference."
                )
            provenance = "unverified"
        elif not _compatible(tokens):
            if not accept_unverified:
                raise SystemExit(
                    "Reference provenance is not a recognized AIT21 taxonomy "
                    f"({', '.join(sorted(tokens))}). Pass --accept-unverified-reference "
                    "after confirming the labels are original WMB / AIT21 labels."
                )
            provenance = "unverified"
        else:
            provenance = "ait21"

        if label_key not in ref.obs.columns:
            raise SystemExit(f"Reference is missing obs[{label_key!r}].")
        if label_key not in adata.obs.columns:
            raise RuntimeError(f"Section is missing obs[{label_key!r}].")

        reference_labels = set(ref.obs[label_key].astype(str).unique())
        labels = adata.obs[label_key].astype(str)
        covered = labels.isin(reference_labels)
        coverage = float(covered.mean()) if len(labels) else float("nan")
        observed = set(labels.unique())
        shared = sorted(observed & reference_labels)
        merfish_only = sorted(observed - reference_labels)
        reference_only = sorted(reference_labels - observed)

        gene_overlap = None
        our_genes = pd.Index(adata.var_names.astype(str))
        ref_genes = pd.Index(ref.var_names.astype(str))
        if len(our_genes) and len(ref_genes):
            our_ens = float(our_genes.str.startswith("ENS").mean())
            ref_ens = float(ref_genes.str.startswith("ENS").mean())
            if our_ens > 0.5 and ref_ens > 0.5:
                gene_overlap = {
                    "n_shared_genes": int(len(our_genes.intersection(ref_genes))),
                    "n_merfish_genes": int(len(our_genes)),
                    "n_reference_genes": int(len(ref_genes)),
                }

        if coverage < coverage_warn_threshold:
            warnings.warn(
                f"Reference coverage at {label_key!r} is {coverage:.1%}, "
                f"below {coverage_warn_threshold:.0%}. All cells are retained.",
                stacklevel=2,
            )
        return {
            "provenance": provenance,
            "label_key": label_key,
            "coverage": coverage,
            "n_shared_labels": len(shared),
            "n_merfish_only_labels": len(merfish_only),
            "n_reference_only_labels": len(reference_only),
            "shared_labels": shared,
            "merfish_only_labels": merfish_only,
            "reference_only_labels": reference_only,
            "reference_labels": sorted(reference_labels),
            "gene_overlap": gene_overlap,
            "coverage_warn_threshold": coverage_warn_threshold,
        }
    finally:
        if owns_file:
            file_obj = getattr(ref, "file", None)
            if file_obj is not None:
                file_obj.close()


def validate_section(
    adata: ad.AnnData,
    *,
    section_id: str,
    label_key: str,
    region_label: str | None = None,
    region_column: str | None = None,
) -> None:
    """Acceptance checks for one exported physical section."""
    if adata.n_obs == 0:
        raise RuntimeError(f"Section {section_id} has no cells to export.")
    if not adata.obs_names.is_unique:
        raise RuntimeError("obs_names must be unique.")
    if not adata.var_names.is_unique:
        raise RuntimeError("var_names must be unique.")
    if label_key not in adata.obs.columns:
        raise RuntimeError(f"obs lacks {label_key}.")
    if _blank_labels(adata.obs[label_key]).any():
        raise RuntimeError(f"obs[{label_key!r}] contains missing or blank labels.")
    sections = adata.obs["brain_section_label"].astype(str).unique()
    if len(sections) != 1 or sections[0] != str(section_id):
        raise RuntimeError(
            f"Expected one physical section {section_id}, found {list(sections)}."
        )
    if region_label is not None:
        if region_column is None or region_column not in adata.obs.columns:
            raise RuntimeError("Region crop validation requires the parcellation column.")
        bad = adata.obs[region_column].astype(str) != str(region_label)
        if bad.any():
            raise RuntimeError(
                f"{int(bad.sum())} cells do not match region '{region_label}'."
            )
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    if spatial.shape != (adata.n_obs, 2) or not np.isfinite(spatial).all():
        raise RuntimeError("obsm['spatial'] must be a finite (n_obs, 2) array in µm.")
    if not sparse.issparse(adata.X):
        raise RuntimeError("Expression matrix must remain sparse.")
    if adata.n_obs and adata.X.min() < 0:
        raise RuntimeError("Expression contains negative values.")
    blanks = [name for name in adata.var_names.astype(str) if name.startswith("Blank")]
    if blanks:
        raise RuntimeError(f"Blank controls remain in X: {blanks[:5]}")


def plot_section(adata: ad.AnnData, path: str | Path) -> None:
    """Scatter section coordinates in micrometers with equal aspect."""
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    section_id = str(adata.obs["brain_section_label"].astype(str).iloc[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=0.15, c="black", linewidths=0)
    ax.set_aspect("equal")
    # Allen section plots use image orientation, with y increasing downward.
    ax.set_ylim(float(np.max(coords[:, 1])), float(np.min(coords[:, 1])))
    ax.set_xlabel("x section (µm)")
    ax.set_ylabel("y section (µm)")
    ax.set_title(section_id)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _atomic_replace(tmp: Path, path: Path) -> None:
    os.replace(tmp, path)


def write_outputs(
    output_dir: str | Path,
    sections: list[ad.AnnData],
    section_summary: pd.DataFrame,
    subclass_counts: pd.DataFrame,
    manifest: dict,
    overwrite: bool = False,
    make_plots: bool = False,
) -> Path:
    """Write one h5ad per section plus summary tables. Writes are atomic."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    planned = []
    for adata in sections:
        section_id = str(adata.obs["brain_section_label"].astype(str).iloc[0])
        path = output_path / f"{section_id}.h5ad"
        if path.exists() and not overwrite:
            raise SystemExit(f"Output exists: {path}. Pass --overwrite to replace it.")
        planned.append((path, adata))

    for path, adata in planned:
        tmp = path.with_suffix(path.suffix + ".partial")
        try:
            adata.write_h5ad(tmp, compression="gzip")
            _atomic_replace(tmp, path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        print(f"Wrote {path}")
        if make_plots:
            plot_path = output_path / "plots" / f"{path.stem}.png"
            plot_section(adata, plot_path)
            print(f"Wrote {plot_path}")

    summary_path = output_path / "section_summary.csv"
    counts_path = output_path / "subclass_counts.csv"
    manifest_path = output_path / "retrieval_manifest.json"
    for frame, path in ((section_summary, summary_path), (subclass_counts, counts_path)):
        tmp = path.with_suffix(path.suffix + ".partial")
        frame.to_csv(tmp, index=False)
        _atomic_replace(tmp, path)
        print(f"Wrote {path}")
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    tmp_manifest.write_text(json.dumps(json_safe(manifest), indent=2) + "\n")
    _atomic_replace(tmp_manifest, manifest_path)
    print(f"Wrote {manifest_path}")
    return output_path
