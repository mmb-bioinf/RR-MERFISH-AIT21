#!/usr/bin/env python3
"""Retrieve Allen MERFISH sections with original AIT21 labels.

One physical section is written to one AnnData file. Expression stays raw
counts in section coordinates converted to micrometers. Grid binning is left
to ``snmf.pp.compress_to_grid``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from merfish_retrieval.coordinates import verify_section_coordinate_units
from merfish_retrieval.datasets import REGION_ALIASES_AIT21, get_dataset
from merfish_retrieval.export import (
    attach_metadata_and_coordinates,
    json_safe,
    package_version,
    report_reference_compatibility,
    validate_section,
    write_outputs,
)
from merfish_retrieval.expression import ExpressionSession, load_section_expression
from merfish_retrieval.metadata import (
    load_gene_metadata,
    load_metadata,
    load_region_descriptions,
)
from merfish_retrieval.regions import (
    choose_sections,
    eligibility_report,
    list_regions,
    rank_sections,
    region_column,
    resolve_region,
    select_section_cells,
)

try:
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "abc_atlas_access is required. Install with:\n"
        "  pip install git+https://github.com/alleninstitute/abc_atlas_access.git"
    ) from exc


DEFAULT_N_SECTIONS = 5
DEFAULT_MIN_CELLS = 5000
DEFAULT_COVERAGE_WARN = 0.95


def initialize_cache(cache_dir: str | Path, manifest: str | None) -> AbcProjectCache:
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    cache = AbcProjectCache.from_cache_dir(cache_dir=cache_path)
    if manifest:
        cache.load_manifest(manifest)
    print(f"ABC Atlas manifest: {cache.current_manifest}")
    return cache


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve segmented Allen MERFISH sections with original AIT21 "
            "(CCN20230722) labels, optionally cropped to a CCF anatomical region."
        )
    )
    parser.add_argument(
        "--dataset",
        default="MERFISH-C57BL6J-638850",
        help="MERFISH dataset adapter (default: MERFISH-C57BL6J-638850).",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Anatomical region name or acronym (case-insensitive).",
    )
    parser.add_argument(
        "--region-level",
        default="division",
        choices=["division", "structure", "substructure"],
        help="CCF parcellation level used to resolve --region (default: division).",
    )
    parser.add_argument(
        "--list-regions",
        action="store_true",
        help="List anatomical regions with cell and section counts, then exit.",
    )
    parser.add_argument(
        "--list-sections",
        action="store_true",
        help="List physical sections for --region, then exit.",
    )
    parser.add_argument(
        "--n-sections",
        type=int,
        default=DEFAULT_N_SECTIONS,
        help=f"Automatically selected sections (default: {DEFAULT_N_SECTIONS}).",
    )
    parser.add_argument(
        "--min-cells-per-section",
        type=int,
        default=DEFAULT_MIN_CELLS,
        help=(
            "Minimum annotated cells in the selected region "
            f"(default: {DEFAULT_MIN_CELLS})."
        ),
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=None,
        help="Official section IDs. Overrides --n-sections; still checks eligibility.",
    )
    parser.add_argument(
        "--scope",
        default="region",
        choices=["region", "full-section"],
        help="region crops to the selection; full-section keeps the whole section.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional AIT21 reference .h5ad used only for compatibility reports.",
    )
    parser.add_argument(
        "--label-key",
        default="subclass",
        help="Taxonomy level used for reporting (default: subclass).",
    )
    parser.add_argument(
        "--coverage-warn-threshold",
        type=float,
        default=DEFAULT_COVERAGE_WARN,
        help=f"Warn when reference coverage is below this (default: {DEFAULT_COVERAGE_WARN}).",
    )
    parser.add_argument(
        "--accept-unverified-reference",
        action="store_true",
        help="Proceed when the reference has no recognized AIT21 provenance.",
    )
    parser.add_argument(
        "--cache-dir",
        default="./abc_atlas_cache",
        help="AbcProjectCache directory (default: ./abc_atlas_cache).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional Allen manifest name, e.g. releases/20260711/manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-section .h5ad files and summaries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write a sanity-check scatter plot per section, in micrometers.",
    )
    parser.add_argument(
        "--allow-missing-expression",
        action="store_true",
        help=(
            "Drop selected cells that have no raw-count row and record their IDs. "
            "By default any missing expression fails the run."
        ),
    )
    return parser.parse_args(argv)


def _print_regions(regions: pd.DataFrame, level: str) -> None:
    print(f"Available {level} regions:")
    print()
    for _, row in regions.iterrows():
        print(
            f"  {row['label']:16s}  {row['description']:28s}  "
            f"{int(row['n_cells']):8,d} cells  {int(row['n_sections']):3d} sections"
        )


def _print_sections(ranked: pd.DataFrame, min_cells: int) -> None:
    print(f"Sections with cells in the selected region (threshold {min_cells} cells):")
    print()
    for _, row in ranked.iterrows():
        flag = "eligible" if bool(row["eligible"]) else "below-threshold"
        print(
            f"  {row['rank']:2d}. {row['brain_section_label']}  "
            f"z={float(row['z_section']):.1f} mm  "
            f"regional={int(row['n_regional_cells']):,}  "
            f"section={int(row['n_section_cells']):,}  {flag}"
        )


def _print_exclusions(reasons: dict[str, int]) -> None:
    print("Metadata eligibility:")
    print(f"  input cells:    {reasons['n_input']:,}")
    print(f"  eligible cells: {reasons['n_eligible']:,}")
    print(f"  excluded cells: {reasons['n_excluded']:,}")
    for key, count in reasons.items():
        if key.startswith("missing_") and count:
            print(f"    {key}: {count:,}")


def _subclass_table(
    adata,
    reference_labels: set[str] | None,
) -> pd.DataFrame:
    section_id = str(adata.obs["brain_section_label"].astype(str).iloc[0])
    counts = adata.obs["subclass"].astype(str).value_counts()
    frame = counts.rename_axis("subclass").reset_index(name="n_cells")
    frame.insert(0, "brain_section_label", section_id)
    if reference_labels is not None:
        frame["in_reference"] = frame["subclass"].isin(reference_labels)
    return frame.sort_values(["brain_section_label", "subclass"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_cells_per_section < 0:
        raise SystemExit("--min-cells-per-section must be >= 0.")
    if args.n_sections < 1:
        raise SystemExit("--n-sections must be >= 1.")
    if not args.list_regions and not args.region:
        raise SystemExit("Provide --region, or use --list-regions.")
    if not args.list_regions and not args.list_sections and not args.output_dir:
        raise SystemExit("Provide --output-dir, or use a list command.")

    dataset = get_dataset(args.dataset)
    cache = initialize_cache(args.cache_dir, args.manifest)
    metadata = load_metadata(cache, dataset)
    eligible_mask, exclusion_counts = eligibility_report(metadata)
    _print_exclusions(exclusion_counts)
    eligible = metadata.loc[eligible_mask].copy()
    descriptions = load_region_descriptions(cache, dataset)
    regions = list_regions(eligible, args.region_level, descriptions)

    if args.list_regions:
        _print_regions(regions, args.region_level)
        return 0

    region = resolve_region(args.region, regions, REGION_ALIASES_AIT21)
    print(f"Selected region: {region} ({args.region_level})")
    coordinate_info = verify_section_coordinate_units(metadata)
    print(
        "Section coordinates: mm "
        f"(median z spacing {coordinate_info['observed_median_z_spacing_mm']:.3f} mm); "
        "obsm['spatial'] will be stored in µm."
    )
    ranked = rank_sections(
        eligible,
        region,
        region_level=args.region_level,
        min_cells_per_section=args.min_cells_per_section,
    )
    if args.list_sections:
        if ranked.empty:
            raise SystemExit(f"No sections contain cells from region {region}.")
        _print_sections(ranked, args.min_cells_per_section)
        return 0

    chosen, shortfall = choose_sections(
        ranked,
        n_sections=args.n_sections,
        explicit=args.sections,
        min_cells_per_section=args.min_cells_per_section,
    )
    if shortfall:
        print(shortfall)
    print("Sections to export:")
    for _, row in chosen.iterrows():
        print(
            f"  {row['brain_section_label']}  "
            f"z={float(row['z_section']):.1f} mm  "
            f"regional={int(row['n_regional_cells']):,}"
        )

    output_dir = Path(args.output_dir)
    for _, row in chosen.iterrows():
        path = output_dir / f"{row['brain_section_label']}.h5ad"
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists: {path}. Pass --overwrite to replace it.")

    gene_metadata = load_gene_metadata(cache, dataset)
    column = region_column(args.region_level)
    retrieval_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    versions = {
        "abc_atlas_access": package_version("abc_atlas_access"),
        "anndata": package_version("anndata"),
        "pandas": package_version("pandas"),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
    }
    shared_info = {
        "taxonomy": dataset.taxonomy,
        "taxonomy_id": dataset.taxonomy_id,
        "taxonomy_name": dataset.taxonomy_name,
        "taxonomy_directory": dataset.taxonomy_directory,
        "dataset": dataset.key,
        "region": region,
        "region_query": args.region,
        "region_level": args.region_level,
        "scope": args.scope,
        "label_key": args.label_key,
        "expression_scale": "raw_counts",
        "allen_manifest": str(cache.current_manifest),
        "retrieval_date": retrieval_date,
        "package_versions": versions,
        "exclusion_counts": exclusion_counts,
        "min_cells_per_section": args.min_cells_per_section,
    }

    exported = []
    summary_rows = []
    count_frames = []
    session = ExpressionSession(cache)
    try:
        for _, row in chosen.iterrows():
            section_id = str(row["brain_section_label"])
            selected = select_section_cells(
                eligible,
                section_id,
                region,
                region_level=args.region_level,
                scope=args.scope,
            )
            print(
                f"Loading raw counts for {section_id} "
                f"({len(selected):,} cells, scope={args.scope})"
            )
            expression, expression_report = load_section_expression(
                cache,
                section_id,
                selected.index,
                gene_metadata,
                dataset,
                session=session,
                annotated_ids=metadata.index,
                allow_missing=args.allow_missing_expression,
            )
            if expression_report["n_missing_expression"]:
                print(
                    f"  excluded {expression_report['n_missing_expression']:,} "
                    "cells with no raw expression row"
                )
            info = dict(shared_info)
            info.update(
                {
                    "brain_section_label": section_id,
                    "donor_label": str(selected["donor_label"].iloc[0]),
                    "z_section_mm": float(row["z_section"]),
                    "n_cells_regional": int(row["n_regional_cells"]),
                    "n_cells_exported": int(expression.n_obs),
                    "expression_source": expression_report["source"],
                    "removed_control_feature_ids": expression_report[
                        "removed_control_feature_ids"
                    ],
                    "n_expression_only_cells": expression_report["n_expression_only"],
                    "n_expression_rows": expression_report["n_expression_rows"],
                    "n_missing_expression": expression_report["n_missing_expression"],
                    "missing_expression_ids": expression_report["missing_expression_ids"],
                }
            )
            adata = attach_metadata_and_coordinates(
                expression,
                selected,
                coordinate_info,
                info,
                gene_metadata=gene_metadata,
            )
            reference_report = None
            reference_labels = None
            if args.reference:
                reference_report = report_reference_compatibility(
                    adata,
                    args.reference,
                    label_key=args.label_key,
                    coverage_warn_threshold=args.coverage_warn_threshold,
                    accept_unverified=args.accept_unverified_reference,
                )
                reference_labels = set(reference_report["reference_labels"])
                stored = dict(reference_report)
                stored.pop("reference_labels", None)
                adata.uns["retrieval_info"]["reference_compatibility"] = json_safe(stored)
            validate_section(
                adata,
                section_id=section_id,
                label_key=args.label_key,
                region_label=None if args.scope == "full-section" else region,
                region_column=None if args.scope == "full-section" else column,
            )
            exported.append(adata)
            count_frames.append(_subclass_table(adata, reference_labels))
            summary = {
                "brain_section_label": section_id,
                "z_section_mm": float(row["z_section"]),
                "donor_label": info["donor_label"],
                "n_cells_regional": int(row["n_regional_cells"]),
                "n_cells_exported": int(adata.n_obs),
                "n_cells_in_section": int(row["n_section_cells"]),
                "n_subclasses": int(adata.obs["subclass"].nunique()),
                "n_genes": int(adata.n_vars),
                "n_expression_only_cells": expression_report["n_expression_only"],
                "n_missing_expression": expression_report["n_missing_expression"],
                "n_removed_control_features": len(
                    expression_report["removed_control_feature_ids"]
                ),
                "rank": int(row["rank"]),
            }
            if reference_report is not None:
                summary["reference_coverage"] = reference_report["coverage"]
                summary["n_shared_subclasses"] = reference_report["n_shared_labels"]
                summary["n_merfish_only_subclasses"] = reference_report["n_merfish_only_labels"]
                summary["n_reference_only_subclasses"] = reference_report[
                    "n_reference_only_labels"
                ]
                overlap = reference_report.get("gene_overlap") or {}
                summary["n_genes_shared_with_reference"] = overlap.get("n_shared_genes")
            summary_rows.append(summary)
            print(
                f"  exported {adata.n_obs:,} cells, {adata.n_vars:,} genes, "
                f"{summary['n_subclasses']} subclasses"
            )
            if expression_report["n_expression_only"]:
                print(
                    "  expression rows without published annotations: "
                    f"{expression_report['n_expression_only']:,}"
                )
    finally:
        session.close()

    section_summary = pd.DataFrame(summary_rows)
    subclass_counts = (
        pd.concat(count_frames, ignore_index=True)
        if count_frames
        else pd.DataFrame(columns=["brain_section_label", "subclass", "n_cells"])
    )
    manifest = dict(shared_info)
    manifest["n_sections_requested"] = None if args.sections else args.n_sections
    manifest["explicit_sections"] = args.sections
    manifest["n_sections_exported"] = len(exported)
    manifest["shortfall"] = shortfall
    manifest["sections"] = summary_rows
    manifest["coordinate_provenance"] = coordinate_info
    write_outputs(
        output_dir,
        exported,
        section_summary,
        subclass_counts,
        manifest,
        overwrite=args.overwrite,
        make_plots=args.plot,
    )
    print()
    print(f"Exported {len(exported)} section(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
