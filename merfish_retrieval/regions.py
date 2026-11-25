"""Anatomical region resolution and physical-section ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from merfish_retrieval.datasets import REGION_LEVELS

_MISSING_REGION_TOKENS = {"", "nan", "none", "null", "unassigned", "<na>"}
LABEL_COLUMNS = ("class", "subclass", "supertype", "cluster")
REQUIRED_COORDINATE_COLUMNS = ("x_section", "y_section", "z_section")


def region_column(region_level: str) -> str:
    if region_level not in REGION_LEVELS:
        known = ", ".join(REGION_LEVELS)
        raise SystemExit(
            f"Unknown --region-level '{region_level}'. Choose from: {known}."
        )
    return REGION_LEVELS[region_level]


def _blank_mask(series: pd.Series) -> pd.Series:
    as_str = series.astype("string")
    stripped = as_str.str.strip()
    return as_str.isna() | stripped.eq("") | stripped.str.lower().isin(
        ["nan", "none", "null", "<na>"]
    )


def eligibility_report(metadata: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    """Mark cells with the metadata required for export.

    A cell can fail more than one check. ``n_excluded`` counts unique cells
    that fail any check. Cells that fail are omitted from later selection and
    the reason counts are reported; they are not dropped silently.
    """
    reasons: dict[str, int] = {}
    ok = pd.Series(True, index=metadata.index)
    required = ("brain_section_label", "donor_label", *LABEL_COLUMNS, *REQUIRED_COORDINATE_COLUMNS)
    missing_cols = [col for col in required if col not in metadata.columns]
    if missing_cols:
        raise RuntimeError(f"Metadata is missing required columns: {missing_cols}")
    for col in required:
        if col in REQUIRED_COORDINATE_COLUMNS:
            bad = ~np.isfinite(pd.to_numeric(metadata[col], errors="coerce"))
        else:
            bad = _blank_mask(metadata[col])
        reasons[f"missing_{col}"] = int(bad.sum())
        ok = ok & ~bad
    reasons["n_excluded"] = int((~ok).sum())
    reasons["n_input"] = int(len(metadata))
    reasons["n_eligible"] = int(ok.sum())
    return ok, reasons


def list_regions(
    metadata: pd.DataFrame,
    region_level: str = "division",
    descriptions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return regions present in metadata with cell and section counts."""
    column = region_column(region_level)
    if column not in metadata.columns:
        raise RuntimeError(f"Metadata lacks {column}")
    if "brain_section_label" not in metadata.columns:
        raise RuntimeError("Metadata lacks brain_section_label")

    labels = metadata[column].astype("string").str.strip()
    keep = labels.notna() & ~labels.str.lower().isin(_MISSING_REGION_TOKENS)
    frame = pd.DataFrame(
        {
            "label": labels[keep].astype(str),
            "brain_section_label": metadata.loc[keep, "brain_section_label"].astype(str),
        }
    )
    if frame.empty:
        return pd.DataFrame(columns=["label", "description", "n_cells", "n_sections"])

    counts = (
        frame.groupby("label", sort=False)
        .agg(n_cells=("label", "size"), n_sections=("brain_section_label", "nunique"))
        .reset_index()
    )
    desc_map = descriptions or {}
    counts["description"] = counts["label"].map(lambda lab: desc_map.get(str(lab), str(lab)))
    counts = counts.sort_values("label").reset_index(drop=True)
    return counts[["label", "description", "n_cells", "n_sections"]]


def resolve_region(
    region_query: str,
    available_regions: pd.DataFrame,
    region_aliases: dict[str, str] | None = None,
) -> str:
    """Resolve a region query to one official acronym.

    Exact case-insensitive matches on the acronym, the official description,
    and verified aliases are accepted. Substring hits are reported and rejected
    so ambiguous names are not guessed.
    """
    query = region_query.strip()
    if not query:
        raise SystemExit("Empty --region value.")

    aliases = region_aliases or {}
    labels = available_regions["label"].astype(str).tolist()
    descriptions = available_regions["description"].astype(str).tolist()
    label_by_lower = {lab.lower(): lab for lab in labels}
    desc_by_lower: dict[str, str] = {}
    duplicate_descriptions: set[str] = set()
    for lab, desc in zip(labels, descriptions):
        key = desc.lower()
        if key in desc_by_lower and desc_by_lower[key] != lab:
            duplicate_descriptions.add(key)
        else:
            desc_by_lower[key] = lab

    matches: list[str] = []
    alias_target = aliases.get(query.lower())
    if alias_target is not None and alias_target in labels:
        matches.append(alias_target)
    if query.lower() in label_by_lower:
        matches.append(label_by_lower[query.lower()])
    if query.lower() in desc_by_lower and query.lower() not in duplicate_descriptions:
        matches.append(desc_by_lower[query.lower()])

    unique: list[str] = []
    for match in matches:
        if match not in unique:
            unique.append(match)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        lines = [f"Region '{region_query}' matched multiple Allen regions:", ""]
        for i, lab in enumerate(unique, start=1):
            desc = available_regions.loc[
                available_regions["label"] == lab, "description"
            ].iloc[0]
            lines.append(f"{i}. {lab} ({desc})")
        lines.extend(["", "Please select one explicitly."])
        raise SystemExit("\n".join(lines))

    candidates = [
        (lab, desc)
        for lab, desc in zip(labels, descriptions)
        if query.lower() in lab.lower() or query.lower() in desc.lower()
    ]
    if candidates:
        header = (
            f"Region '{region_query}' matched multiple Allen regions:"
            if len(candidates) > 1
            else f"Region '{region_query}' did not exactly match; closest:"
        )
        lines = [header, ""]
        for i, (lab, desc) in enumerate(candidates, start=1):
            lines.append(f"{i}. {lab} ({desc})")
        lines.extend(["", "Please select one explicitly (use --list-regions)."])
        raise SystemExit("\n".join(lines))

    raise SystemExit(
        f"Region '{region_query}' not found. Use --list-regions to see options."
    )


def rank_sections(
    metadata: pd.DataFrame,
    region_label: str,
    region_level: str = "division",
    min_cells_per_section: int = 5000,
) -> pd.DataFrame:
    """Rank sections by regional cell count, then section ID.

    Counts are computed after anatomical selection. Sections with no cells in
    the region are omitted. Eligibility uses the regional count, including when
    a later export keeps the full section.
    """
    column = region_column(region_level)
    labels = metadata[column].astype("string").str.strip()
    in_region = labels.eq(str(region_label))
    regional = metadata.loc[in_region.to_numpy()]
    columns = [
        "brain_section_label",
        "z_section",
        "donor_label",
        "n_regional_cells",
        "n_section_cells",
        "eligible",
        "rank",
    ]
    if regional.empty:
        return pd.DataFrame(columns=columns)

    grouped = regional.groupby("brain_section_label", sort=False)
    counts = grouped.size().rename("n_regional_cells").to_frame()
    counts["z_section"] = grouped["z_section"].median()
    if "donor_label" in regional.columns:
        counts["donor_label"] = grouped["donor_label"].first()
    else:
        counts["donor_label"] = pd.NA
    section_totals = (
        metadata.groupby("brain_section_label", sort=False).size().rename("n_section_cells")
    )
    counts = counts.join(section_totals, how="left")
    counts = counts.reset_index()
    counts["eligible"] = counts["n_regional_cells"] >= int(min_cells_per_section)
    counts = counts.sort_values(
        ["n_regional_cells", "brain_section_label"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    counts["rank"] = np.arange(1, len(counts) + 1)
    return counts[columns]


def choose_sections(
    ranked: pd.DataFrame,
    n_sections: int,
    explicit: list[str] | None = None,
    min_cells_per_section: int = 5000,
) -> tuple[pd.DataFrame, str | None]:
    """Pick sections to export.

    Explicit IDs override automatic ranking but must still meet the regional
    cell threshold. Automatic selection exports fewer than requested when not
    enough sections qualify, and fails when none do.
    """
    if n_sections < 1 and not explicit:
        raise SystemExit("--n-sections must be >= 1.")
    if ranked.empty:
        raise SystemExit("No sections contain cells from the selected region.")

    by_id = ranked.set_index("brain_section_label", drop=False)
    if explicit:
        missing = []
        ineligible = []
        rows = []
        for section_id in explicit:
            if section_id not in by_id.index:
                missing.append(section_id)
                continue
            row = by_id.loc[section_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            n_regional = int(row["n_regional_cells"])
            if n_regional < min_cells_per_section:
                ineligible.append(f"{section_id} ({n_regional} regional cells)")
            else:
                rows.append(row)
        if missing or ineligible:
            lines = ["Explicit --sections failed region or threshold checks."]
            if missing:
                lines.append(
                    "Not present in the selected region: " + ", ".join(missing) + "."
                )
            if ineligible:
                lines.append(
                    f"Below --min-cells-per-section {min_cells_per_section}: "
                    + ", ".join(ineligible)
                    + "."
                )
            raise SystemExit(" ".join(lines))
        chosen = pd.DataFrame(rows).reset_index(drop=True)
        return chosen, None

    eligible_mask = ranked["n_regional_cells"] >= int(min_cells_per_section)
    eligible = ranked.loc[eligible_mask].reset_index(drop=True)
    if eligible.empty:
        best = int(ranked["n_regional_cells"].max())
        raise SystemExit(
            "No sections have at least "
            f"{min_cells_per_section} cells in the selected region. "
            f"The largest section has {best} regional cells."
        )
    chosen = eligible.head(n_sections).copy()
    message = None
    if len(chosen) < n_sections:
        message = (
            f"Requested {n_sections} sections with at least "
            f"{min_cells_per_section} regional cells; only {len(eligible)} "
            f"eligible. Exporting {len(chosen)}."
        )
    return chosen, message


def select_section_cells(
    metadata: pd.DataFrame,
    section_id: str,
    region_label: str,
    region_level: str = "division",
    scope: str = "region",
) -> pd.DataFrame:
    """Return annotated cells for one physical section.

    ``region`` crops to the anatomical selection. ``full-section`` keeps every
    eligible cell on that section. Subclass abundance is not filtered.
    """
    if scope not in {"region", "full-section"}:
        raise SystemExit(f"Unknown --scope '{scope}'. Choose region or full-section.")
    section = metadata.loc[metadata["brain_section_label"].astype(str) == str(section_id)]
    if section.empty:
        raise RuntimeError(f"No eligible cells found for section '{section_id}'.")
    if scope == "full-section":
        return section.sort_index().copy()
    column = region_column(region_level)
    cropped = section.loc[section[column].astype(str) == str(region_label)]
    if cropped.empty:
        raise RuntimeError(
            f"Section '{section_id}' has no cells in region '{region_label}'."
        )
    return cropped.sort_index().copy()
