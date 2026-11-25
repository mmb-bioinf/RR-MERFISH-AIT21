"""Section-space coordinates and the millimeter-to-micrometer conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from merfish_retrieval.datasets import (
    EXPECTED_Z_SPACING_MM,
    MM_TO_UM,
    Z_SPACING_TOLERANCE_MM,
)

UNIT_BASIS = (
    "Allen MERFISH section coordinates are the original planar section frame. "
    "Published z is the physical section position: consecutive coronal sections "
    "are ~200 µm apart, equal to 0.2 mm in the MERFISH and CCF registration "
    "tutorials. Retrieval accepts the scale only when the median spacing of "
    "adjacent section z positions matches 0.2 mm, then multiplies x_section and "
    "y_section by 1000 to store obsm['spatial'] in micrometers. "
    "CCF-warped coordinates are not used for binning."
)


def verify_section_coordinate_units(
    metadata: pd.DataFrame,
    *,
    expected_spacing_mm: float = EXPECTED_Z_SPACING_MM,
    tolerance_mm: float = Z_SPACING_TOLERANCE_MM,
) -> dict:
    """Confirm section coordinates are millimeters using documented z spacing.

    Fails when the spacing cannot be checked. Does not infer a scale from the
    magnitude of x or y.
    """
    if "z_section" not in metadata.columns or "brain_section_label" not in metadata.columns:
        raise RuntimeError(
            "Cannot establish coordinate units without brain_section_label and z_section."
        )
    z = pd.to_numeric(metadata["z_section"], errors="coerce")
    labels = metadata["brain_section_label"].astype(str)
    per_section = (
        pd.DataFrame({"brain_section_label": labels, "z_section": z})
        .dropna()
        .groupby("brain_section_label", sort=False)["z_section"]
        .median()
        .sort_values()
    )
    if len(per_section) < 2:
        raise RuntimeError(
            "Cannot establish coordinate units: fewer than two sections have a "
            "finite z_section. Refusing to infer a scale from coordinate magnitudes."
        )
    diffs = np.diff(per_section.to_numpy(dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        raise RuntimeError(
            "Cannot establish coordinate units: section z positions are not "
            "strictly ordered. Refusing to infer a scale from coordinate magnitudes."
        )
    observed = float(np.median(diffs))
    if abs(observed - expected_spacing_mm) > tolerance_mm:
        raise RuntimeError(
            "Cannot establish coordinate units: median adjacent-section z spacing "
            f"is {observed:.4f}, expected {expected_spacing_mm:.1f} mm "
            f"(200 µm, tolerance ±{tolerance_mm} mm). "
            "Refusing to infer a scale from coordinate magnitudes."
        )
    return {
        "source_columns": ["x_section", "y_section"],
        "z_column": "z_section",
        "source_unit": "mm",
        "coordinate_frame": "section_space",
        "conversion_factor_to_um": MM_TO_UM,
        "expected_z_spacing_mm": expected_spacing_mm,
        "observed_median_z_spacing_mm": observed,
        "unit_basis": UNIT_BASIS,
    }


def section_coordinates_to_um(
    metadata: pd.DataFrame,
    conversion_factor: float = MM_TO_UM,
) -> np.ndarray:
    """Convert verified section-space x/y coordinates to micrometers."""
    if conversion_factor != MM_TO_UM:
        raise RuntimeError(
            f"Unsupported conversion factor {conversion_factor}. "
            "MERFISH section coordinates are converted from mm to µm by 1000."
        )
    missing = [col for col in ("x_section", "y_section") if col not in metadata.columns]
    if missing:
        raise RuntimeError(f"Metadata lacks section coordinate columns: {missing}")
    xy = metadata[["x_section", "y_section"]].apply(pd.to_numeric, errors="coerce")
    coords = xy.to_numpy(dtype=float) * conversion_factor
    if coords.shape != (len(metadata), 2) or not np.isfinite(coords).all():
        raise RuntimeError("Section coordinates contain non-finite values.")
    return coords
