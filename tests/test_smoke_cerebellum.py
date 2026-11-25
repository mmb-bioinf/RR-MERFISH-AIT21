"""Smoke test: one real cerebellar MERFISH section and SNMF grid compression.

Skipped unless ABC_ATLAS_CACHE points at a cache that already has MERFISH
metadata. Set MERFISH_RUN_SMOKE=1 to download expression and run the check.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("abc_atlas_access")


def test_cerebellum_section_matches_snmf_grid(tmp_path):
    if os.environ.get("MERFISH_RUN_SMOKE") != "1":
        pytest.skip("Set MERFISH_RUN_SMOKE=1 to download a real MERFISH section.")
    cache_dir = os.environ.get("ABC_ATLAS_CACHE")
    if not cache_dir or not Path(cache_dir).exists():
        pytest.skip("Set ABC_ATLAS_CACHE to the Allen cache directory.")

    snmf = pytest.importorskip("snmf")
    import anndata as ad

    from retrieve_allen_merfish import main

    output_dir = tmp_path / "cerebellum"
    code = main(
        [
            "--region",
            "cerebellum",
            "--n-sections",
            "1",
            "--min-cells-per-section",
            "5000",
            "--cache-dir",
            cache_dir,
            "--output-dir",
            str(output_dir),
        ]
    )
    assert code == 0
    paths = sorted(output_dir.glob("C57BL6J-638850.*.h5ad"))
    assert len(paths) == 1
    adata = ad.read_h5ad(paths[0])
    assert adata.obs["brain_section_label"].nunique() == 1
    assert set(adata.obs["parcellation_division"].astype(str)) == {"CB"}
    assert adata.obs_names.is_unique and adata.var_names.is_unique
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    assert spatial.shape == (adata.n_obs, 2) and np.isfinite(spatial).all()

    binned = snmf.pp.compress_to_grid(
        adata,
        label_key="subclass",
        bin_size=55.0,
        spatial_key="spatial",
        layer=None,
        drop_empty=True,
    )
    original = np.asarray(adata.X.sum(axis=0)).ravel().astype(np.float64)
    compressed = np.asarray(binned.X.sum(axis=0)).ravel().astype(np.float64)
    np.testing.assert_allclose(compressed, original, rtol=1e-4, atol=1.0)
    np.testing.assert_allclose(binned.obsm["H_true"].sum(), adata.n_obs, atol=1e-3)
    fractions = np.asarray(binned.obsm["cell_type_fraction"], dtype=float)
    occupied = np.asarray(binned.obs["n_cells"]) > 0
    np.testing.assert_allclose(fractions[occupied].sum(axis=1), 1.0, atol=1e-5)
    cell_types = list(binned.uns["snmf_spatial_grid"]["cell_types"])
    counts = adata.obs["subclass"].astype(str).value_counts()
    for cell_type, total in zip(cell_types, binned.obsm["H_true"].sum(axis=0)):
        assert counts[cell_type] == pytest.approx(float(total))
    centers = np.asarray(binned.obsm["spatial"], dtype=float)
    assert np.isfinite(centers).all()
    xs = np.unique(np.round(centers[:, 0], 5))
    if len(xs) > 1:
        # Bin centers are float32, so adjacent centers sit within a small
        # fraction of a micrometre of the requested 55 µm pitch.
        assert np.min(np.diff(xs)) == pytest.approx(55.0, abs=0.05)
