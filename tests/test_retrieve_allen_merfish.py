"""Unit tests for MERFISH section retrieval. No Allen download required."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse

from merfish_retrieval.coordinates import (
    section_coordinates_to_um,
    verify_section_coordinate_units,
)
from merfish_retrieval.datasets import MERFISH_C57BL6J_638850, REGION_ALIASES_AIT21
from merfish_retrieval.export import (
    attach_metadata_and_coordinates,
    report_reference_compatibility,
    validate_section,
)
from merfish_retrieval.expression import (
    drop_control_features,
    resolve_raw_expression_source,
    subset_expression_by_cell_ids,
)
from merfish_retrieval.regions import (
    choose_sections,
    list_regions,
    rank_sections,
    resolve_region,
    select_section_cells,
)


def _sections_z() -> dict[str, float]:
    return {f"S{i}": 1.0 + 0.2 * i for i in range(4)}


def _metadata() -> pd.DataFrame:
    z = _sections_z()
    rows = []
    specs = [
        ("c0", "S0", "CB", "VI", "Purkinje", 1.0, 2.0),
        ("c1", "S0", "CB", "VI", "Granule", 1.2, 2.1),
        ("c2", "S0", "CB", "VI", "Rare", 1.4, 2.2),
        ("c3", "S1", "CB", "VII", "Granule", 3.0, 4.0),
        ("c4", "S1", "HY", "HY", "Other", 3.2, 4.2),
        ("c5", "S2", "CB", "VIII", "Granule", 5.0, 1.0),
        ("c6", "S2", "CB", "VIII", "Purkinje", 5.1, 1.1),
        ("c7", "S3", "MY", "MY", "Hind", 0.2, 0.3),
    ]
    for cell_id, section, division, structure, subclass, x, y in specs:
        rows.append(
            {
                "brain_section_label": section,
                "donor_label": "donor",
                "class": "class",
                "subclass": subclass,
                "supertype": subclass,
                "cluster": subclass,
                "parcellation_division": division,
                "parcellation_structure": structure,
                "parcellation_substructure": structure,
                "x_section": x,
                "y_section": y,
                "z_section": z[section],
            }
        )
    # Tie: S1 and a duplicate-count section. S0 has 3 CB, S2 has 2 CB, S1 has 1 CB.
    frame = pd.DataFrame(rows).set_index(pd.Index([row[0] for row in specs], name="cell_label"))
    return frame


def _tie_metadata() -> pd.DataFrame:
    """Two sections with the same regional count; IDs must break the tie."""
    rows = []
    cells = [
        ("a", "C57BL6J-638850.20", "CB", 2.0),
        ("b", "C57BL6J-638850.10", "CB", 4.0),
        ("c", "C57BL6J-638850.10", "CB", 4.0),
        ("d", "C57BL6J-638850.20", "CB", 2.0),
        ("e", "C57BL6J-638850.05", "HY", 0.4),
    ]
    for i, (cell_id, section, division, z) in enumerate(cells):
        rows.append(
            {
                "cell_id": cell_id,
                "brain_section_label": section,
                "donor_label": "donor",
                "class": "c",
                "subclass": "s",
                "supertype": "s",
                "cluster": "s",
                "parcellation_division": division,
                "parcellation_structure": division,
                "parcellation_substructure": division,
                "x_section": float(i),
                "y_section": float(i),
                "z_section": z,
            }
        )
    return pd.DataFrame(rows).set_index("cell_id")


def test_resolve_cerebellum_and_reject_ambiguous_or_partial():
    available = pd.DataFrame(
        {
            "label": ["CB", "HPF", "MY"],
            "description": ["Cerebellum", "Hippocampal formation", "Medulla"],
        }
    )
    assert resolve_region("cerebellum", available, REGION_ALIASES_AIT21) == "CB"
    assert resolve_region("CB", available, REGION_ALIASES_AIT21) == "CB"
    assert resolve_region("Cerebellum", available, REGION_ALIASES_AIT21) == "CB"
    assert resolve_region("hindbrain", available, REGION_ALIASES_AIT21) == "MY"
    with pytest.raises(SystemExit, match="did not exactly match"):
        resolve_region("cereb", available, REGION_ALIASES_AIT21)
    with pytest.raises(SystemExit, match="not found"):
        resolve_region("spinal cord", available, REGION_ALIASES_AIT21)


def test_list_regions_counts_cells_and_sections():
    regions = list_regions(
        _metadata(),
        "division",
        {"CB": "Cerebellum", "HY": "Hypothalamus", "MY": "Medulla"},
    )
    cb = regions.loc[regions["label"] == "CB"].iloc[0]
    assert int(cb["n_cells"]) == 6
    assert int(cb["n_sections"]) == 3
    assert cb["description"] == "Cerebellum"


def test_rank_sections_orders_by_count_then_id_and_keeps_rare_cells():
    ranked = rank_sections(_metadata(), "CB", min_cells_per_section=2)
    assert ranked["brain_section_label"].tolist() == ["S0", "S2", "S1"]
    assert ranked["eligible"].tolist() == [True, True, False]
    assert int(ranked.loc[ranked["brain_section_label"] == "S0", "n_regional_cells"].iloc[0]) == 3

    tie = rank_sections(_tie_metadata(), "CB", min_cells_per_section=1)
    assert tie["brain_section_label"].tolist() == ["C57BL6J-638850.10", "C57BL6J-638850.20"]

    selected = select_section_cells(_metadata(), "S0", "CB", scope="region")
    assert selected["subclass"].value_counts()["Rare"] == 1
    full = select_section_cells(_metadata(), "S1", "CB", scope="full-section")
    assert set(full.index) == {"c3", "c4"}


def test_choose_sections_reports_shortfall_and_checks_explicit_ids():
    ranked = rank_sections(_metadata(), "CB", min_cells_per_section=2)
    chosen, message = choose_sections(ranked, n_sections=5, min_cells_per_section=2)
    assert chosen["brain_section_label"].tolist() == ["S0", "S2"]
    assert message is not None and "only 2 eligible" in message

    with pytest.raises(SystemExit, match="No sections have at least"):
        choose_sections(ranked, n_sections=1, min_cells_per_section=100)

    explicit, note = choose_sections(
        ranked,
        n_sections=1,
        explicit=["S2"],
        min_cells_per_section=2,
    )
    assert note is None
    assert explicit["brain_section_label"].tolist() == ["S2"]
    with pytest.raises(SystemExit, match="Below"):
        choose_sections(ranked, n_sections=1, explicit=["S1"], min_cells_per_section=2)


def test_subset_alignment_uses_cell_ids_when_rows_are_shuffled():
    ids = ["c0", "c1", "c2", "extra"]
    values = np.arange(12, dtype=float).reshape(4, 3)
    order = [2, 0, 3, 1]
    obs = pd.DataFrame(
        {"brain_section_label": ["S0", "S0", "S0", "S0"]},
        index=[ids[i] for i in order],
    )
    adata = ad.AnnData(X=sparse.csr_matrix(values[order]), obs=obs)
    subset, report = subset_expression_by_cell_ids(
        adata,
        ["c1", "c0"],
        section_id="S0",
        annotated_ids=["c0", "c1", "c2"],
    )
    assert list(subset.obs_names) == ["c1", "c0"]
    np.testing.assert_array_equal(subset.X.toarray()[1], values[0])
    np.testing.assert_array_equal(subset.X.toarray()[0], values[1])
    assert report["n_expression_only"] == 1
    assert sparse.issparse(subset.X)
    with pytest.raises(RuntimeError, match="no expression row"):
        subset_expression_by_cell_ids(adata, ["c0", "missing"])
    allowed, allowed_report = subset_expression_by_cell_ids(
        adata, ["c0", "missing", "c1"], allow_missing=True
    )
    assert list(allowed.obs_names) == ["c0", "c1"]
    assert allowed_report["missing_expression_ids"] == ["missing"]


def test_attach_metadata_follows_cell_id_not_row_order():
    metadata = _metadata().loc[["c0", "c1", "c2"]]
    expression_order = ["c2", "c0"]
    matrix = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 4.0]]))
    expression = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=expression_order),
        var=pd.DataFrame(index=["ENSMUSG00000000001", "ENSMUSG00000000002"]),
    )
    coordinate_info = {"conversion_factor_to_um": 1000.0}
    adata = attach_metadata_and_coordinates(
        expression,
        metadata,
        coordinate_info,
        {"dataset": "test"},
    )
    assert list(adata.obs_names) == expression_order
    assert str(adata.obs.loc["c0", "subclass"]) == "Purkinje"
    assert str(adata.obs.loc["c2", "subclass"]) == "Rare"
    np.testing.assert_allclose(adata.obsm["spatial"][1], [1000.0, 2000.0])
    np.testing.assert_allclose(adata.obsm["spatial"][0], [1400.0, 2200.0])


def test_control_features_are_removed_and_zero_genes_remain():
    genes = ["ENSMUSG00000000001", "ENSMUSG00000000002", "Blank-1", "Blank-2"]
    matrix = sparse.csr_matrix(
        np.array(
            [
                [0.0, 3.0, 1.0, 0.0],
                [0.0, 0.0, 2.0, 4.0],
            ]
        )
    )
    adata = ad.AnnData(X=matrix, obs=pd.DataFrame(index=["c0", "c1"]), var=pd.DataFrame(index=genes))
    gene_metadata = pd.DataFrame(
        {"gene_symbol": ["GeneA", "GeneB", "Blank-1", "Blank-2"]},
        index=genes,
    )
    kept, removed = drop_control_features(adata, gene_metadata)
    assert removed == ["Blank-1", "Blank-2"]
    assert list(kept.var_names) == ["ENSMUSG00000000001", "ENSMUSG00000000002"]
    assert kept.X[:, 0].sum() == 0
    assert kept.n_obs == 2


def test_unit_conversion_accepts_documented_millimeter_spacing():
    frame = _metadata()
    info = verify_section_coordinate_units(frame)
    assert info["source_unit"] == "mm"
    assert info["conversion_factor_to_um"] == 1000.0
    coords = section_coordinates_to_um(frame.loc[["c0"]], info["conversion_factor_to_um"])
    np.testing.assert_allclose(coords, [[1000.0, 2000.0]])

    wrong = frame.copy()
    wrong["z_section"] = wrong["z_section"] * 1000.0
    with pytest.raises(RuntimeError, match="Cannot establish coordinate units"):
        verify_section_coordinate_units(wrong)


def test_incompatible_reference_taxonomy_is_rejected(tmp_path):
    section = _section_anndata()
    ref = ad.AnnData(
        X=sparse.csr_matrix(np.ones((2, 2))),
        obs=pd.DataFrame({"subclass": ["Purkinje", "Granule"]}),
        var=pd.DataFrame(index=["ENSMUSG00000000001", "ENSMUSG00000000002"]),
    )
    ref.uns["reference_info"] = {"taxonomy": "AIT33", "taxonomy_id": "CCN20251031"}
    path = tmp_path / "ait33.h5ad"
    ref.write_h5ad(path)
    with pytest.raises(SystemExit, match="incompatible"):
        report_reference_compatibility(section, path)

    ref.uns["reference_info"] = {}
    missing = tmp_path / "missing.h5ad"
    ref.write_h5ad(missing)
    with pytest.raises(SystemExit, match="no taxonomy provenance"):
        report_reference_compatibility(section, missing)


def test_reference_report_keeps_uncovered_cells(tmp_path):
    section = _section_anndata()
    ref = ad.AnnData(
        X=sparse.eye(2, format="csr"),
        obs=pd.DataFrame({"subclass": ["Purkinje", "Granule"]}),
        var=pd.DataFrame(index=["ENSMUSG00000000001", "ENSMUSG00000000999"]),
    )
    ref.uns["reference_info"] = {
        "taxonomy": "AIT21",
        "taxonomy_id": "CCN20230722",
        "taxonomy_directory": "WMB-taxonomy",
    }
    path = tmp_path / "ait21.h5ad"
    ref.write_h5ad(path)
    with pytest.warns(UserWarning, match="retained"):
        report = report_reference_compatibility(section, path, coverage_warn_threshold=0.95)
    assert report["provenance"] == "ait21"
    assert report["coverage"] == pytest.approx(2 / 3)
    assert "Rare" in report["merfish_only_labels"]
    assert section.n_obs == 3
    assert "Rare" in set(section.obs["subclass"].astype(str))
    assert report["gene_overlap"]["n_shared_genes"] == 1


def test_resolve_raw_source_prefers_section_raw_and_rejects_log2():
    class FakeCache:
        def list_expression_matrix_files(self, directory):
            return {
                "MERFISH-C57BL6J-638850-sections": [
                    "C57BL6J-638850.01/log2",
                    "C57BL6J-638850.01/raw",
                    "C57BL6J-638850.02/log2",
                ],
                "MERFISH-C57BL6J-638850": [
                    "C57BL6J-638850/log2",
                    "C57BL6J-638850/raw",
                ],
            }[directory]

    dataset = MERFISH_C57BL6J_638850
    raw = resolve_raw_expression_source(FakeCache(), "C57BL6J-638850.01", dataset)
    assert raw.kind == "section_raw"
    assert raw.file_name.endswith("/raw")
    fallback = resolve_raw_expression_source(FakeCache(), "C57BL6J-638850.02", dataset)
    assert fallback.kind == "whole_brain_raw"

    class LogOnly:
        def list_expression_matrix_files(self, directory):
            if directory.endswith("-sections"):
                return ["C57BL6J-638850.03/log2"]
            return ["C57BL6J-638850/log2"]

    with pytest.raises(RuntimeError, match="Refusing to substitute log-normalized"):
        resolve_raw_expression_source(LogOnly(), "C57BL6J-638850.03", dataset)


def _section_anndata() -> ad.AnnData:
    metadata = _metadata().loc[["c0", "c1", "c2"]]
    expression = ad.AnnData(
        X=sparse.csr_matrix(np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 5.0]])),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["ENSMUSG00000000001", "ENSMUSG00000000002"]),
    )
    adata = attach_metadata_and_coordinates(
        expression,
        metadata,
        {"conversion_factor_to_um": 1000.0},
        {"dataset": "test"},
    )
    validate_section(
        adata,
        section_id="S0",
        label_key="subclass",
        region_label="CB",
        region_column="parcellation_division",
    )
    return adata
