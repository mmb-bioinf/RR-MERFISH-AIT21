"""MERFISH section retrieval for Allen AIT21 spatial data.

Dataset-specific cache access lives in ``metadata`` and ``expression``.
Region selection, coordinate conversion, and AnnData export are generic.
"""

from merfish_retrieval.coordinates import (
    section_coordinates_to_um,
    verify_section_coordinate_units,
)
from merfish_retrieval.datasets import REGION_ALIASES_AIT21
from merfish_retrieval.export import (
    attach_metadata_and_coordinates,
    report_reference_compatibility,
    validate_section,
    write_outputs,
)
from merfish_retrieval.expression import (
    drop_control_features,
    load_section_expression,
    resolve_raw_expression_source,
    subset_expression_by_cell_ids,
)
from merfish_retrieval.metadata import load_metadata
from merfish_retrieval.regions import (
    choose_sections,
    list_regions,
    rank_sections,
    resolve_region,
    select_section_cells,
)

__all__ = [
    "REGION_ALIASES_AIT21",
    "attach_metadata_and_coordinates",
    "choose_sections",
    "drop_control_features",
    "list_regions",
    "load_metadata",
    "load_section_expression",
    "rank_sections",
    "report_reference_compatibility",
    "resolve_raw_expression_source",
    "resolve_region",
    "section_coordinates_to_um",
    "select_section_cells",
    "subset_expression_by_cell_ids",
    "validate_section",
    "verify_section_coordinate_units",
    "write_outputs",
]
