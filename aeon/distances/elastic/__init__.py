"""Distance computation."""

__all__ = [
    "create_bounding_matrix",
    "adtw_distance",
    "adtw_pairwise_distance",
    "adtw_cost_matrix",
    "adtw_alignment_path",
    "dtw_distance",
    "dtw_pairwise_distance",
    "dtw_cost_matrix",
    "dtw_alignment_path",
    "ddtw_distance",
    "ddtw_pairwise_distance",
    "ddtw_alignment_path",
    "ddtw_cost_matrix",
    "wdtw_distance",
    "wdtw_pairwise_distance",
    "wdtw_cost_matrix",
    "wdtw_alignment_path",
    "wddtw_distance",
    "wddtw_pairwise_distance",
    "wddtw_alignment_path",
    "wddtw_cost_matrix",
    "lcss_distance",
    "lcss_pairwise_distance",
    "lcss_alignment_path",
    "lcss_cost_matrix",
    "erp_distance",
    "erp_pairwise_distance",
    "erp_alignment_path",
    "erp_cost_matrix",
    "edr_distance",
    "edr_pairwise_distance",
    "edr_alignment_path",
    "edr_cost_matrix",
    "twe_distance",
    "twe_pairwise_distance",
    "twe_alignment_path",
    "twe_cost_matrix",
    "msm_distance",
    "msm_alignment_path",
    "msm_cost_matrix",
    "msm_pairwise_distance",
    "shape_dtw_distance",
    "shape_dtw_cost_matrix",
    "shape_dtw_alignment_path",
    "shape_dtw_pairwise_distance",
    "soft_dtw_distance",
    "soft_dtw_pairwise_distance",
    "soft_dtw_alignment_path",
    "soft_dtw_cost_matrix",
    "swale_distance",
    "swale_pairwise_distance",
]

from aeon.distances.elastic._adtw import (
    adtw_alignment_path,
    adtw_cost_matrix,
    adtw_distance,
    adtw_pairwise_distance,
)
from aeon.distances.elastic._bounding_matrix import create_bounding_matrix
from aeon.distances.elastic._ddtw import (
    ddtw_alignment_path,
    ddtw_cost_matrix,
    ddtw_distance,
    ddtw_pairwise_distance,
)
from aeon.distances.elastic._dtw import (
    dtw_alignment_path,
    dtw_cost_matrix,
    dtw_distance,
    dtw_pairwise_distance,
)
from aeon.distances.elastic._edr import (
    edr_alignment_path,
    edr_cost_matrix,
    edr_distance,
    edr_pairwise_distance,
)
from aeon.distances.elastic._erp import (
    erp_alignment_path,
    erp_cost_matrix,
    erp_distance,
    erp_pairwise_distance,
)
from aeon.distances.elastic._lcss import (
    lcss_alignment_path,
    lcss_cost_matrix,
    lcss_distance,
    lcss_pairwise_distance,
)
from aeon.distances.elastic._msm import (
    msm_alignment_path,
    msm_cost_matrix,
    msm_distance,
    msm_pairwise_distance,
)
from aeon.distances.elastic._shape_dtw import (
    shape_dtw_alignment_path,
    shape_dtw_cost_matrix,
    shape_dtw_distance,
    shape_dtw_pairwise_distance,
)
from aeon.distances.elastic._soft_dtw import (
    soft_dtw_alignment_path,
    soft_dtw_cost_matrix,
    soft_dtw_distance,
    soft_dtw_pairwise_distance,
)
from aeon.distances.elastic._swale import swale_distance, swale_pairwise_distance
from aeon.distances.elastic._twe import (
    twe_alignment_path,
    twe_cost_matrix,
    twe_distance,
    twe_pairwise_distance,
)
from aeon.distances.elastic._wddtw import (
    wddtw_alignment_path,
    wddtw_cost_matrix,
    wddtw_distance,
    wddtw_pairwise_distance,
)
from aeon.distances.elastic._wdtw import (
    wdtw_alignment_path,
    wdtw_cost_matrix,
    wdtw_distance,
    wdtw_pairwise_distance,
)
