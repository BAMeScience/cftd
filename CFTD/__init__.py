"""
High-level package exports for Coarse-Fine Transport Distance.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="`torch_geometric.distributed` has been deprecated",
    category=DeprecationWarning,
)

from .gcn import (  # noqa: E402
    EGNNLayer,
    EquivariantCrystalGCN,
    NonEquivariantCrystalGCN,
    info_nce_loss,
    train_contrastive_model,
    validate,
)
from .utils import *  # noqa: E402

from .CoarseFineTransportDistance import CoarseFineTransportDistance

try:  # Optional dependency: MACE is not required for SG-only training.
    from .mace import MaceFeatureWrapper  # noqa: E402
except Exception:  # pragma: no cover - keep package importable without mace
    MaceFeatureWrapper = None

__all__ = [
    "EquivariantCrystalGCN",
    "NonEquivariantCrystalGCN",
    "info_nce_loss",
    "train_contrastive_model",
    "validate",
    "StructureDataset",
    "augment",
    "coverage_score",
    "novelty_score",
    "perturb_structures_gaussian",
    "read_structure_from_csv",
    "structure_to_graph",
]

if MaceFeatureWrapper is not None:
    __all__.append("MaceFeatureWrapper")
__all__.append("CoarseFineTransportDistance")
