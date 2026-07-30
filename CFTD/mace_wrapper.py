import contextlib
import io
import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from mace.calculators import mace_mp


class MaceFeatureWrapper(nn.Module):
    """
    Wraps a pre-trained MACE model to extract invariant features.
    Extracts 'node_feats' directly from the model's forward pass dictionary.
    """

    def __init__(
        self,
        model_name: str = "medium-0b3",
        device: str = "cuda",
        projection_dim: int = 128,
        projection_seed: int = 0,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.projectionDim = int(projection_dim)
        self.projectionSeed = int(projection_seed)
        self.register_buffer("randomProjection", torch.empty(0), persistent=False)

        prev_default_dtype = torch.get_default_dtype()
        root_logger = logging.getLogger()
        old_level = root_logger.level
        out = io.StringIO()
        err = io.StringIO()
        try:
            root_logger.setLevel(logging.ERROR)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
                    category=UserWarning,
                )
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    mace_model = mace_mp(
                        model=model_name,
                        device=str(self.device),
                        default_dtype="float32",
                    )
        finally:
            root_logger.setLevel(old_level)
        # MACE init may change global default dtype; restore caller default.
        torch.set_default_dtype(prev_default_dtype)

        # The model is usually a ScaleShiftMACE instance
        self.model = mace_model.models[0]
        # set to eval mode
        self.model.eval()

        # require grad false, to save some gpu mem
        for param in self.model.parameters():
            param.requires_grad = False

        # Handle z_table (Atomic Number -> Index mapping)
        self.atomic_numbers = self.model.atomic_numbers.tolist()

    def _project(self, pooled: torch.Tensor) -> torch.Tensor:

        # Use an isolated generator so fixed projections do not mess with other seeds 
        # projection is good because it helps us to reduce dimnsions 
        # also enables the use of larger mace features
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.projectionSeed))
        projection = torch.randn(
            int(pooled.shape[-1]),
            int(self.projectionDim),
            generator=generator,
            dtype=torch.float32,
        )
        # random projections on the sphere 
        projection = F.normalize(projection, dim=0)
        self.randomProjection = projection.to(device=pooled.device, dtype=pooled.dtype)
        return pooled @ self.randomProjection


    # building the mace input 
    def _to_mace_input(self, batch: Batch):
        z = batch.x.long()

        supported_zs = self.atomic_numbers
        z_map = {int(z_val): idx for idx, z_val in enumerate(supported_zs)}

        indices = []
        valid_mask = []

        for atom_z in z.cpu().numpy():
            atom_z = int(atom_z)
            if atom_z in z_map:
                indices.append(z_map[atom_z])
                valid_mask.append(1.0)
            else:
                indices.append(0)
                valid_mask.append(0.0)
                print(f"Warning: Z={atom_z} unsupported by MACE. Treating as ghost.")

        indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        valid_mask = torch.tensor(valid_mask, device=self.device, dtype=self.dtype).unsqueeze(1)

        node_attrs = F.one_hot(indices, num_classes=len(supported_zs)).to(self.dtype)
        node_attrs = node_attrs * valid_mask

        input_dict = {
            "positions": batch.pos.to(dtype=self.dtype),
            "node_attrs": node_attrs,
            "batch": batch.batch,
            "ptr": batch.ptr,
            "cell": batch.cell.view(-1, 3, 3).to(dtype=self.dtype) if hasattr(batch, "cell") else None,
            "edge_index": batch.edge_index,
            "shifts": batch.edge_shift.to(dtype=self.dtype),
        }

        return input_dict



    def forward(self, batch: Batch) -> torch.Tensor:
        """Returns pooled invariant features (Batch_Size, Hidden_Dim)."""
        input_dict = self._to_mace_input(batch)

        # Run forward pass with forces disabled for speed
        out = self.model(
            input_dict,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False
        )

        node_feats = out["node_feats"]
        node_feats = torch.clamp(node_feats, min=-10.0, max=10.0)
        # Mean pooling to get crystal-level features
        pooled = global_mean_pool(node_feats, batch.batch)

        return self._project(pooled).float()

    def predict_forces(self, batch: Batch) -> torch.Tensor:
        """
        Returns per-atom force targets [N, 3] from frozen MACE.
        """
        input_dict = self._to_mace_input(batch)
        # Force computation in MACE needs autograd w.r.t. positions.
        pos = input_dict["positions"].detach().clone().requires_grad_(True)
        input_dict["positions"] = pos
        with torch.enable_grad():
            out = self.model(
                input_dict,
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
                compute_displacement=False,
            )
        if "forces" not in out:
            raise KeyError("MACE output did not contain 'forces'.")
        forces = out["forces"].detach()
        if not torch.isfinite(forces).all():
            forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
        return forces.float()
