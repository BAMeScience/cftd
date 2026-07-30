"""
Coarse-Fine Transport Distance implementation.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from .mace_wrapper import MaceFeatureWrapper
from .utils import structure_to_graph, weighted_quantile


class CoarseFineTransportDistance:
    """
    Two-threshold CFTD with mixed OT coupling.

    Mathematics:
    1. Coupling cost C = alpha * D_id + (1 - alpha) * D_qual
    2. Thresholds from OT-plan-weighted costs:
       tail = (1 - betaGoldilocks) / 2
       tau_mem = Q_tail(D_id), tau_qual = Q_(1-tail)(D_qual).
    3. Penalty for similar and bad quality materials = 
       Thresholding with taus on the respective distance matrices
    4. qualityMass = qualityPenalty * OT-plan
       memoryMass = memoryPenalty * OT-plan
    5. Calibrate M from the same split:
       M = qualityMass / memoryMass using exclusive masks.
    6. Score:
       quality = qualityMass
       memory  = M * memoryMass
       total   = quality + memory
    """

    def __init__(
        self,
        train_structs: Sequence,
        calib_structs: Optional[Sequence] = None,
        geo_model: Optional[nn.Module] = None,
        mace_model_name: str = "medium-0b3",
        *,
        coupling_alpha: float = 0.5,
        beta_goldilocks: float = 0.6,
        memory_weight: Optional[float] = None,
        mace_cutoff: float = 6.0,
        device: str = "cuda",
        batch_size: int = 32,
        ot_num_itermax: int = 20_000_000,
    ) -> None:
        self.device = torch.device(device)
        # batch sizer for featurization 
        self.batchSize = int(batch_size)
        # ot steps, since the problem is quite high dim and many particles, 
        # we need a bit 
        self.otSteps = int(ot_num_itermax)
        # weighting of id and mace of distances
        self.alpha = float(coupling_alpha)
        # size of zone that is not penalized 
        self.betaGoldilocks = float(beta_goldilocks)
        # Cutoff neighbouring distances to create the graphs from structures
        self.maceCutoff = float(mace_cutoff)

        # load train and calibration structures 
        self.trainStructs = list(train_structs)
        self.calibStructs = None if calib_structs is None else list(calib_structs)
        # GNN to identify similar structures; Used to return the identity vector
        self.geoModel = geo_model.to(self.device).eval()
        # MACE model wrapper; returns the MACE h-vector
        self.maceWrapper = MaceFeatureWrapper(model_name=mace_model_name, device=str(self.device))
        # featurize train and calibration sets
        with torch.no_grad():
            self.trainIdFeats, self.trainMaceFeats = self.featurizeBatchRaw(self.trainStructs)
            if self.calibStructs is not None:
                self.calibIdFeats, self.calibMaceFeats = self.featurizeBatchRaw(self.calibStructs)
        # identity is already normalized 
        self.idScale = 1.0
        # for mace, scale is important, but for the final score, 
        # we want to live in similar numerical ranges as identity
        self.maceScale = float(self.trainMaceFeats.norm(dim=1).mean().item())

        # calibrate the hyperparameter
        self.tauMem, self.tauQual, calibratedMBalance = self.calibrateFromFeats(
            self.trainIdFeats,
            self.trainMaceFeats,
            self.calibIdFeats,
            self.calibMaceFeats,
        )
        self.calibratedMBalance = float(calibratedMBalance)
        self.mBalance = self.calibratedMBalance if memory_weight is None else float(memory_weight)

        if memory_weight is None:
            print(
                f"CoarseFineTransportDistance calibrated: tau_mem={self.tauMem:.6f}, "
                f"tau_qual={self.tauQual:.6f}, M={self.mBalance:.6f}"
            )
        else:
            print(
                f"CoarseFineTransportDistance calibrated: tau_mem={self.tauMem:.6f}, "
                f"tau_qual={self.tauQual:.6f}, M={self.mBalance:.6f} "
                f"(calibrated_M={self.calibratedMBalance:.6f})"
            )


    # feautrize  batches helper 
    def featurizeBatchRaw(self, structures: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        idChunks = []
        maceChunks = []

        for start in range(0, len(structures), self.batchSize):
            chunk = structures[start : start + self.batchSize]

            geoBatch = Batch.from_data_list([structure_to_graph(s) for s in chunk]).to(self.device)
            geoBatch.x = geoBatch.x.long()
            geoBatch.edge_index = geoBatch.edge_index.long()
            geoBatch.batch = geoBatch.batch.long()

            maceBatch = Batch.from_data_list(
                [structure_to_graph(s, cutoff=self.maceCutoff) for s in chunk]
            ).to(self.device)
            
            # identity is trained with normalized (internally, during loss calculation)
            idFeat = F.normalize(self.geoModel(geoBatch), dim=1)
            # mace is not normalized here
            maceFeat = self.maceWrapper(maceBatch)

            idChunks.append(idFeat.detach().cpu())
            maceChunks.append(maceFeat.detach().cpu())

        return torch.cat(idChunks, dim=0), torch.cat(maceChunks, dim=0)

    def otPlanFromCost(self, cost: torch.Tensor) -> torch.Tensor:
        nRows, nCols = cost.shape
        rowMass = np.ones(nRows) / float(nRows)
        colMass = np.ones(nCols) / float(nCols)
        plan = ot.emd(rowMass, colMass, cost.detach().cpu().numpy(), numItermax=self.otSteps)
        return torch.as_tensor(plan, dtype=cost.dtype)

    def transportPlanFromFeats(
        self,
        cost: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cost is None:
            raise ValueError("EMD coupling requires a dense cost matrix")
        return self.otPlanFromCost(cost)

    def calibrateFromFeats(
        self,
        refId: torch.Tensor,
        refMace: torch.Tensor,
        calibId: torch.Tensor,
        calibMace: torch.Tensor,
    ) -> tuple[float, float, float]:
        
        idDistance = torch.cdist(refId, calibId) / self.idScale
        # equjivalent to rescaling mace features
        qualDistance = torch.cdist(refMace, calibMace) / self.maceScale
        # coupling
        couplingCost = self.alpha * idDistance + (1.0 - self.alpha) * qualDistance
        # get plan 
        plan = self.transportPlanFromFeats(couplingCost)
        # find thresholds such that the split leaves beta goldlilocks mass in the middle     
        # and 1-beta/2 in the tails under the OT plan
        tailProb = 0.5 * (1.0 - self.betaGoldilocks)
        tauMem = float(weighted_quantile(idDistance, plan, tailProb).item())
        tauQual = float(weighted_quantile(qualDistance, plan, 1.0 - tailProb).item())
        # if mem and qual clash then preference is to do a mem penalty 
        memMask = idDistance <= tauMem
        qualMask = (qualDistance > tauQual) & (~memMask)
        # combine masks with plan to get CFTD
        qualityPenalty = (qualDistance - tauQual) * qualMask.to(qualDistance.dtype)
        memoryPenalty = (tauMem - idDistance) * memMask.to(idDistance.dtype)
        qualityMass = torch.sum(plan * qualityPenalty)
        memoryMass = torch.sum(plan * memoryPenalty)
        mBalance = float((qualityMass / memoryMass).item())

        return float(tauMem), float(tauQual), float(mBalance)

    def computeCFTD(self, gen_structures: Sequence, return_mem_mass: bool = False):
        with torch.no_grad():
            genId, genMace = self.featurizeBatchRaw(gen_structures)

        idDistance = torch.cdist(self.trainIdFeats, genId) / self.idScale
        qualDistance = torch.cdist(self.trainMaceFeats, genMace) / self.maceScale
        couplingCost = self.alpha * idDistance + (1.0 - self.alpha) * qualDistance
        plan = self.transportPlanFromFeats(couplingCost)

        memMask = idDistance <= float(self.tauMem)
        qualMask = (qualDistance > float(self.tauQual)) & (~memMask)

        qualityPenalty = (qualDistance - float(self.tauQual)) * qualMask.to(qualDistance.dtype)
        memoryPenalty = (float(self.tauMem) - idDistance) * memMask.to(idDistance.dtype)

        quality = torch.sum(plan * qualityPenalty)
        memoryMassRaw = torch.sum(plan * memoryPenalty)
        memory = float(self.mBalance) * memoryMassRaw

        total = quality + memory

        print(
            f"CoarseFineTransportDistance | quality={float(quality):.6f} | "
            f"memory={float(memory):.6f} | total={float(total):.6f}"
        )
        if bool(return_mem_mass):
            return (
                float(total.item()),
                float(quality.item()),
                float(memory.item()),
                float(memoryMassRaw.item()),
            )
        return float(total.item()), float(quality.item()), float(memory.item())
