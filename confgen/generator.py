"""Conformer generation via ETKDG and RMSD-based diversity clustering."""
from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolAlign
from rdkit.Geometry import Point3D

_logger = logging.getLogger(__name__)


class ConformerGenerator:
    """Generate diverse 3D conformers using ETKDG with RMSD-based pruning."""

    def __init__(
        self,
        n_confs: int = 200,
        rmsd_threshold: float = 1.5,
        seed: int = 42,
        num_threads: int = 1,
        timeout: int = 60,
        coord_map: dict[int, tuple[float, float, float]] | None = None,
    ):
        """
        Args:
            n_confs: Number of conformer embedding attempts.
            rmsd_threshold: RMSD threshold (A) for diversity clustering.
            seed: Random seed (-1 for non-deterministic).
            num_threads: Threads for ETKDG embedding and RMSD calculation.
            timeout: Timeout in seconds per molecule embedding.
            coord_map: Optional atom_idx -> (x, y, z) map for constrained embedding.
        """
        self.n_confs = n_confs
        self.rmsd_threshold = rmsd_threshold
        self.seed = seed if seed >= 0 else -1
        self.num_threads = num_threads
        self.timeout = timeout
        self.coord_map = coord_map

    def generate(self, mol: Chem.Mol) -> tuple[Chem.Mol | None, list[int]]:
        """Embed conformers and cluster by RMSD.

        Args:
            mol: RDKit molecule (2D or 3D; Hs will be added).

        Returns:
            (mol_with_Hs, selected_conf_ids) or (None, []) on failure.
        """
        mol = Chem.AddHs(mol)

        params = rdDistGeom.ETKDGv3()
        params.randomSeed = self.seed
        params.numThreads = self.num_threads
        params.pruneRmsThresh = -1.0  # we do our own RMSD clustering
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True

        if self.coord_map:
            coord_map_rd = {
                idx: Point3D(*xyz) for idx, xyz in self.coord_map.items()
            }
            params.coordMap = coord_map_rd

        conf_ids = rdDistGeom.EmbedMultipleConfs(
            mol, numConfs=self.n_confs, params=params
        )

        if len(conf_ids) == 0 or (len(conf_ids) == 1 and conf_ids[0] == -1):
            _logger.warning("Failed to generate any conformers")
            return None, []

        selected = self._cluster_by_rmsd(mol)
        if not selected:
            _logger.warning("No conformers survived RMSD clustering")
            return None, []

        return mol, selected

    def _cluster_by_rmsd(self, mol: Chem.Mol) -> list[int]:
        """Greedy RMSD clustering: keep first conformer, add others only if
        RMSD to all selected representatives >= threshold."""
        conf_ids = [c.GetId() for c in mol.GetConformers()]
        if len(conf_ids) <= 1:
            return conf_ids

        rms_list = rdMolAlign.GetAllConformerBestRMS(
            mol, numThreads=self.num_threads
        )

        n = len(conf_ids)
        full = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i):
                full[i, j] = full[j, i] = rms_list[idx]
                idx += 1

        selected = [0]
        for i in range(1, n):
            if min(full[i, s] for s in selected) >= self.rmsd_threshold:
                selected.append(i)

        return [conf_ids[i] for i in selected]
