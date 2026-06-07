"""Conformer generation via ETKDG and RMSD-based diversity clustering."""
from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolAlign
from rdkit.Geometry import Point3D

_logger = logging.getLogger(__name__)


class ConformerGenerator:
    """Generate diverse 3D conformers using ETKDG with RMSD-based pruning.

    Supports two generation backends:
    - ``rdkit``    — CPU-based ETKDG via RDKit (default)
    - ``nvmolkit`` — GPU-accelerated ETKDG via NVIDIA nvMolKit
    """

    def __init__(
        self,
        n_confs: int = 200,
        rmsd_threshold: float = 1.5,
        seed: int = 42,
        num_threads: int = 1,
        timeout: int = 10,
        coord_map: dict[int, tuple[float, float, float]] | None = None,
        backend: str = "rdkit",
    ):
        """
        Args:
            n_confs: Number of conformer embedding attempts.
            rmsd_threshold: RMSD threshold (A) for diversity clustering.
            seed: Random seed (-1 for non-deterministic).
            num_threads: Threads for ETKDG embedding and RMSD calculation.
            timeout: Timeout in seconds per molecule embedding.
            coord_map: Optional atom_idx -> (x, y, z) map for constrained embedding.
            backend: Embedding backend — ``rdkit`` (CPU) or ``nvmolkit`` (GPU).
        """
        self.n_confs = n_confs
        self.rmsd_threshold = rmsd_threshold
        self.seed = seed if seed >= 0 else -1
        self.num_threads = num_threads
        self.timeout = timeout
        self.coord_map = coord_map
        self.backend = backend

    def generate(self, mol: Chem.Mol) -> tuple[Chem.Mol | None, list[int]]:
        """Embed conformers and cluster by RMSD.

        Args:
            mol: RDKit molecule (2D or 3D; Hs will be added).

        Returns:
            (mol_with_Hs, selected_conf_ids) or (None, []) on failure.
        """
        mol = Chem.AddHs(mol)
        params = self._make_etkdg_params()

        if self.backend == "nvmolkit":
            conf_ids = self._embed_nvmolkit(mol, params)
        else:
            conf_ids = self._embed_rdkit(mol, params)

        if len(conf_ids) == 0 or (len(conf_ids) == 1 and conf_ids[0] == -1):
            _logger.warning("Failed to generate any conformers")
            return None, []

        selected = self._cluster_by_rmsd(mol)
        if not selected:
            _logger.warning("No conformers survived RMSD clustering")
            return None, []

        return mol, selected

    def _make_etkdg_params(self) -> rdDistGeom.EmbedParameters:
        """Build ETKDGv3 parameters shared by both embedding backends."""
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = self.seed
        params.numThreads = self.num_threads
        params.pruneRmsThresh = -1.0  # we do our own RMSD clustering
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        return params

    def _embed_rdkit(self, mol: Chem.Mol, params: rdDistGeom.EmbedParameters) -> list[int]:
        """Embed conformers using RDKit's CPU-based ETKDG implementation."""
        if self.coord_map:
            params.coordMap = {idx: Point3D(*xyz) for idx, xyz in self.coord_map.items()}
        return list(rdDistGeom.EmbedMultipleConfs(mol, numConfs=self.n_confs, params=params))

    def _embed_nvmolkit(self, mol: Chem.Mol, params: rdDistGeom.EmbedParameters) -> list[int]:
        """Embed conformers using NVIDIA nvMolKit's GPU-accelerated ETKDG.

        Modifies ``mol`` in-place (same contract as the RDKit path).
        ``params.useRandomCoords`` is forced to ``True`` as required by nvMolKit.
        Constrained embedding (``coord_map``) is not supported by nvMolKit and
        will be silently ignored with a warning.
        """
        from nvmolkit.embedMolecules import EmbedMolecules  # type: ignore[import]
        if self.coord_map:
            _logger.warning(
                "nvmolkit backend does not support constrained embedding; coord_map ignored"
            )
        params.useRandomCoords = True
        EmbedMolecules(molecules=[mol], params=params, confsPerMolecule=self.n_confs)
        return [c.GetId() for c in mol.GetConformers()]

    def _cluster_by_rmsd(self, mol: Chem.Mol) -> list[int]:
        """Greedy RMSD clustering: keep first conformer, add others only if
        RMSD to all selected representatives >= threshold."""
        conf_ids = [c.GetId() for c in mol.GetConformers()]
        if len(conf_ids) <= 1:
            return conf_ids

        n = len(conf_ids)

        if self.backend == "nvmolkit":
            rms_condensed = self._rmsd_nvmolkit(mol)
        else:
            rms_condensed = np.array(
                rdMolAlign.GetAllConformerBestRMS(mol, numThreads=self.num_threads)
            )

        full = np.zeros((n, n))
        i_idx, j_idx = np.tril_indices(n, -1)
        full[i_idx, j_idx] = rms_condensed
        full[j_idx, i_idx] = rms_condensed

        selected = [0]
        for i in range(1, n):
            if full[i, selected].min() >= self.rmsd_threshold:
                selected.append(i)

        return [conf_ids[i] for i in selected]

    def _rmsd_nvmolkit(self, mol: Chem.Mol) -> np.ndarray:
        """Compute pairwise conformer RMSD on GPU via nvMolKit.

        Falls back to RDKit CPU if nvmolkit is unavailable.
        Returns a condensed 1-D array of length n*(n-1)//2.
        """
        try:
            from nvmolkit.conformerRmsd import GetConformerRMSMatrix  # type: ignore[import]
        except ImportError:
            _logger.warning("nvmolkit not available; falling back to RDKit RMSD")
            return np.array(rdMolAlign.GetAllConformerBestRMS(mol, numThreads=self.num_threads))

        mol_noh = Chem.RemoveHs(mol)
        result = GetConformerRMSMatrix(mol_noh)
        return result.numpy()
