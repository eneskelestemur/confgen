"""
Conformer generation and molecule preparation utilities.

This module provides:
- ConformerGenerator: Generate diverse 3D conformations with RMSD-based clustering
- MoleculePreparator: High-level preparation from various input sources
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers, rdMolAlign
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

from VeQTOR.constants import ORGANIC_ATOMS

_logger = logging.getLogger(__name__)


class ConformerGenerator:
    """
    Generate diverse 3D conformations for molecules.
    
    Uses ETKDG for conformer generation and RMSD-based clustering to select
    diverse conformers. Conformers within the same RMSD cluster are represented
    by a single conformer (lowest energy if optimization is enabled).
    """
    
    def __init__(
        self,
        n_iters: int = 100,
        rmsd_threshold: float = 2.5,
        optimize: bool = True,
        random_seed: int = 42,
        num_threads: int = 1,
        timeout: int = 30,
    ):
        """
        Args:
            n_iters: Number of conformer generation attempts
            rmsd_threshold: RMSD threshold (Å) for clustering conformers
            optimize: Whether to run MMFF/UFF optimization
            random_seed: Random seed for reproducibility (-1 for random)
            num_threads: Number of threads for parallel operations
            timeout: Timeout in seconds for conformer generation
        """
        self.n_iters = n_iters
        self.rmsd_threshold = rmsd_threshold
        self.optimize = optimize
        self.random_seed = random_seed if random_seed >= 0 else -1
        self.num_threads = num_threads
        self.timeout = timeout
    
    def generate(self, mol: Mol) -> tuple[Mol | None, list[int]]:
        """
        Generate diverse conformers for a molecule.
        
        Args:
            mol: RDKit molecule (can be 2D or 3D)
            
        Returns:
            Tuple of the molecule with conformers and a list of selected conformer IDs.
            Returns None, [] if conformer generation fails completely.
        """
        mol = Chem.AddHs(mol)
        
        params = rdDistGeom.EmbedParameters()
        params.timeout = self.timeout
        params.randomSeed = self.random_seed
        params.numThreads = self.num_threads
        
        conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=self.n_iters, params=params)
        
        if (len(conf_ids) == 0) or (len(conf_ids) == 1 and conf_ids[0] == -1):
            _logger.warning(f"Failed to generate any conformers for molecule")
            return None, []
        
        if self.optimize:
            self._optimize_conformers(mol)
        
        selected_ids = self._cluster_by_rmsd(mol)
        
        if len(selected_ids) == 0:
            _logger.warning(f"No conformers survived clustering")
            return None, []
        
        return mol, selected_ids
    
    def _optimize_conformers(self, mol: Mol) -> None:
        """
        Optimize conformers using MMFF94, falling back to UFF if MMFF fails.
        """
        has_mmff_params = rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol)
        has_uff_params = rdForceFieldHelpers.UFFHasAllMoleculeParams(mol)
        
        if has_mmff_params:
            results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
                mol, 
                mmffVariant='MMFF94',
                numThreads=self.num_threads,
                maxIters=500,
                nonBondedThresh=100.0,
            )
            failed = sum(1 for conv, _ in results if conv != 0)
            if failed > 0:
                _logger.debug(f"{failed}/{len(results)} MMFF optimizations did not converge")
        elif has_uff_params:
            results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
                mol,
                numThreads=self.num_threads,
                maxIters=500,
                vdwThresh=100.0,
            )
            failed = sum(1 for conv, _ in results if conv != 0)
            if failed > 0:
                _logger.debug(f"{failed}/{len(results)} UFF optimizations did not converge")
        else:
            _logger.debug("Molecule lacks parameters for MMFF and UFF; skipping optimization")
    
    def _cluster_by_rmsd(self, mol: Mol) -> list[int]:
        """
        Cluster conformers by RMSD and select one representative per cluster.
        
        Uses a greedy approach: iterate through conformers, adding to a new
        cluster if RMSD to all existing cluster representatives exceeds threshold.
        """
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        if len(conf_ids) <= 1:
            return conf_ids
        
        # Get all pairwise RMSDs efficiently using RDKit's threaded implementation
        rms_matrix = rdMolAlign.GetAllConformerBestRMS(mol, numThreads=self.num_threads)
        
        # Convert to full matrix for easier indexing
        n_confs = len(conf_ids)
        full_matrix = np.zeros((n_confs, n_confs))
        idx = 0
        for i in range(n_confs):
            for j in range(i):
                full_matrix[i, j] = rms_matrix[idx]
                full_matrix[j, i] = rms_matrix[idx]
                idx += 1
        
        # Greedy clustering
        selected = [0]
        for i in range(1, n_confs):
            min_rmsd_to_selected = min(full_matrix[i, s] for s in selected)
            if min_rmsd_to_selected >= self.rmsd_threshold:
                selected.append(i)
        
        return [conf_ids[i] for i in selected]


class MoleculePreparator:
    """
    Prepare molecules with 3D conformations from various input sources.
    
    Handles:
    - Reading from SMILES files, SDF files, or directories
    - Molecule curation (fragment selection, atom validation)
    - Conformer generation with diversity filtering
    - Output to SDF with conformer metadata
    """
    
    def __init__(
        self,
        conformer_generator: ConformerGenerator | None = None,
        max_heavy_atoms: int = 64,
        verbose: bool = True,
    ):
        """
        Args:
            conformer_generator: ConformerGenerator instance (creates default if None)
            max_heavy_atoms: Maximum number of heavy atoms allowed
            verbose: Whether to show progress bars
        """
        self.conformer_generator = conformer_generator or ConformerGenerator()
        self.max_heavy_atoms = max_heavy_atoms
        self.verbose = verbose
    
    def prepare_from_file(
        self,
        input_path: str,
        output_path: str,
        smiles_column: str = 'SMILES',
        name_column: str | None = None,
    ) -> dict:
        """
        Prepare molecules from input file and save to SDF.
        
        Args:
            input_path: Path to input file (.sdf, .csv, .smi) or directory of SDFs
            output_path: Path to output SDF file
            smiles_column: Column name for SMILES in CSV/SMI files
            name_column: Column name for molecule names (optional)
            
        Returns:
            Statistics dictionary with counts of processed molecules
        """
        stats = {
            'total_input': 0,
            'successful': 0,
            'failed_curation': 0,
            'failed_conformer': 0,
            'total_conformers': 0,
        }
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mol_iter = self._read_input(input_path, smiles_column=smiles_column, name_column=name_column)
        mol_list = list(mol_iter)
        stats['total_input'] = len(mol_list)
        
        writer = Chem.SDWriter(output_path)
        mol_idx = 0
        
        iterator = tqdm(mol_list, desc='Preparing molecules', disable=not self.verbose)
        for mol, name in iterator:
            if mol is None:
                stats['failed_curation'] += 1
                continue
            
            curated_mol = self._curate_molecule(mol)
            if curated_mol is None:
                stats['failed_curation'] += 1
                continue
            
            mol_embed, conf_ids = self.conformer_generator.generate(curated_mol)
            if len(conf_ids) == 0:
                stats['failed_conformer'] += 1
                continue
            
            stats['successful'] += 1
            stats['total_conformers'] += len(conf_ids)
            
            for i, conf_id in enumerate(conf_ids):
                sample_name = f"m{mol_idx:08d}_c{i:03d}"
                mol_embed.SetProp('_Name', sample_name)
                mol_embed.SetProp('mol_idx', str(mol_idx))
                mol_embed.SetProp('conf_idx', str(i))
                if name:
                    mol_embed.SetProp('original_name', name)
                writer.write(mol_embed, confId=conf_id)
            
            mol_idx += 1
        
        writer.close()
        
        _logger.info(f"Preparation complete: {stats['successful']}/{stats['total_input']} molecules, "
                     f"{stats['total_conformers']} total conformers")
        
        return stats
    
    def _curate_molecule(self, mol: Mol) -> Mol | None:
        """
        Curate molecule: keep largest fragment, validate atoms, check size.
        
        Returns None if molecule fails validation.
        """
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda x: x.GetNumHeavyAtoms())
        
        if mol.GetNumHeavyAtoms() > self.max_heavy_atoms:
            _logger.debug(f"Molecule has {mol.GetNumHeavyAtoms()} heavy atoms, exceeds limit of {self.max_heavy_atoms}")
            return None
        
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in ORGANIC_ATOMS:
                _logger.debug(f"Molecule contains disallowed element: {atom.GetSymbol()}")
                return None
        
        return mol
    
    def _read_input(
        self,
        input_path: str,
        smiles_column: str = 'SMILES',
        name_column: str | None = None,
    ) -> Iterator[tuple[Mol | None, str | None]]:
        """
        Read molecules from various input formats.
        
        Yields:
            Tuples of (molecule, name) where molecule may be None if reading fails
        """
        path = Path(input_path)
        
        if path.is_dir():
            yield from self._read_sdf_directory(path)
        elif path.suffix.lower() == '.sdf':
            yield from self._read_sdf_file(path)
        elif path.suffix.lower() in ('.csv', '.smi', '.smiles'):
            yield from self._read_smiles_file(path, smiles_column, name_column)
        else:
            raise ValueError(f"Unsupported input format: {path.suffix}")
    
    def _read_sdf_file(self, path: Path) -> Iterator[tuple[Mol | None, str | None]]:
        """Read molecules from an SDF file."""
        suppl = Chem.SDMolSupplier(str(path), removeHs=False)
        for mol in suppl:
            name = mol.GetProp('_Name') if mol and mol.HasProp('_Name') else None
            yield mol, name
    
    def _read_sdf_directory(self, path: Path) -> Iterator[tuple[Mol | None, str | None]]:
        """Read molecules from all SDF files in a directory."""
        for sdf_file in sorted(path.glob('*.sdf')):
            yield from self._read_sdf_file(sdf_file)
    
    def _read_smiles_file(
        self,
        path: Path,
        smiles_column: str,
        name_column: str | None,
    ) -> Iterator[tuple[Mol | None, str | None]]:
        """Read molecules from a SMILES file (CSV or SMI format)."""
        sep = '\t' if path.suffix.lower() == '.smi' else ','
        
        try:
            df = pd.read_csv(path, sep=sep)
        except Exception as e:
            _logger.error(f"Failed to read file {path}: {e}")
            return
        
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in {path}")
        
        for idx, row in df.iterrows():
            smiles = row[smiles_column]
            name = row[name_column] if name_column and name_column in df.columns else None
            
            try:
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                mol = None
            
            yield mol, name
