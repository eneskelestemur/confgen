"""Constrained embedding: fix substructure coordinates during conformer generation."""
from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Geometry import Point3D

_logger = logging.getLogger(__name__)


def build_coord_map(
    mol: Chem.Mol,
    smarts: str,
    ref_mol: Chem.Mol,
) -> dict[int, tuple[float, float, float]] | None:
    """Build an atom-index -> (x, y, z) map for constrained ETKDG embedding.

    Matches `smarts` against both `mol` and `ref_mol`, then maps the matched
    atom positions from `ref_mol`'s first conformer onto the corresponding
    atom indices in `mol`.

    Args:
        mol: Target molecule to embed (must have Hs if ref_mol does).
        smarts: SMARTS pattern identifying the fixed substructure.
        ref_mol: Reference molecule with 3D coordinates for the substructure.

    Returns:
        dict mapping atom indices in `mol` to (x, y, z) coordinates from `ref_mol`,
        or None if the SMARTS match fails for either molecule.
    """
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        _logger.error(f"Invalid SMARTS pattern: {smarts}")
        return None

    mol_match = mol.GetSubstructMatch(pattern)
    if not mol_match:
        _logger.warning("SMARTS did not match target molecule")
        return None

    ref_match = ref_mol.GetSubstructMatch(pattern)
    if not ref_match:
        _logger.warning("SMARTS did not match reference molecule")
        return None

    if ref_mol.GetNumConformers() == 0:
        _logger.error("Reference molecule has no conformer")
        return None

    ref_conf = ref_mol.GetConformer(0)
    coord_map: dict[int, tuple[float, float, float]] = {}
    for mol_idx, ref_idx in zip(mol_match, ref_match):
        pt = ref_conf.GetAtomPosition(ref_idx)
        coord_map[mol_idx] = (pt.x, pt.y, pt.z)

    _logger.info(f"Built coordinate map for {len(coord_map)} constrained atoms")
    return coord_map


def load_reference_mol(path: str) -> Chem.Mol | None:
    """Load a reference molecule from SDF, PDB, or MOL2."""
    from pathlib import Path

    p = Path(path)
    if p.suffix.lower() == ".sdf":
        suppl = Chem.SDMolSupplier(str(p), removeHs=False)
        mols = [m for m in suppl if m is not None]
        return mols[0] if mols else None
    elif p.suffix.lower() == ".pdb":
        return Chem.MolFromPDBFile(str(p), removeHs=False)
    elif p.suffix.lower() == ".mol2":
        return Chem.MolFromMol2File(str(p), removeHs=False)
    else:
        _logger.error(f"Unsupported reference file format: {p.suffix}")
        return None
