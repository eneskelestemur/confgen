"""Tests for constraints module."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from confgen.constraints import build_coord_map


def test_build_coord_map_basic():
    """Build coord map from benzene substructure match."""
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc(O)cc1"))

    ref_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(ref_mol, randomSeed=42)

    smarts = "c1ccccc1"
    cmap = build_coord_map(mol, smarts, ref_mol)
    assert cmap is not None
    assert len(cmap) == 6  # 6 aromatic carbons


def test_build_coord_map_no_match():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    ref_mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(ref_mol, randomSeed=42)

    cmap = build_coord_map(mol, "c1ccccc1", ref_mol)
    assert cmap is None


def test_build_coord_map_invalid_smarts():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    ref_mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(ref_mol, randomSeed=42)

    cmap = build_coord_map(mol, "[INVALID", ref_mol)
    assert cmap is None
