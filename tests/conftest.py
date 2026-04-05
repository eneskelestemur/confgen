"""Shared fixtures for confgen tests."""
from __future__ import annotations

import pytest
from rdkit import Chem


@pytest.fixture
def ethanol_mol() -> Chem.Mol:
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def benzene_mol() -> Chem.Mol:
    return Chem.MolFromSmiles("c1ccccc1")


@pytest.fixture
def aspirin_mol() -> Chem.Mol:
    return Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")


@pytest.fixture
def stereo_mol() -> Chem.Mol:
    """Molecule with unspecified stereocenters."""
    return Chem.MolFromSmiles("CC(O)C(N)C")


@pytest.fixture
def tmp_smi_file(tmp_path) -> str:
    """Create a temporary SMILES file."""
    smi_file = tmp_path / "test.smi"
    smi_file.write_text("CCO ethanol\nc1ccccc1 benzene\nCC(=O)O acetic_acid\n")
    return str(smi_file)


@pytest.fixture
def tmp_sdf_file(tmp_path, ethanol_mol) -> str:
    """Create a temporary single-molecule SDF file with 3D coords."""
    mol = Chem.AddHs(ethanol_mol)
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol, randomSeed=42)
    mol.SetProp("_Name", "ethanol")

    sdf_path = tmp_path / "test.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return str(sdf_path)
