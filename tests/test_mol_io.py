"""Tests for mol_io module."""
from pathlib import Path

from rdkit import Chem

from confgen.mol_io import (
    assign_mol_ids,
    read_molecules,
    write_input_molecules_smi,
)


def test_read_smiles_file(tmp_smi_file):
    mols = read_molecules(tmp_smi_file)
    assert len(mols) == 3
    assert all(mol is not None for mol, _ in mols)
    assert mols[0][1] == "ethanol"
    assert mols[1][1] == "benzene"


def test_read_smiles_file_no_header(tmp_path):
    """Test reading SMILES with no IDs."""
    smi_file = tmp_path / "noid.smi"
    smi_file.write_text("CCO\nc1ccccc1\n")
    mols = read_molecules(str(smi_file))
    assert len(mols) == 2
    assert mols[0][1] is None  # no ID provided


def test_read_sdf_file(tmp_sdf_file):
    mols = read_molecules(tmp_sdf_file)
    assert len(mols) == 1
    assert mols[0][0] is not None
    assert mols[0][1] == "ethanol"


def test_read_directory(tmp_path):
    """Test reading all SDF files from a directory."""
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol, randomSeed=42)

    for i in range(3):
        mol.SetProp("_Name", f"mol_{i}")
        writer = Chem.SDWriter(str(tmp_path / f"mol_{i}.sdf"))
        writer.write(mol)
        writer.close()

    mols = read_molecules(str(tmp_path))
    assert len(mols) == 3


def test_assign_mol_ids():
    mol1 = Chem.MolFromSmiles("CCO")
    mol2 = Chem.MolFromSmiles("C")
    raw = [(mol1, "ethanol"), (None, "bad"), (mol2, None)]
    result = assign_mol_ids(raw)
    assert len(result) == 2
    assert result[0][1] == "ethanol"
    assert result[1][1] == "mol_000001"


def test_write_input_molecules_smi(tmp_path, ethanol_mol, benzene_mol):
    mols = [(ethanol_mol, "eth"), (benzene_mol, "benz")]
    out_path = tmp_path / "out.smi"
    write_input_molecules_smi(mols, out_path)
    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert "eth" in lines[0]
    assert "benz" in lines[1]


def test_invalid_smiles_skipped(tmp_path):
    smi_file = tmp_path / "bad.smi"
    smi_file.write_text("CCO ethanol\nINVALID$$$ bad_mol\nc1ccccc1 benzene\n")
    mols = read_molecules(str(smi_file))
    valid = [m for m, _ in mols if m is not None]
    assert len(valid) == 2
