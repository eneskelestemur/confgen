"""Tests for curation module."""
from rdkit import Chem

from confgen.curation import curate_molecule, enumerate_stereoisomers


def test_curate_basic(ethanol_mol):
    result = curate_molecule(ethanol_mol)
    assert result is not None
    assert result.GetNumHeavyAtoms() == 3


def test_curate_none():
    assert curate_molecule(None) is None


def test_curate_too_many_atoms():
    """A molecule exceeding max_heavy_atoms should be rejected."""
    mol = Chem.MolFromSmiles("C" * 20)
    assert curate_molecule(mol, max_heavy_atoms=5) is None


def test_curate_disallowed_element():
    mol = Chem.MolFromSmiles("[Fe]")
    assert curate_molecule(mol) is None


def test_curate_keeps_largest_fragment():
    """Salts: keep the larger fragment."""
    mol = Chem.MolFromSmiles("CCO.[Na+]")
    result = curate_molecule(mol)
    assert result is not None
    assert result.GetNumHeavyAtoms() == 3


def test_enumerate_stereo_specified():
    """Fully specified stereo: should return single isomer."""
    mol = Chem.MolFromSmiles("[C@@H](O)(F)Cl")
    isomers = enumerate_stereoisomers(mol)
    assert len(isomers) == 1


def test_enumerate_stereo_unspecified(stereo_mol):
    """Molecule with unspecified stereo should produce >1 isomers."""
    isomers = enumerate_stereoisomers(stereo_mol)
    assert len(isomers) >= 2


def test_enumerate_stereo_max_limit():
    mol = Chem.MolFromSmiles("CC(O)C(N)C(F)C(Cl)C")
    isomers = enumerate_stereoisomers(mol, max_isomers=4)
    assert len(isomers) <= 4
