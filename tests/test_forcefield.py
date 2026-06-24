"""Tests for force field provider module."""
import pytest
from rdkit import Chem

from confgen.forcefield import ForceFieldProvider


def test_rdkit_backend():
    assert ForceFieldProvider("mmff").backend == "rdkit"
    assert ForceFieldProvider("uff").backend == "rdkit"


def test_openmm_backend():
    assert ForceFieldProvider("gaff").backend == "openmm"
    assert ForceFieldProvider("smirnoff").backend == "openmm"
    assert ForceFieldProvider("espaloma").backend == "openmm"


def test_tblite_backend():
    assert ForceFieldProvider("gfn2-xtb").backend == "tblite"
    assert ForceFieldProvider("gfn1-xtb").backend == "tblite"
    assert ForceFieldProvider("ipea1-xtb").backend == "tblite"


def test_unknown_backend():
    with pytest.raises(ValueError, match="Unknown forcefield"):
        ForceFieldProvider("invalid").backend


def test_has_rdkit_params():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol, randomSeed=42)
    assert ForceFieldProvider("mmff").has_rdkit_params(mol) is True
    assert ForceFieldProvider("uff").has_rdkit_params(mol) is True


def test_tblite_method():
    assert ForceFieldProvider("gfn2-xtb").get_tblite_method() == "GFN2-xTB"
    assert ForceFieldProvider("gfn1-xtb").get_tblite_method() == "GFN1-xTB"
    assert ForceFieldProvider("ipea1-xtb").get_tblite_method() == "IPEA1-xTB"


@pytest.mark.slow
def test_openmm_vacuum_system_is_nonperiodic():
    """Vacuum OpenMM systems must not request periodic boundary conditions."""
    pytest.importorskip("openmm")
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(mol, randomSeed=42)
    system, modeller = ForceFieldProvider("smirnoff").build_openmm_system(
        mol, solvent=None
    )

    assert system.usesPeriodicBoundaryConditions() is False
    assert modeller.topology.getPeriodicBoxVectors() is None
