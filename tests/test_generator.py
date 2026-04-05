"""Tests for conformer generator module."""
from rdkit import Chem

from confgen.generator import ConformerGenerator


def test_generate_ethanol(ethanol_mol):
    gen = ConformerGenerator(n_confs=10, rmsd_threshold=0.5, seed=42)
    mol, conf_ids = gen.generate(ethanol_mol)
    assert mol is not None
    assert len(conf_ids) >= 1
    assert all(isinstance(cid, int) for cid in conf_ids)


def test_generate_aspirin(aspirin_mol):
    gen = ConformerGenerator(n_confs=50, rmsd_threshold=1.0, seed=42)
    mol, conf_ids = gen.generate(aspirin_mol)
    assert mol is not None
    assert len(conf_ids) >= 1


def test_generate_with_high_rmsd_threshold(aspirin_mol):
    """Very high threshold should collapse to ~1 conformer."""
    gen = ConformerGenerator(n_confs=50, rmsd_threshold=100.0, seed=42)
    _, conf_ids = gen.generate(aspirin_mol)
    assert len(conf_ids) == 1


def test_generate_with_low_rmsd_threshold(aspirin_mol):
    """Low threshold should keep more conformers."""
    gen = ConformerGenerator(n_confs=50, rmsd_threshold=0.3, seed=42)
    _, conf_ids = gen.generate(aspirin_mol)
    assert len(conf_ids) >= 2


def test_generate_seed_reproducibility(ethanol_mol):
    gen1 = ConformerGenerator(n_confs=20, rmsd_threshold=0.5, seed=123)
    gen2 = ConformerGenerator(n_confs=20, rmsd_threshold=0.5, seed=123)
    _, ids1 = gen1.generate(Chem.MolFromSmiles("CCO"))
    _, ids2 = gen2.generate(Chem.MolFromSmiles("CCO"))
    assert len(ids1) == len(ids2)
