"""Tests for minimizer module (RDKit backend only — others require optional deps)."""
from rdkit import Chem
from rdkit.Chem import AllChem

from confgen.forcefield import ForceFieldProvider
from confgen.minimizer import Minimizer
from confgen.generator import ConformerGenerator


def _gen_conformers(smiles: str, n: int = 5):
    mol = Chem.MolFromSmiles(smiles)
    gen = ConformerGenerator(n_confs=n, rmsd_threshold=0.3, seed=42)
    return gen.generate(mol)


def test_minimize_mmff():
    mol, conf_ids = _gen_conformers("CCO")
    assert mol is not None
    ff = ForceFieldProvider("mmff")
    minimizer = Minimizer(ff, max_iters=200, num_threads=1)
    energies = minimizer.minimize(mol, conf_ids)
    assert len(energies) == len(conf_ids)
    for cid, energy in energies:
        assert isinstance(energy, float)


def test_minimize_uff():
    mol, conf_ids = _gen_conformers("c1ccccc1")
    assert mol is not None
    ff = ForceFieldProvider("uff")
    minimizer = Minimizer(ff, max_iters=200, num_threads=1)
    energies = minimizer.minimize(mol, conf_ids)
    assert len(energies) == len(conf_ids)


def test_minimize_returns_sorted_by_conf_id():
    mol, conf_ids = _gen_conformers("CC(=O)O", n=10)
    assert mol is not None
    ff = ForceFieldProvider("mmff")
    minimizer = Minimizer(ff, max_iters=200)
    energies = minimizer.minimize(mol, conf_ids)
    returned_ids = [cid for cid, _ in energies]
    assert returned_ids == conf_ids
