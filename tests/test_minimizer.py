"""Tests for minimizer module (RDKit backend always; OpenMM requires optional deps)."""
from concurrent.futures import Future

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import confgen.minimizer as minimizer_mod
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


def test_ps_to_steps():
    """_ps_to_steps should correctly convert picoseconds to step count."""
    ff = ForceFieldProvider("mmff")
    minimizer = Minimizer(ff, md_timestep_fs=0.5)
    assert minimizer._ps_to_steps(50.0) == 100_000   # 50 ps / 0.5 fs
    assert minimizer._ps_to_steps(1.0) == 2_000       # 1 ps / 0.5 fs

    minimizer2 = Minimizer(ff, md_timestep_fs=2.0)
    assert minimizer2._ps_to_steps(50.0) == 25_000    # 50 ps / 2 fs


def test_run_parallel_preserves_job_order(monkeypatch):
    """Parallel OpenMM workers should return results in conformer/job order."""
    submitted = []

    class FakeExecutor:
        def __init__(self, max_workers, mp_context):
            self.max_workers = max_workers
            self.mp_context = mp_context

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, **kwargs):
            submitted.append(kwargs)
            future = Future()
            cid = kwargs["original_cid"]
            if cid == 2:
                future.set_exception(RuntimeError("boom"))
            else:
                future.set_result((cid, float(cid), None))
            return future

    monkeypatch.setattr(minimizer_mod, "ProcessPoolExecutor", FakeExecutor)

    ff = ForceFieldProvider("smirnoff")
    minimizer = Minimizer(ff, gpu_streams=2)
    jobs = [{"original_cid": 3}, {"original_cid": 1}, {"original_cid": 2}]

    results = minimizer._run_parallel(jobs)

    assert submitted == jobs
    assert [cid for cid, _, _ in results] == [3, 1, 2]
    assert results[-1][1] != results[-1][1]


@pytest.mark.slow
def test_nvmolkit_minimize_mmff():
    """nvmolkit MMFF minimization should return valid energies and update conformer positions."""
    pytest.importorskip("nvmolkit")
    mol, conf_ids = _gen_conformers("CCO", n=3)
    assert mol is not None
    ff = ForceFieldProvider("mmff")
    minimizer = Minimizer(ff, max_iters=200, gen_backend="nvmolkit")
    energies = minimizer.minimize(mol, conf_ids)
    assert len(energies) == len(conf_ids)
    for cid, energy in energies:
        assert isinstance(energy, float)
        assert energy == energy  # not NaN


@pytest.mark.slow
def test_nvmolkit_minimize_uff():
    """nvmolkit UFF minimization should return valid energies."""
    pytest.importorskip("nvmolkit")
    mol, conf_ids = _gen_conformers("c1ccccc1", n=3)
    assert mol is not None
    ff = ForceFieldProvider("uff")
    minimizer = Minimizer(ff, max_iters=200, gen_backend="nvmolkit")
    energies = minimizer.minimize(mol, conf_ids)
    assert len(energies) == len(conf_ids)
    for cid, energy in energies:
        assert isinstance(energy, float)
        assert energy == energy  # not NaN


@pytest.mark.slow
def test_openmm_vacuum_no_md():
    """OpenMM vacuum minimization without MD should return valid energies."""
    openmm = pytest.importorskip("openmm")
    mol, conf_ids = _gen_conformers("CCO", n=2)
    assert mol is not None
    ff = ForceFieldProvider("smirnoff")
    minimizer = Minimizer(ff, max_iters=50, run_md=False)
    energies = minimizer.minimize(mol, conf_ids[:1])
    assert len(energies) == 1
    assert isinstance(energies[0][1], float)


@pytest.mark.slow
def test_openmm_vacuum_with_md():
    """run_md with vacuum: initial minimize → NVT → production → final minimize."""
    pytest.importorskip("openmm")
    mol, conf_ids = _gen_conformers("CCO", n=2)
    assert mol is not None
    ff = ForceFieldProvider("smirnoff")
    minimizer = Minimizer(
        ff,
        max_iters=50,
        run_md=True,
        md_nvt_time_ps=0.005,   # 10 steps at 0.5 fs — fast for testing
        md_prod_time_ns=0.00005,
    )
    energies = minimizer.minimize(mol, conf_ids[:1])
    assert len(energies) == 1
    assert isinstance(energies[0][1], float)


@pytest.mark.slow
def test_openmm_save_trajectories(tmp_path):
    """save_trajectories writes topology PDB and DCD trajectory per conformer."""
    pytest.importorskip("openmm")
    mol, conf_ids = _gen_conformers("CCO", n=2)
    assert mol is not None
    ff = ForceFieldProvider("smirnoff")
    minimizer = Minimizer(
        ff,
        max_iters=50,
        run_md=True,
        md_nvt_time_ps=0.005,
        md_prod_time_ns=0.00005,
        save_trajectories=True,
        output_dir=str(tmp_path),
    )
    energies = minimizer.minimize(mol, conf_ids[:1], mol_id="ethanol")
    assert len(energies) == 1

    traj_dir = tmp_path / "mdsims" / "ethanol"
    cid = conf_ids[0]
    assert (traj_dir / f"conf_{cid:03d}_topology.pdb").exists()
    assert (traj_dir / f"conf_{cid:03d}_trajectory.dcd").exists()
