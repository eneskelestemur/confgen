"""Energy minimization dispatching across RDKit, OpenMM, and tblite backends."""
from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
from scipy.optimize import minimize as scipy_minimize
from tblite.interface import Calculator

from confgen._constants import HARTREE_TO_KCALMOL, KJ_TO_KCAL
from confgen.forcefield import ForceFieldProvider
from confgen.solvation import is_explicit

_logger = logging.getLogger(__name__)


def _openmm_worker(
    mol: Chem.Mol,
    original_cid: int,
    ff_name: str,
    solvent: str | None,
    platform: str,
    seed: int,
    run_md: bool,
    max_iters: int,
    md_timestep_fs: float,
    md_temperature_k: float,
    md_pressure_atm: float,
    md_nvt_time_ps: float,
    md_prod_time_ns: float,
    traj_dir: str | None,
    conf_tag: str,
) -> tuple[int, float, np.ndarray]:
    """Run OpenMM minimize + optional MD for a single conformer.

    Designed to execute in an isolated subprocess: all arguments are picklable
    and no OpenMM objects cross the process boundary.  ``mol`` must contain
    exactly one conformer (the solute only).

    Returns ``(original_cid, energy_kcal, solute_positions_angstrom)``.
    The caller is responsible for writing positions back to the parent molecule.
    """
    import openmm as mm
    from openmm import app, unit

    n_solute = mol.GetNumAtoms()
    ff_provider = ForceFieldProvider(ff_name)
    system, modeller = ff_provider.build_openmm_system(mol, solvent=solvent)

    integrator = mm.LangevinMiddleIntegrator(
        md_temperature_k * unit.kelvin,
        1.0 / unit.picosecond,
        md_timestep_fs * unit.femtosecond,
    )
    if seed >= 0:
        integrator.setRandomNumberSeed(seed)

    omm_platform = mm.Platform.getPlatformByName(platform)
    simulation = app.Simulation(modeller.topology, system, integrator, omm_platform)
    simulation.context.setPositions(modeller.positions)

    if traj_dir:
        traj_path = Path(traj_dir)
        traj_path.mkdir(parents=True, exist_ok=True)
        with open(traj_path / f"{conf_tag}_topology.pdb", "w") as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
        report_interval = max(1, int(1000 / md_timestep_fs))  # every ~1 ps
        simulation.reporters.append(
            app.DCDReporter(str(traj_path / f"{conf_tag}_trajectory.dcd"), report_interval)
        )

    tolerance = 1.0 * unit.kilojoules_per_mole / unit.nanometer
    simulation.minimizeEnergy(tolerance=tolerance, maxIterations=max_iters)

    if run_md:
        def steps(time_ps: float) -> int:
            return int(time_ps * 1000 / md_timestep_fs)

        T = md_temperature_k * unit.kelvin
        simulation.context.setVelocitiesToTemperature(T, seed)
        simulation.step(steps(md_nvt_time_ps))

        if is_explicit(solvent):
            barostat = mm.MonteCarloBarostat(md_pressure_atm * unit.atmospheres, T)
            system.addForce(barostat)
            simulation.context.reinitialize(preserveState=True)
            simulation.step(steps(md_nvt_time_ps))

        simulation.step(steps(md_prod_time_ns * 1000))
        simulation.minimizeEnergy(tolerance=tolerance, maxIterations=max_iters)

    state = simulation.context.getState(energy=True, positions=True)
    energy_kcal = (
        state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole) * KJ_TO_KCAL
    )
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)[:n_solute]
    return (original_cid, energy_kcal, positions)


class Minimizer:
    """Minimize conformer energies using the appropriate backend."""

    def __init__(
        self,
        ff_provider: ForceFieldProvider,
        max_iters: int = 500,
        num_threads: int = 1,
        platform: str = "CPU",
        seed: int = 42,
        solvent: str | None = None,
        run_md: bool = False,
        md_timestep_fs: float = 0.5,
        md_temperature_k: float = 300.0,
        md_pressure_atm: float = 1.0,
        md_nvt_time_ps: float = 50.0,
        md_prod_time_ns: float = 0.1,
        gpu_streams: int = 1,
        save_trajectories: bool = False,
        output_dir: str | None = None,
        gen_backend: str = "rdkit",
    ):
        self.ff_provider = ff_provider
        self.max_iters = max_iters
        self.num_threads = num_threads
        self.platform = platform
        self.seed = seed
        self.solvent = solvent
        self.run_md = run_md
        self.md_timestep_fs = md_timestep_fs
        self.md_temperature_k = md_temperature_k
        self.md_pressure_atm = md_pressure_atm
        self.md_nvt_time_ps = md_nvt_time_ps
        self.md_prod_time_ns = md_prod_time_ns
        self.gpu_streams = gpu_streams
        self.save_trajectories = save_trajectories
        self.output_dir = output_dir
        self.gen_backend = gen_backend

    def minimize(
        self,
        mol: Chem.Mol,
        conf_ids: list[int],
        mol_id: str = "unknown",
    ) -> list[tuple[int, float]]:
        """Minimize each conformer and return (conf_id, energy_kcal_mol) pairs.

        Conformers are modified in-place on the molecule.
        """
        backend = self.ff_provider.backend
        if backend == "rdkit":
            if self.gen_backend == "nvmolkit":
                return self._minimize_nvmolkit(mol, conf_ids)
            return self._minimize_rdkit(mol, conf_ids)
        if backend == "openmm":
            return self._minimize_openmm(mol, conf_ids, mol_id)
        if backend == "tblite":
            return self._minimize_tblite(mol, conf_ids)
        raise ValueError(f"No minimizer for backend: {backend}")

    def _minimize_rdkit(
        self, mol: Chem.Mol, conf_ids: list[int]
    ) -> list[tuple[int, float]]:
        """Optimize with MMFF94 or UFF using RDKit's threaded batch optimization."""
        ff_name = self.ff_provider.name

        if ff_name == "mmff" and rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
            results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
                mol,
                mmffVariant="MMFF94",
                numThreads=self.num_threads,
                maxIters=self.max_iters,
                nonBondedThresh=100.0,
            )
        elif ff_name == "uff" and rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
            results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
                mol,
                numThreads=self.num_threads,
                maxIters=self.max_iters,
                vdwThresh=100.0,
            )
        else:
            _logger.warning(f"No {ff_name.upper()} parameters; returning unoptimized")
            return [(cid, float("nan")) for cid in conf_ids]

        failed = sum(1 for conv, _ in results if conv != 0)
        if failed:
            _logger.debug(f"{failed}/{len(results)} optimizations did not converge")

        # results is indexed by internal conformer order, map to conf_ids
        energies = []
        for i, cid in enumerate(conf_ids):
            all_cids = [c.GetId() for c in mol.GetConformers()]
            pos = all_cids.index(cid)
            _, energy_kcal = results[pos]
            energies.append((cid, energy_kcal))
        return energies

    def _minimize_nvmolkit(
        self, mol: Chem.Mol, conf_ids: list[int]
    ) -> list[tuple[int, float]]:
        """Optimize with MMFF94 or UFF using nvMolKit's GPU-accelerated batch optimizer.

        Falls back to the RDKit CPU path if nvmolkit is unavailable or the
        molecule lacks required parameters.
        """
        ff_name = self.ff_provider.name
        try:
            if ff_name == "mmff":
                from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs  # type: ignore[import]
                if not self.ff_provider.has_rdkit_params(mol):
                    _logger.warning("No MMFF parameters; returning unoptimized")
                    return [(cid, float("nan")) for cid in conf_ids]
                raw = MMFFOptimizeMoleculesConfs([mol], maxIters=self.max_iters)
            elif ff_name == "uff":
                from nvmolkit.mmffOptimization import UFFOptimizeMoleculesConfs  # type: ignore[import]
                if not self.ff_provider.has_rdkit_params(mol):
                    _logger.warning("No UFF parameters; returning unoptimized")
                    return [(cid, float("nan")) for cid in conf_ids]
                raw = UFFOptimizeMoleculesConfs([mol], maxIters=self.max_iters)
            else:
                _logger.warning(f"nvmolkit does not support {ff_name}; falling back to RDKit")
                return self._minimize_rdkit(mol, conf_ids)
        except ImportError:
            _logger.warning("nvmolkit not available; falling back to RDKit minimization")
            return self._minimize_rdkit(mol, conf_ids)

        # raw is list[list[float]]: [mol_idx][conformer_position] -> energy (kcal/mol)
        mol_energies = raw[0]
        all_cids = [c.GetId() for c in mol.GetConformers()]
        energies = []
        for cid in conf_ids:
            pos = all_cids.index(cid)
            energies.append((cid, float(mol_energies[pos])))
        return energies

    def _minimize_openmm(
        self, mol: Chem.Mol, conf_ids: list[int], mol_id: str
    ) -> list[tuple[int, float]]:
        """Minimize each conformer with OpenMM.

        A fresh system is built per conformer so that all conformers can run
        concurrently via ``gpu_streams`` without sharing CUDA contexts.
        For explicit solvent this also ensures each conformer gets its own
        unbiased water box.  The MD protocol (when ``run_md`` is enabled) runs
        inside ``_openmm_worker``, which executes in a subprocess when
        ``gpu_streams > 1``.
        """
        n_solute = mol.GetNumAtoms()
        traj_dir = (
            str(Path(self.output_dir) / "mdsims" / mol_id)
            if self.save_trajectories and self.run_md and self.output_dir
            else None
        )

        jobs = []
        for cid in conf_ids:
            conf_src = mol.GetConformer(cid)
            new_mol = Chem.RWMol(mol)
            new_mol.RemoveAllConformers()
            new_conf = Chem.Conformer(n_solute)
            for i in range(n_solute):
                pt = conf_src.GetAtomPosition(i)
                new_conf.SetAtomPosition(i, pt)
            new_mol.AddConformer(new_conf, assignId=True)

            jobs.append(dict(
                mol=new_mol.GetMol(),
                original_cid=cid,
                ff_name=self.ff_provider.name,
                solvent=self.solvent,
                platform=self.platform,
                seed=self.seed,
                run_md=self.run_md,
                max_iters=self.max_iters,
                md_timestep_fs=self.md_timestep_fs,
                md_temperature_k=self.md_temperature_k,
                md_pressure_atm=self.md_pressure_atm,
                md_nvt_time_ps=self.md_nvt_time_ps,
                md_prod_time_ns=self.md_prod_time_ns,
                traj_dir=traj_dir,
                conf_tag=f"conf_{cid:03d}",
            ))

        raw_results = (
            self._run_parallel(jobs) if self.gpu_streams > 1
            else self._run_serial(jobs)
        )

        energies = []
        for orig_cid, energy, positions in raw_results:
            if positions is not None:
                conf = mol.GetConformer(orig_cid)
                for i in range(n_solute):
                    conf.SetAtomPosition(i, positions[i].tolist())
            energies.append((orig_cid, energy))
        return energies

    def _run_serial(
        self, jobs: list[dict]
    ) -> list[tuple[int, float, np.ndarray | None]]:
        """Execute OpenMM conformer jobs sequentially in the current process."""
        results = []
        for j in jobs:
            try:
                results.append(_openmm_worker(**j))
            except Exception as exc:
                _logger.error(f"Conformer {j['original_cid']} failed: {exc}")
                results.append((j["original_cid"], float("nan"), None))
        return results

    def _run_parallel(
        self, jobs: list[dict]
    ) -> list[tuple[int, float, np.ndarray | None]]:
        """Execute OpenMM conformer jobs concurrently via ProcessPoolExecutor.

        Uses the ``spawn`` start method so each worker gets a clean process
        without inheriting CUDA state from the parent.  With NVIDIA MPS active,
        worker processes share the GPU's SMs concurrently.
        """
        ctx = multiprocessing.get_context("spawn")
        results = []
        with ProcessPoolExecutor(max_workers=self.gpu_streams, mp_context=ctx) as executor:
            futures = [executor.submit(_openmm_worker, **j) for j in jobs]
            for fut, j in zip(futures, jobs):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    cid = j["original_cid"]
                    _logger.error(f"Conformer {cid} parallel execution failed: {exc}")
                    results.append((cid, float("nan"), None))
        return results

    def _ps_to_steps(self, time_ps: float) -> int:
        """Convert a duration in picoseconds to an integer step count."""
        return int(time_ps * 1000 / self.md_timestep_fs)

    def _minimize_tblite(
        self, mol: Chem.Mol, conf_ids: list[int]
    ) -> list[tuple[int, float]]:
        """Optimize with GFN2-xTB / GFN1-xTB / IPEA1-xTB via tblite.

        Uses scipy L-BFGS-B for geometry optimization.  OpenMP
        parallelism is configured via ``num_threads``.
        """
        # Configure OpenMP threads for tblite's internal parallelism
        omp_threads = f"{self.num_threads},1"
        os.environ["OMP_NUM_THREADS"] = omp_threads
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

        method = self.ff_provider.get_tblite_method()

        atomic_nums = np.array(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32
        )
        charge = float(Chem.GetFormalCharge(mol))
        n_unpaired = 0  # assume closed-shell
        n_atoms = mol.GetNumAtoms()

        energies = []
        for cid in conf_ids:
            conf = mol.GetConformer(cid)
            # tblite wants positions in Bohr
            positions_ang = np.array(conf.GetPositions())
            positions_bohr = positions_ang / 0.52917721067

            calc = Calculator(
                method, atomic_nums, positions_bohr,
                charge=charge, uhf=n_unpaired,
            )
            calc.set("verbosity", 0)

            opt_bohr, energy_hartree = self._tblite_optimize_lbfgs(
                calc, positions_bohr, n_atoms, max_iters=self.max_iters
            )

            energy_kcal = energy_hartree * HARTREE_TO_KCALMOL
            opt_ang = opt_bohr * 0.52917721067

            for i in range(n_atoms):
                conf.SetAtomPosition(i, opt_ang[i].tolist())

            energies.append((cid, energy_kcal))

        return energies

    @staticmethod
    def _tblite_optimize_lbfgs(
        calc: Any,
        positions_bohr: np.ndarray,
        n_atoms: int,
        max_iters: int = 500,
        grad_tol: float = 1e-4,
    ) -> tuple[np.ndarray, float]:
        """L-BFGS-B geometry optimization using scipy + tblite singlepoints."""

        def func_and_grad(flat_pos: np.ndarray) -> tuple[float, np.ndarray]:
            pos = flat_pos.reshape(n_atoms, 3)
            calc.update(positions=pos)
            result = calc.singlepoint()
            energy = float(result["energy"])
            gradient = np.array(result["gradient"])
            return energy, gradient.ravel()

        res = scipy_minimize(
            func_and_grad,
            positions_bohr.ravel().copy(),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iters, "gtol": grad_tol},
        )
        opt_pos = res.x.reshape(n_atoms, 3)
        return opt_pos, float(res.fun)
