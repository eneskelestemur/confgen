"""Energy minimization dispatching across RDKit, OpenMM, and tblite backends."""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import openmm as mm
from openmm import app, unit
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
from scipy.optimize import minimize as scipy_minimize
from tblite.interface import Calculator

from confgen._constants import HARTREE_TO_KCALMOL, KJ_TO_KCAL
from confgen.forcefield import ForceFieldProvider
from confgen.solvation import is_explicit

_logger = logging.getLogger(__name__)


class Minimizer:
    """Minimize conformer energies using the appropriate backend."""

    # MD parameters: 0.1 ns with 0.5 fs timestep = 100,000 steps
    _MD_STEPS: int = 100_000
    _MD_TIMESTEP_FS: float = 0.5 * unit.femtosecond
    _OPENMM_TOLERANCE: float = 1.0 * unit.kilojoules_per_mole / unit.nanometer

    def __init__(
        self,
        ff_provider: ForceFieldProvider,
        max_iters: int = 500,
        num_threads: int = 1,
        platform: str = "CPU",
        seed: int = 42,
        solvent: str | None = None,
        run_md: bool = False,
    ):
        self.ff_provider = ff_provider
        self.max_iters = max_iters
        self.num_threads = num_threads
        self.platform = platform
        self.seed = seed
        self.solvent = solvent
        self.run_md = run_md

    def minimize(
        self,
        mol: Chem.Mol,
        conf_ids: list[int],
    ) -> list[tuple[int, float]]:
        """Minimize each conformer and return (conf_id, energy_kcal_mol) pairs.

        Conformers are modified in-place on the molecule.
        """
        backend = self.ff_provider.backend
        if backend == "rdkit":
            return self._minimize_rdkit(mol, conf_ids)
        if backend == "openmm":
            return self._minimize_openmm(mol, conf_ids)
        if backend == "tblite":
            return self._minimize_tblite(mol, conf_ids)
        raise ValueError(f"No minimizer for backend: {backend}")

    # ---- RDKit ----

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
            # Find position of cid in the molecule's conformer list
            all_cids = [c.GetId() for c in mol.GetConformers()]
            pos = all_cids.index(cid)
            _, energy_kcal = results[pos]
            energies.append((cid, energy_kcal))
        return energies

    # ---- OpenMM ----

    def _minimize_openmm(
        self, mol: Chem.Mol, conf_ids: list[int]
    ) -> list[tuple[int, float]]:
        """Minimize each conformer with OpenMM.

        For vacuum/implicit solvent the system is built once and conformer
        positions are swapped in.  For explicit solvent a fresh solvation
        box is constructed for every conformer so that water molecules are
        placed without bias from a previous minimization.
        """
        explicit = is_explicit(self.solvent)
        n_solute = mol.GetNumAtoms()

        if explicit:
            return self._minimize_openmm_explicit(mol, conf_ids, n_solute)
        return self._minimize_openmm_vacuum(mol, conf_ids)

    def _minimize_openmm_vacuum(
        self, mol: Chem.Mol, conf_ids: list[int]
    ) -> list[tuple[int, float]]:
        """Vacuum / implicit: build system once, swap solute positions per conformer."""
        system, modeller = self.ff_provider.build_openmm_system(
            mol, solvent=self.solvent
        )
        simulation = self._make_simulation(modeller.topology, system)

        energies = []
        for cid in conf_ids:
            positions = self._rdkit_conf_to_openmm_positions(mol, cid)
            simulation.context.setPositions(positions)
            if self.run_md:
                self._run_md_steps(simulation)
            simulation.minimizeEnergy(
                tolerance=self._OPENMM_TOLERANCE,
                maxIterations=self.max_iters,
            )
            state = simulation.context.getState(energy=True, positions=True)
            energy_kcal = (
                state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                * KJ_TO_KCAL
            )
            self._update_rdkit_conf(mol, cid, state, mol.GetNumAtoms())
            energies.append((cid, energy_kcal))
        return energies

    def _minimize_openmm_explicit(
        self, mol: Chem.Mol, conf_ids: list[int], n_solute: int
    ) -> list[tuple[int, float]]:
        """Explicit solvent: rebuild solvation box per conformer."""
        energies = []
        for cid in conf_ids:
            conf_src = mol.GetConformer(cid)
            # Create a single-conformer copy for system building
            new_mol = Chem.RWMol(mol)
            new_mol.RemoveAllConformers()
            new_conf = Chem.Conformer(n_solute)
            for i in range(n_solute):
                pt = conf_src.GetAtomPosition(i)
                new_conf.SetAtomPosition(i, pt)
            new_mol.AddConformer(new_conf, assignId=True)

            system, modeller = self.ff_provider.build_openmm_system(
                new_mol.GetMol(), solvent=self.solvent
            )
            simulation = self._make_simulation(modeller.topology, system)
            simulation.context.setPositions(modeller.positions)
            if self.run_md:
                self._run_md_steps(simulation)
            simulation.minimizeEnergy(
                tolerance=self._OPENMM_TOLERANCE,
                maxIterations=self.max_iters,
            )
            state = simulation.context.getState(energy=True, positions=True)
            energy_kcal = (
                state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                * KJ_TO_KCAL
            )
            # Write back only solute positions to the original mol
            self._update_rdkit_conf(mol, cid, state, n_solute)
            energies.append((cid, energy_kcal))
        return energies

    def _make_simulation(
        self, topology: Any, system: Any
    ) -> app.Simulation:
        """Create an OpenMM Simulation with the configured integrator/platform."""
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            self._MD_TIMESTEP_FS,
        )
        if self.seed >= 0:
            integrator.setRandomNumberSeed(self.seed)
        platform = mm.Platform.getPlatformByName(self.platform)
        return app.Simulation(topology, system, integrator, platform)

    def _run_md_steps(self, simulation: app.Simulation) -> None:
        """Run a short MD trajectory (0.1 ns, 0.5 fs timestep)."""
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, self.seed)
        simulation.step(self._MD_STEPS)
        _logger.debug(f"Completed {self._MD_STEPS} MD steps (0.1 ns)")

    @staticmethod
    def _update_rdkit_conf(
        mol: Chem.Mol,
        conf_id: int,
        state: Any,
        n_atoms: int,
    ) -> None:
        """Write minimized positions from an OpenMM state back to an RDKit conformer."""
        new_pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        conf = mol.GetConformer(conf_id)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, new_pos[i].tolist())

    # ---- tblite ----

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

    # ---- helpers ----

    @staticmethod
    def _rdkit_conf_to_openmm_positions(mol: Chem.Mol, conf_id: int) -> Any:
        """Convert RDKit conformer coordinates to OpenMM Quantity (nanometers)."""
        conf = mol.GetConformer(conf_id)
        positions = []
        for i in range(mol.GetNumAtoms()):
            pt = conf.GetAtomPosition(i)
            positions.append((pt.x * 0.1, pt.y * 0.1, pt.z * 0.1))
        return positions * unit.nanometer
