"""Energy minimization dispatching across RDKit, OpenMM, and tblite backends."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from confgen._constants import HARTREE_TO_KCALMOL, KJ_TO_KCAL
from confgen.forcefield import ForceFieldProvider

_logger = logging.getLogger(__name__)


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
    ):
        self.ff_provider = ff_provider
        self.max_iters = max_iters
        self.num_threads = num_threads
        self.platform = platform
        self.seed = seed
        self.solvent = solvent

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
        """Build an OpenMM system once, minimize each conformer by updating positions."""
        import openmm as mm
        from openmm import app, unit

        from confgen.solvation import is_explicit

        explicit = is_explicit(self.solvent)
        n_solute = mol.GetNumAtoms()

        system, modeller = self.ff_provider.build_openmm_system(
            mol, solvent=self.solvent
        )

        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1.0 / unit.picosecond,
            0.5 * unit.femtosecond,
        )
        if self.seed >= 0:
            integrator.setRandomNumberSeed(self.seed)

        platform = mm.Platform.getPlatformByName(self.platform)
        simulation = app.Simulation(modeller.topology, system, integrator, platform)

        # For explicit solvent, cache water/ion positions (nm) and update after
        # each minimization so the next conformer starts from relaxed solvent.
        if explicit:
            init_pos_nm = np.array(
                modeller.positions.value_in_unit(unit.nanometer)
            )
            water_pos_nm = init_pos_nm[n_solute:]

        energies = []
        for cid in conf_ids:
            if explicit:
                solute_nm = self._conf_positions_nm(mol, cid)
                full_pos = np.vstack([solute_nm, water_pos_nm])
                simulation.context.setPositions(full_pos * unit.nanometer)
            else:
                positions = self._rdkit_conf_to_openmm_positions(mol, cid)
                simulation.context.setPositions(positions)

            simulation.minimizeEnergy(
                tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
                maxIterations=self.max_iters,
            )
            state = simulation.context.getState(energy=True, positions=True)
            energy_kj = state.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            energy_kcal = energy_kj * KJ_TO_KCAL

            # Write minimized solute positions back to RDKit conformer
            new_pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            conf = mol.GetConformer(cid)
            for i in range(n_solute):
                conf.SetAtomPosition(i, new_pos[i].tolist())

            # Carry relaxed water positions to the next conformer
            if explicit:
                new_pos_nm = state.getPositions(asNumpy=True).value_in_unit(
                    unit.nanometer
                )
                water_pos_nm = new_pos_nm[n_solute:]

            energies.append((cid, energy_kcal))

        return energies

    # ---- tblite ----

    def _minimize_tblite(
        self, mol: Chem.Mol, conf_ids: list[int]
    ) -> list[tuple[int, float]]:
        """Optimize with GFN2-xTB / GFN1-xTB / IPEA1-xTB via tblite."""
        from tblite.interface import Calculator

        method = self.ff_provider.get_tblite_method()

        atomic_nums = np.array(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32
        )
        charge = float(Chem.GetFormalCharge(mol))
        n_unpaired = 0  # assume closed-shell

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
            if self.max_iters > 0:
                calc.set("max-iter", self.max_iters)

            # Gradient-based geometry optimization
            positions_bohr, energy_hartree = self._tblite_optimize(
                calc, positions_bohr, max_steps=self.max_iters
            )

            energy_kcal = energy_hartree * HARTREE_TO_KCALMOL
            positions_ang = positions_bohr * 0.52917721067

            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, positions_ang[i].tolist())

            energies.append((cid, energy_kcal))

        return energies

    @staticmethod
    def _tblite_optimize(
        calc: Any,
        positions_bohr: np.ndarray,
        max_steps: int = 500,
        grad_tol: float = 1e-4,
        step_size: float = 0.5,
    ) -> tuple[np.ndarray, float]:
        """Steepest-descent geometry optimization using tblite singlepoint calls."""
        pos = positions_bohr.copy()
        energy = None
        for _ in range(max_steps):
            calc.update(positions=pos)
            result = calc.singlepoint()
            energy = result["energy"]
            grad = result["gradient"]
            rms_grad = np.sqrt(np.mean(grad**2))
            if rms_grad < grad_tol:
                break
            pos -= step_size * grad
        return pos, energy

    # ---- helpers ----

    @staticmethod
    def _conf_positions_nm(mol: Chem.Mol, conf_id: int) -> np.ndarray:
        """Return conformer positions as a numpy array in nanometers."""
        conf = mol.GetConformer(conf_id)
        return np.array([
            [conf.GetAtomPosition(i).x * 0.1,
             conf.GetAtomPosition(i).y * 0.1,
             conf.GetAtomPosition(i).z * 0.1]
            for i in range(mol.GetNumAtoms())
        ])

    @staticmethod
    def _rdkit_conf_to_openmm_positions(mol: Chem.Mol, conf_id: int) -> Any:
        """Convert RDKit conformer coordinates to OpenMM Quantity (nanometers)."""
        from openmm import unit

        conf = mol.GetConformer(conf_id)
        positions = []
        for i in range(mol.GetNumAtoms()):
            pt = conf.GetAtomPosition(i)
            positions.append((pt.x * 0.1, pt.y * 0.1, pt.z * 0.1))
        return positions * unit.nanometer
