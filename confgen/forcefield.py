"""Force field abstraction: unified interface for MMFF, UFF, GAFF, SMIRNOFF, Espaloma, and tblite."""
from __future__ import annotations

import logging
from typing import Any

from rdkit import Chem

from confgen._constants import (
    DEFAULT_ESPALOMA_MODEL,
    DEFAULT_GAFF_VERSION,
    DEFAULT_SMIRNOFF_FF,
    OPENMM_FORCEFIELDS,
    RDKIT_FORCEFIELDS,
    TBLITE_METHODS,
)

_logger = logging.getLogger(__name__)


class ForceFieldProvider:
    """Unified factory that dispatches to the appropriate FF backend.

    Calling code only asks for a provider by name; the provider knows how to
    set up the backend and return an object suitable for the Minimizer.
    """

    def __init__(self, name: str):
        self.name = name.lower()

    @property
    def backend(self) -> str:
        if self.name in RDKIT_FORCEFIELDS:
            return "rdkit"
        if self.name in OPENMM_FORCEFIELDS:
            return "openmm"
        if self.name in TBLITE_METHODS:
            return "tblite"
        raise ValueError(f"Unknown forcefield: {self.name}")

    # ---- RDKit helpers ----

    def has_rdkit_params(self, mol: Chem.Mol) -> bool:
        """Check if the molecule has the required RDKit FF parameters."""
        from rdkit.Chem import rdForceFieldHelpers

        if self.name == "mmff":
            return rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol)
        if self.name == "uff":
            return rdForceFieldHelpers.UFFHasAllMoleculeParams(mol)
        return False

    # ---- OpenMM helpers ----

    def _make_template_generator(self, off_mol: Any) -> Any:
        if self.name == "gaff":
            from openmmforcefields.generators.template_generators import (
                GAFFTemplateGenerator,
            )
            return GAFFTemplateGenerator(off_mol, forcefield=DEFAULT_GAFF_VERSION)

        if self.name == "smirnoff":
            from openmmforcefields.generators.template_generators import (
                SMIRNOFFTemplateGenerator,
            )
            return SMIRNOFFTemplateGenerator(off_mol, forcefield=DEFAULT_SMIRNOFF_FF)

        if self.name == "espaloma":
            from openmmforcefields.generators.template_generators import (
                EspalomaTemplateGenerator,
            )
            return EspalomaTemplateGenerator(off_mol, forcefield=DEFAULT_ESPALOMA_MODEL)

        raise ValueError(f"No OpenMM template generator for: {self.name}")

    def build_openmm_system(
        self,
        mol: Chem.Mol,
        solvent: str | None = None,
        padding_nm: float = 0.5,
        ionic_strength_molar: float = 0.0,
    ) -> tuple[Any, Any]:
        """Build an OpenMM System for a small molecule.

        Supports vacuum (solvent=None), implicit GB models, and explicit
        water-box solvation with optional ions.

        Returns:
            (system, modeller) — OpenMM System and Modeller with positions set.
        """
        from openmm import app, unit
        from openff.toolkit import Molecule as OFFMol

        from confgen.solvation import add_explicit_solvent, get_solvent_xmls, is_explicit

        off_mol = OFFMol.from_rdkit(mol, allow_undefined_stereo=True)
        off_top = off_mol.to_topology()
        omm_top = off_top.to_openmm()
        positions = off_top.get_positions().to_openmm()

        template_gen = self._make_template_generator(off_mol)

        solvent_xmls = get_solvent_xmls(solvent)
        ff = app.ForceField(*solvent_xmls) if solvent_xmls else app.ForceField()
        ff.registerTemplateGenerator(template_gen.generator)

        modeller = app.Modeller(omm_top, positions)

        explicit = is_explicit(solvent)
        if explicit:
            add_explicit_solvent(
                modeller, ff, solvent, padding_nm, ionic_strength_molar
            )
            system = ff.createSystem(
                modeller.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                rigidWater=True,
            )
        else:
            system = ff.createSystem(
                modeller.topology,
                nonbondedMethod=app.NoCutoff,
                rigidWater=False,
            )

        return system, modeller

    # ---- tblite helpers ----

    def get_tblite_method(self) -> str:
        """Return the tblite method string for the Calculator."""
        from confgen._constants import TBLITE_METHOD_MAP
        return TBLITE_METHOD_MAP[self.name]
