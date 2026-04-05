"""
    A builder class responsible for constructing an OpenMM ForceField object.
"""
from typing import List, Set, Dict, Optional
import logging

from openmm import app
from openmm.app.forcefield import NonbondedGenerator, _createResidueSignature
from openmmforcefields.generators.template_generators import SmallMoleculeTemplateGenerator
try:
    from openmm.app.internal import compiled
    matchResidueToTemplate = compiled.matchResidueToTemplate
except ImportError:
    matchResidueToTemplate = app.forcefield._matchResidue

from affinea.constants import (
    TARGET_FORCE_FIELDS, IMPLICIT_SOLVENT_MODELS, 
    EXPLICIT_SOLVENT_MODELS, ATOM_RADII
)


class ForceFieldError(Exception):
    """Custom exception for force field building errors."""
    pass


class ForceFieldBuilder:
    """
        Build an OpenMM ForceField, auto-generating custom templates for any
        residues in the topology that lack built-in parameters.
    """
    def __init__(
        self, 
        ff_files: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
            Args:
                ff_files: List of force field files to use. If None, uses defaults.
                logger: Optional logger for debugging.
        """
        self.logger = logger or logging.getLogger(__name__)
        self._ff_files: List[str] = ff_files or TARGET_FORCE_FIELDS

    def build(
            self, 
            topology: app.Topology, 
            ligand_template_generator: Optional[SmallMoleculeTemplateGenerator] = None,
            explicit_solvent: bool = False
        ) -> app.ForceField:
        """
            Create and configure an OpenMM ForceField for `topology`, registering
            custom templates for all unrecognized residues.

            Args:
                topology: The OpenMM Topology of your system.
                ligand_template_generator: Optional SmallMoleculeTemplateGenerator for ligand residues.
                explicit_solvent: Whether to include explicit solvent molecules.

            Returns:
                A fully configured app.ForceField.
                
            Raises:
                ForceFieldError: If force field creation fails.
        """
        try:
            if explicit_solvent:
                self._ff_files.extend(EXPLICIT_SOLVENT_MODELS)
            else:
                self._ff_files.extend(IMPLICIT_SOLVENT_MODELS)
            self.logger.debug(f"Building ForceField with files: {self._ff_files}")
            ff = app.ForceField(*self._ff_files)
            nonbonded = self._find_nonbonded_generator(ff)
            bond_map = self._build_bond_map(topology)
            lig_temp_exists = False
            
            for residue in topology.residues():
                sig = _createResidueSignature([atom.element for atom in residue.atoms()])

                # Skip ligand residue
                if residue.name.upper() in ['LIG', 'UNL', 'UNK']:
                    # Check if ligand has already an existing template (rare but possible)
                    if self._residue_matches_template(residue, sig, ff, bond_map):
                        lig_temp_exists = True  # No need for the generator
                        self.logger.debug(f"Found existing template in forcefield for ligand {residue.name}")
                        continue

                    if ligand_template_generator is not None:
                        success = ligand_template_generator.generator(ff, residue)
                        
                        if success:
                            continue
                        else:
                            self.logger.warning(
                                f"Failed to register ligand residue {residue.name} with template generator. "
                                f"Make sure the template generator contains the ligand molecule. "
                                f"Falling back to custom template generation on-the-fly."
                            )
                    else:
                        self.logger.warning(
                            f"Encountered a ligand residue, {residue.name}, without template generator. "
                            f"Consider providing a template generator for ligand residues."
                        )

                if not self._residue_matches_template(residue, sig, ff, bond_map):
                    self._create_template_data(residue, sig, ff, nonbonded, bond_map)
                    self.logger.debug(f"Registered custom template for residue {residue.name} with signature {sig}")

            # Register ligand template generator
            if ligand_template_generator is not None and not lig_temp_exists:
                ff.registerTemplateGenerator(ligand_template_generator.generator)
                self.logger.debug(f"Registered ligand template generator for {ligand_template_generator._molecules}")
            return ff
            
        except Exception as e:
            raise ForceFieldError(f"Failed to build force field: {e}")

    def _find_nonbonded_generator(self, ff: app.ForceField) -> NonbondedGenerator:
        """
            Find the NonbondedGenerator in the force field.

            Args:
                ff: The OpenMM ForceField to search.

            Returns:
                The NonbondedGenerator instance if found.

            Raises:
                RuntimeError: If no NonbondedGenerator is found.
        """
        for gen in ff._forces:
            if isinstance(gen, NonbondedGenerator):
                return gen
        raise RuntimeError("No NonbondedGenerator found in the force field")

    def _build_bond_map(self, topology: app.Topology) -> List[Set[int]]:
        """
            Create a per-atom set of bonded neighbors for quick lookup.

            Args:
                topology: The OpenMM Topology of your system.

            Returns:
                A list where each index corresponds to an atom and contains 
                a set of indices of atoms that are bonded to it.
        """
        n = topology.getNumAtoms()
        bond_map: List[Set[int]] = [set() for _ in range(n)]
        for atom1, atom2 in topology.bonds():
            bond_map[atom1.index].add(atom2.index)
            bond_map[atom2.index].add(atom1.index)
        return bond_map

    def _residue_matches_template(
        self,
        residue: app.Residue,
        signature: str,
        ff: app.ForceField,
        bond_map: List[Set[int]],
    ) -> bool:
        """
            Return True if `residue` matches any existing template in `ff`.

            Args:
                residue: The OpenMM Residue to check.
                signature: The residue signature to match against templates.
                ff: The OpenMM ForceField containing templates.
                bond_map: A list mapping each atom index to a set of bonded atom indices.

            Returns:
                True if the residue matches any template, False otherwise.
        """
        if signature not in ff._templateSignatures:
            return False
        for tpl in ff._templateSignatures[signature]:
            if matchResidueToTemplate(residue, tpl, bond_map) is not None:
                return True
        return False
    
    def _create_template_data(
        self,
        residue: app.Residue,
        signature: str,
        ff: app.ForceField,
        nonbonded: NonbondedGenerator,
        bond_map: List[Set[int]],
    ) -> None:
        """
            Generate and register a custom template for a non-standard residue.

            Args:
                residue: The OpenMM Residue to create a template for.
                signature: The residue signature to use for the template.
                ff: The OpenMM ForceField to register the template with.
                nonbonded: The NonbondedGenerator to register new atom types with.
                bond_map: A list mapping each atom index to a set of bonded atom indices.
        """
        template = app.ForceField._TemplateData(f"extra_{residue.name}")
        ff._templates[f"extra_{residue.name}"] = template

        # Create atoms for the template
        index_map: Dict[int, int] = {}
        for atom in residue.atoms():
            idx = atom.index
            element = atom.element
            type_name = f"extra_{element.symbol}_0"

            if type_name not in ff._atomTypes:
                atom_type = app.ForceField._AtomType(
                    type_name, f"extra_{element.symbol}", 0.0, element
                )
                ff._atomTypes[type_name] = atom_type
                sigma = ATOM_RADII.get(element.symbol, 0.5)
                nonbonded.registerAtom({
                    'type': type_name,
                    'charge': '0',
                    'sigma': str(sigma),
                    'epsilon': '0'
                })
            
            index_map[idx] = len(template.atoms)
            template.atoms.append(
                app.ForceField._TemplateAtomData(atom.name, type_name, element)
            )

        # Create bonds and update atom-level bond records
        for atom in residue.atoms():
            i = index_map[atom.index]       # i: index in the template
            for j in bond_map[atom.index]:  # j: index of bonded atom in the original topology
                if j in index_map: 
                    k = index_map[j]        # k: index of bonded atom in the template
                    if i < k:
                        template.bonds.append((i, k))
                        template.atoms[i].bondedTo.append(k)
                        template.atoms[k].bondedTo.append(i)
                else:
                    template.externalBonds.append(i)
                    template.atoms[i].externalBonds += 1

        # Register the new template
        ff._templateSignatures.setdefault(signature, []).append(template)

