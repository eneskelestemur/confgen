from typing import Optional, List, Dict, Tuple
import logging

import numpy as np
import openmm as mm
from openmm import app, unit
from openmmforcefields.generators.template_generators import SmallMoleculeTemplateGenerator
from rcsbapi.data import DataQuery

from affinea.curator.ff_builder import ForceFieldBuilder
from affinea.curator.neighbor_search import NeighborSearch
from affinea.constants import METAL_RES_NAMES


class MinimizationError(Exception):
    """Custom exception for minimization errors."""
    pass


class Minimizer:
    """
        Pure OpenMM-based energy minimization for protein-ligand complexes.
    """
    
    def __init__(
        self, 
        complex_modeller: app.Modeller,
        ligand_chain_id: Optional[str] = None,
        structure_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
            Initialize Minimizer for a specific complex.
            
            Args:
                complex_modeller: Complex structure to minimize.
                ligand_chain_id: Chain ID for ligand. If None, will search for 'LIG' residue.
                structure_id: PDB ID or structure identifier for pH lookup.
                logger: Optional logger for debugging.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.ff_builder = ForceFieldBuilder(logger=self.logger)
        
        # Set structure attributes
        self.complex_modeller = complex_modeller
        self.structure_id = structure_id
        self.ligand_chain_id = self._determine_ligand_chain_id(ligand_chain_id)
        
        # Cache system for external access
        self.system = None

    def minimize_complex(
        self,
        ligand_template_generator: Optional[SmallMoleculeTemplateGenerator] = None,
        add_solvent: bool = False,
        max_iterations: int = 0,
        tolerance: float = 10.0,
        platform: str = "CPU",
        seed: int = 42,
        pocket_cutoff: float = 7.5,
        freeze_pocket_backbone: bool = True,
        freeze_metals: bool = True,
    ) -> app.Modeller:
        """
            Complete minimization workflow for the loaded complex.
            
            Args:
                ligand_template_generator: Template generator for ligand.
                add_solvent: Whether to add water molecules in minimization.
                max_iterations: Maximum iterations for minimization, 0 for convergence.
                tolerance: Energy tolerance for convergence in kJ/mol/nm.
                platform: Computing platform ("CPU", "CUDA", etc.).
                seed: Random seed for reproducibility.
                pocket_cutoff: Distance cutoff for freezing in Angstroms.
                freeze_pocket_backbone: Whether to freeze backbone atoms in pocket.
                freeze_metals: Whether to always freeze metal atoms.

            Returns:
                Minimized complex modeller.
                
            Raises:
                MinimizationError: If minimization fails.
        """
        try:
            system = self.setup_system(ligand_template_generator, add_solvent, pocket_cutoff)
            
            frozen_count = self.configure_freezing(
                system, pocket_cutoff, freeze_pocket_backbone, freeze_metals
            )
            self.logger.info(f"Frozen {frozen_count} atoms for minimization")
            
            minimized_modeller = self.run_minimization(
                system, platform, max_iterations, tolerance, seed
            )
            
            return minimized_modeller
            
        except Exception as e:
            raise MinimizationError(f"Minimization failed: {e}") from e

    def _determine_ligand_chain_id(self, ligand_chain_id: Optional[str]) -> str:
        """
            Determine ligand chain ID with fallback to LIG residue search.
            
            Args:
                ligand_chain_id: Provided chain ID or None for auto-detection.
                
            Returns:
                Valid ligand chain ID.
                
            Raises:
                MinimizationError: If ligand cannot be found.
        """
        if ligand_chain_id is not None:
            for chain in self.complex_modeller.topology.chains():
                if chain.id == ligand_chain_id:
                    return ligand_chain_id
            raise MinimizationError(f"Ligand chain '{ligand_chain_id}' not found in complex")
        
        # Search for LIG residue
        for chain in self.complex_modeller.topology.chains():
            for residue in chain.residues():
                if residue.name == 'LIG':
                    self.logger.info(f"Found LIG residue in chain '{chain.id}'")
                    return chain.id
        
        raise MinimizationError("No ligand found: provide ligand_chain_id or ensure LIG residue exists")

    def setup_system(
        self,
        ligand_template_generator: Optional[SmallMoleculeTemplateGenerator],
        add_solvent: bool,
        pocket_cutoff: float,
    ) -> mm.System:
        """
            Create OpenMM system with appropriate force fields.
            
            Args:
                ligand_template_generator: Template generator for ligand. None for custom template generation.
                add_solvent: Whether to add water molecules.
                pocket_cutoff: Pocket size for minimization.
                    
            Returns:
                Configured OpenMM System.
                
            Raises:
                MinimizationError: If system setup fails.
        """
        try:
            # Get pH from PDB (online -- RCSB-API) if structure_id is available
            ph = 7.0
            if self.structure_id:
                try:
                    ph_data = self.get_pdb_ph_rcsb(self.structure_id[:4].lower())
                    if ph_data.get("pH_values"):
                        ph = ph_data["pH_values"][0]  # Use first pH value
                        self.logger.info(f"Using pH {ph} from PDB {self.structure_id}")
                    else:
                        self.logger.info(f"No pH data found for PDB {self.structure_id}, using default pH 7.0")
                except Exception as e:
                    self.logger.warning(f"Failed to get pH for PDB {self.structure_id}: {e}, using default pH 7.0")
            
            self.complex_modeller.deleteWater()
            self.complex_modeller.addHydrogens(pH=ph)

            forcefield = self.ff_builder.build(self.complex_modeller.topology, ligand_template_generator)
            
            if add_solvent:
                self.complex_modeller.addSolvent(
                    forcefield, 
                    model='spce',
                    padding=0.5*unit.nanometer,
                    neutralize=False,
                    residueTemplates=self._get_fe_residue_templates(),
                )
                # Remove water molecules outside ligand bounding box
                self._remove_distant_water_molecules(pocket_cutoff + 2.0)
                        
            # Create system
            # NOTE: rigidWater=True causes an exception: "A constraint cannot involve a massless particle"
            system = forcefield.createSystem(
                self.complex_modeller.topology,
                rigidWater=False,
                residueTemplates=self._get_fe_residue_templates(),
                nonbondedMethod=app.NoCutoff,
                nonbondedCutoff=1.5*unit.nanometer,
            )
            
            return system
            
        except Exception as e:
            raise MinimizationError(f"System setup failed: {e}") from e
    
    def configure_freezing(
        self,
        system: mm.System,
        pocket_cutoff: float,
        freeze_pocket_backbone: bool,
        freeze_metals: bool,
    ) -> int:
        """
            Configure atom freezing based on spatial and chemical criteria.
            
            Args:
                system: OpenMM system to modify.
                pocket_cutoff: Distance cutoff for freezing in Angstroms.
                freeze_pocket_backbone: Whether to freeze backbone atoms in pocket.
                freeze_metals: Whether to freeze metal atoms.
                    
            Returns:
                Number of atoms frozen.
        """
        ligand_coords = self._get_ligand_coordinates()
        if not ligand_coords:
            self.logger.warning("No ligand atoms found for freezing configuration")
            return 0

        pocket_atom_indices = self._identify_pocket_atoms(
            ligand_coords, pocket_cutoff, True, True
        )
        
        frozen_count = self._apply_freezing_rules(
            system, pocket_atom_indices, freeze_pocket_backbone, freeze_metals
        )
        
        return frozen_count

    def run_minimization(
        self,
        system: mm.System,
        platform: str,
        max_iterations: int,
        tolerance: float,
        seed: int,
    ) -> app.Modeller:
        """
            Execute energy minimization simulation and return new Modeller.
            
            Args:
                system: Configured OpenMM system.
                platform: Computing platform name.
                max_iterations: Maximum minimization iterations.
                tolerance: Energy tolerance for convergence.
                seed: Random seed for reproducibility. 0 for random seed.

            Returns:
                Minimized complex modeller.
                
            Raises:
                MinimizationError: If minimization fails.
        """
        try:
            simulation = self._setup_simulation(system, platform, seed)
            
            state = simulation.context.getState(positions=True, energy=True)
            initial_energy = state.getPotentialEnergy()
            initial_positions = state.getPositions()
            
            simulation.minimizeEnergy(
                tolerance * unit.kilojoules_per_mole / unit.nanometer, 
                max_iterations
            )

            state = simulation.context.getState(positions=True, energy=True)
            final_positions = state.getPositions()
            final_energy = state.getPotentialEnergy()
            n_iters = state.getStepCount()

            # Cache the system for external access
            self.system = system
            
            self._log_minimization_progress(initial_energy, final_energy, n_iters)
            
            minimized_modeller = app.Modeller(self.complex_modeller.topology, final_positions)
            
            return minimized_modeller
            
        except Exception as e:
            raise MinimizationError(f"Minimization execution failed: {e}") from e

    def get_system(self) -> Optional[mm.System]:
        """
            Get the cached OpenMM system after minimization.
            
            Returns:
                The OpenMM system used for minimization, or None if not yet cached.
        """
        return self.system

    def _get_ligand_coordinates(self) -> List[List[float]]:
        """
            Extract ligand coordinates for spatial calculations.
                
            Returns:
                List of [x, y, z] coordinates in Angstroms.
        """
        ligand_coords = []
        for i, atom in enumerate(self.complex_modeller.topology.atoms()):
            if atom.residue.chain.id == self.ligand_chain_id:
                pos = self.complex_modeller.positions[i].value_in_unit(unit.angstrom)
                ligand_coords.append([pos[0], pos[1], pos[2]])
        return ligand_coords
    
    def _get_fe_residue_templates(self):
        """
            Get residue templates for iron (FE) based on existing residues.

            Returns:
                Dictionary of {residue: template_name} for FE residues.
        """
        metal_templates = {}
        for residue in self.complex_modeller.topology.residues():
            if 'fe' in residue.name.lower():
                metal_templates[residue] = 'FE2' if 'fe2' in residue.name.lower() else 'FE'
    
        return metal_templates

    def _remove_distant_water_molecules(self, padding: float = 10.0) -> None:
        """
            Remove water molecules outside ligand bounding box.
            
            Args:
                padding: Extra space around ligand bounding box in Angstroms.
        """
        # Get ligand bounding box
        ligand_coords = self._get_ligand_coordinates()
        if not ligand_coords:
            self.logger.warning("No ligand coordinates found for water removal")
            return
        
        coords_array = np.array(ligand_coords)
        min_coords = np.min(coords_array, axis=0) - padding
        max_coords = np.max(coords_array, axis=0) + padding
        
        waters_to_remove = []
        removed_count = 0
        for residue in self.complex_modeller.topology.residues():
            if residue.name in ['HOH', 'WAT', 'TIP3', 'TIP4', 'SPC']:
                # Get water center of mass (using Oxygen (first) atom as approximation)
                water_center = None
                for atom in residue.atoms():
                    pos = self.complex_modeller.positions[atom.index].value_in_unit(unit.angstrom)
                    water_center = np.array([pos[0], pos[1], pos[2]])
                    break

                if water_center is not None:
                    # Check if water is outside bounding box
                    if (np.any(water_center < min_coords) or np.any(water_center > max_coords)):
                        waters_to_remove.extend(list(residue.atoms()))
                        removed_count += 1
        
        if waters_to_remove:
            self.complex_modeller.delete(waters_to_remove)
            self.logger.info(f"Removed {removed_count} water molecules outside ligand region")
    
    def _identify_pocket_atoms(
        self,
        ligand_coords: List[List[float]],
        pocket_cutoff: float,
        expand_residues: bool = True,
        keep_water: bool = False,
    ) -> List[int]:
        """
            Identify atoms within pocket cutoff of ligand using NeighborSearch.
            
            Args:
                ligand_coords: Ligand coordinates in Angstroms.
                pocket_cutoff: Distance cutoff in Angstroms.
                expand_residues: Whether to include entire residues of pocket atoms.
                keep_water: Whether to keep water molecules in pocket residues.
                
            Returns:
                List of atom indices within pocket.
        """
        non_ligand_atoms = []
        non_ligand_coords = []
        for atom in self.complex_modeller.topology.atoms():
            if atom.residue.chain.id != self.ligand_chain_id:
                non_ligand_atoms.append(atom)
                pos = self.complex_modeller.positions[atom.index].value_in_unit(unit.angstrom)
                non_ligand_coords.append([pos[0], pos[1], pos[2]])
        
        if not non_ligand_atoms:
            return []
        
        # Use NeighborSearch to find pocket atom indices
        ns = NeighborSearch(np.array(non_ligand_coords), logger=self.logger)
        pocket_indices = ns.search_multiple_within_cutoff(
            coords=np.array(ligand_coords),
            cutoff=pocket_cutoff
        )
        
        # Convert back to atom objects
        pocket_atoms = set(non_ligand_atoms[i] for i in pocket_indices)
        
        if expand_residues:
            pocket_residues = set(atom.residue for atom in pocket_atoms)
            expanded_pocket_atoms = set()
            for residue in pocket_residues:
                if not keep_water and residue.name in ['HOH', 'WAT', 'TIP3', 'TIP4', 'SPC']:
                    continue
                expanded_pocket_atoms.update(residue.atoms())
            return [atom.index for atom in expanded_pocket_atoms]
        else:
            return [atom.index for atom in pocket_atoms]
    
    def _apply_freezing_rules(
        self,
        system: mm.System,
        pocket_atom_indices: List[int],
        freeze_pocket_backbone: bool,
        freeze_metals: bool
    ) -> int:
        """
            Apply freezing rules to determine which atoms to freeze.
            
            Args:
                system: OpenMM system to modify.
                pocket_atom_indices: Indices of atoms in binding pocket.
                freeze_pocket_backbone: Whether to freeze backbone atoms in pocket.
                freeze_metals: Whether to freeze metal atoms.
                
            Returns:
                Number of atoms frozen.
        """
        frozen_count = 0
        pocket_indices_set = set(pocket_atom_indices)
        
        for atom in self.complex_modeller.topology.atoms():
            should_freeze = False
            
            # Never freeze ligand atoms
            if atom.residue.chain.id == self.ligand_chain_id:
                should_freeze = False
            # Freeze metal atoms if enabled
            elif freeze_metals and atom.residue.name in METAL_RES_NAMES:
                should_freeze = True
            # Freeze atoms outside pocket
            elif atom.index not in pocket_indices_set:
                should_freeze = True
            # For pocket residues, optionally freeze backbone
            elif freeze_pocket_backbone and atom.name in ['CA', 'C', 'N', 'O']:
                should_freeze = True
                
            if should_freeze:
                system.setParticleMass(atom.index, 0.0)
                frozen_count += 1
                
        return frozen_count
    
    def _setup_simulation(
        self,
        system: mm.System,
        platform_name: str,
        seed: int
    ) -> app.Simulation:
        """
            Setup OpenMM simulation for minimization.
            
            Args:
                system: Configured OpenMM system.
                platform_name: Computing platform name.
                seed: Random seed for reproducibility.
                
            Returns:
                Configured OpenMM simulation.
        """
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0/unit.picosecond, 0.5 * unit.femtosecond
        )
        integrator.setRandomNumberSeed(seed)
        
        platform = mm.Platform.getPlatformByName(platform_name)
        simulation = app.Simulation(self.complex_modeller.topology, system, integrator, platform)
        simulation.context.setPositions(self.complex_modeller.positions)
        
        return simulation
    
    def _log_minimization_progress(
        self,
        initial_energy: unit.Quantity,
        final_energy: unit.Quantity,
        iterations: int,
    ) -> None:
        """
            Log minimization progress and results (economic logging).
            
            Args:
                initial_energy: Initial potential energy.
                final_energy: Final potential energy.
                iterations: Number of iterations performed.
        """
        energy_change = final_energy - initial_energy
        self.logger.info(f"Minimized {iterations} steps: "
                         f"ΔΔE = {str(energy_change)}")

    @staticmethod
    def get_pdb_ph_rcsb(pdb_id: str) -> dict:
        """
            Look up pH for a PDB entry via RCSB Data API.
            
            Args:
                pdb_id: PDB identifier to query.
                
            Returns:
                Dictionary containing:
                - pdb_id: The queried PDB ID
                - methods: List of experimental methods
                - pH_values: List of pH values (sorted, unique) or None
                - sources: List of data sources for pH values
        """
        try:
            pid = pdb_id.strip().upper()
            
            # Query RCSB Data API for pH information
            query = DataQuery(
                input_type="entries",
                input_ids=[pid],
                return_data_list=[
                    "exptl.method",
                    "exptl_crystal_grow.pH",               # Crystallography pH
                    "pdbx_nmr_exptl_sample_conditions.pH", # NMR sample pH
                ],
            )
            
            data = query.exec()
            entries = data.get("data", {}).get("entries", [])
            
            if not entries:
                return {"pdb_id": pid, "methods": [], "pH_values": None, "sources": []}

            entry = entries[0]
            
            # Extract experimental methods
            methods = [
                method_data.get("method")
                for method_data in (entry.get("exptl") or [])
                if isinstance(method_data, dict) and method_data.get("method")
            ]

            # Extract pH values from different sources
            ph_values, sources = [], []

            # Crystallography pH
            for crystal_data in (entry.get("exptl_crystal_grow") or []):
                ph_value = crystal_data.get("pH")
                if ph_value is not None:
                    try:
                        ph_values.append(float(ph_value))
                        sources.append("exptl_crystal_grow")
                    except (TypeError, ValueError):
                        continue

            # NMR sample pH
            for nmr_data in (entry.get("pdbx_nmr_exptl_sample_conditions") or []):
                ph_value = nmr_data.get("pH")
                if ph_value is not None:
                    try:
                        ph_values.append(float(ph_value))
                        sources.append("pdbx_nmr_exptl_sample_conditions")
                    except (TypeError, ValueError):
                        continue

            return {
                "pdb_id": entry.get("rcsb_id", pid),
                "methods": methods,
                "pH_values": sorted(set(ph_values)) if ph_values else None,
                "sources": sorted(set(sources)),
            }
            
        except Exception:
            return {"pdb_id": pdb_id, "methods": [], "pH_values": None, "sources": []}
