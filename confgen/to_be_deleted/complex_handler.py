from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict
import logging
from io import StringIO
import warnings

import numpy as np
from pdbfixer import PDBFixer
from openmm import app, unit
from openmmforcefields.generators.template_generators import (
    SmallMoleculeTemplateGenerator,
    GAFFTemplateGenerator,
    SMIRNOFFTemplateGenerator,
)
from openff.toolkit import Molecule as OpenFFMolecule
from openff.toolkit.topology import Topology as OpenFFTopology
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds as rdb

from affinea.curator.neighbor_search import NeighborSearch
from affinea.constants import PDB_CHAIN_IDS, LIGAND_FORCE_FIELDS

# Espaloma is optional - lazy import at usage time
EspalomaTemplateGenerator = None
_espaloma_import_error: Optional[str] = None


def _get_espaloma_generator():
    """Lazy import Espaloma with helpful error message."""
    global EspalomaTemplateGenerator, _espaloma_import_error
    
    if EspalomaTemplateGenerator is not None:
        return EspalomaTemplateGenerator
    
    if _espaloma_import_error is not None:
        raise ImportError(_espaloma_import_error)
    
    try:
        from openmmforcefields.generators.template_generators import EspalomaTemplateGenerator as _EspalomaGen
        EspalomaTemplateGenerator = _EspalomaGen
        return EspalomaTemplateGenerator
    except ImportError:
        _espaloma_import_error = (
            "Espaloma is not installed. To use the 'espaloma' ligand force field, "
            "please install espaloma following the instructions at: "
            "https://github.com/choderalab/espaloma#installation"
        )
        raise ImportError(_espaloma_import_error)


def split_complex_file(
        complex_file: Union[Path, str],
        structure_id: Optional[str] = None,
        ligand_chain_id: Optional[str] = None,
        ligand_smiles: Optional[str] = None,
        make_folder: bool = True,
) -> Dict[str, str]:
    """
        Convenience function to split a complex file into its target and
        ligand components, separately.

        Args:
            complex_file: Path to the complex file to split.
            structure_id: Optional structure ID that will be used in naming,
                otherwise file name will be used along with appropriate suffixes.
            ligand_chain_id: Ligand chain ID to identify the ligand in the complex. 
                If not given, smallest chain with single residue will be selected 
                as the ligand chain.
            ligand_smiles: Optional SMILES string for the ligand. If provided,
                it will be used to create the OpenFF molecule for the ligand.
                This is extremely useful for parsing and validating the ligand structure.
            make_folder: Whether to make a new folder for the output files.

        Returns:
            Dict[str, str]: A dictionary with paths to the 'target' and 'ligand' files.
    """
    complex_file = Path(complex_file) if not isinstance(complex_file, Path) else complex_file
    if structure_id is None:
        structure_id = complex_file.stem

    handler = ComplexHandler()
    complex_struct = handler._load_target(complex_file)

    if ligand_chain_id is None:
        smallest_chain_id = None
        smallest_chain_size = float('inf')
        for chain in complex_struct.topology.chains():
            n_res = len(list(chain.residues()))
            n_atoms = len(list(chain.atoms()))
            if n_res == 1:
                if smallest_chain_id is None or n_atoms < smallest_chain_size:
                    smallest_chain_id = chain.id
                    smallest_chain_size = n_atoms
        ligand_chain_id = smallest_chain_id

    handler.load_from_modeller(complex_struct, ligand_chain_id, structure_id)

    if make_folder:
        output_folder = complex_file.parent / f"{structure_id}"
        output_folder.mkdir(exist_ok=True)

    target_file = output_folder / f"{structure_id}_target.pdb"
    ligand_file = output_folder / f"{structure_id}_ligand.sdf"

    if ligand_smiles is not None:
        ligand_modeller = handler.extract_ligand(keep_water=False)

        # AF3 does not have bond info in the cif files, so we need to infer it
        if ligand_modeller.topology.getNumBonds() == 0:
            from rdkit.Geometry import Point3D
            print(f"Warning: Input structure is missing ligand bonding information, "
                  f"attempting to infer bonds")
            ligand_mol = Chem.MolFromSmiles(ligand_smiles)
            # check if atom order matches
            ligand_mol = Chem.RemoveAllHs(ligand_mol)
            rdkit_atoms = [a for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1]
            modeller_atoms = [a for a in ligand_modeller.topology.atoms() if a.element.symbol != "H"]
            conformer = Chem.Conformer(ligand_mol.GetNumHeavyAtoms())
            for i, (r_atom, m_atom) in enumerate(zip(rdkit_atoms, modeller_atoms)):
                if r_atom.GetSymbol() != m_atom.element.symbol:
                    raise ComplexProcessingError(
                        f"Atom mismatch at index {i}: RDKit {r_atom.GetSymbol()} vs "
                        f"Modeller {m_atom.element.symbol}. Cannot infer bonds."
                    )
                pos = ligand_modeller.positions[m_atom.index].value_in_unit(unit.angstrom)
                conformer.SetAtomPosition(i, Point3D(*pos))
            ligand_mol.AddConformer(conformer, assignId=True)
            Chem.AssignStereochemistryFrom3D(ligand_mol)

            handler.write_target(target_file, keep_water=True)
            Chem.SDWriter(str(ligand_file)).write(ligand_mol)
            return {
                "target": str(target_file),
                "ligand": str(ligand_file)
            }

    handler.write_target(target_file, keep_water=True)
    handler.write_ligand(ligand_file, keep_water=False)

    return {
        "target": str(target_file),
        "ligand": str(ligand_file)
    }


class ComplexProcessingError(Exception):
    """Custom exception for complex processing errors."""
    pass


class ComplexHandler:
    """
        Pure OpenMM-based structure handler for protein-ligand complexes.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
            Initialize ComplexHandler with minimal dependencies.
            
            Args:
                logger: Optional logger for debugging.
        """
        self.logger = logger or logging.getLogger(__name__)
        self._reset()
    
    def _reset(self) -> None:
        """Reset all cached state to free memory between complexes."""
        self.structure_id = None                # PDB id of the loaded structures
        self.complex_modeller = None            # Current complex state
        self.ligand_openff_molecule = None      # Cached OpenFF molecule for the ligand
        self.ligand_chain_id = None             # Cached ligand chain ID
    
    def load_from_files(
        self, 
        target_file: Union[Path, str], 
        ligand_file: Union[Path, str], 
        structure_id: Optional[str] = None,
        fix_protein: bool = True,
    ) -> None:
        """
            Load target and ligand from separate files, combine into complex.
            
            Args:
                target_file: Path to target PDB/PDBx file.
                ligand_file: Path to ligand SDF/MOL2/PDB file.
                structure_id: Unique identifier for the complex. If not provided, 
                    file name before the pattern will be assigned as structure_id.
                fix_protein: Whether to add missing atoms and residues to the protein. 

            Raises:
                ComplexProcessingError: If files cannot be loaded or combined.
        """
        target_file = Path(target_file) if not isinstance(target_file, Path) else target_file
        ligand_file = Path(ligand_file) if not isinstance(ligand_file, Path) else ligand_file

        try:
            self.logger.info(f"Loading complex from {target_file.name} + {ligand_file.name}")
            
            target_modeller = self._load_target(target_file, fix=fix_protein)
            ligand_molecule = self._load_ligand(ligand_file)

            self.structure_id = structure_id or str(target_file.stem).split('_')[0]
            self.complex_modeller, self.ligand_chain_id = self._combine_structures(
                target_modeller, ligand_molecule
            )
            self.ligand_openff_molecule = ligand_molecule
            
        except Exception as e:
            raise ComplexProcessingError(f"Failed to load complex from files: {e}") from e

    def load_from_modeller(
        self, 
        complex_modeller: app.Modeller,
        ligand_chain_id: str,
        structure_id: str
    ) -> None:
        """
            Initialize from existing Modeller object (e.g., from external source).
            
            Args:
                complex_modeller: Pre-existing complex structure as Modeller.
                ligand_chain_id: Chain id containing the ligand molecule.
                structure_id: Unique identifier for the complex.
        """
        try:
            found_ligand = False
            for ch in complex_modeller.topology.chains():
                if ch.id == ligand_chain_id:
                    lig_res = list(ch.residues())
                    if len(lig_res) != 1:
                        raise ComplexProcessingError(
                            f"Ligand chain '{ligand_chain_id}' must have exactly one residue but "
                            f"got {len(lig_res)}: {lig_res}"
                        )
                    self.logger.debug(f"Loaded ligand residue: {lig_res[0].name}")
                    self.ligand_chain_id = ligand_chain_id
                    found_ligand = True
                    break
            
            if not found_ligand:
                raise ComplexProcessingError(f"Ligand chain '{ligand_chain_id}' not found in complex")
            
            self.structure_id = structure_id
            self.complex_modeller = self._copy_modeller(complex_modeller)
            self.logger.info(
                f"Complex loaded from Modeller object. Ligand OpenFF molecule not set - "
                f"use `set_ligand_openff_molecule` if needed."
            )

        except Exception as e:
            raise ComplexProcessingError(f"Failed to load complex from Modeller: {e}") from e

    def get_ligand_template_generator(
        self, 
        template_generator: str = "gaff"
    ) -> SmallMoleculeTemplateGenerator:
        """
            Create SmallMoleculeTemplateGenerator for the loaded ligand.
            This is needed by Minimizer for force field setup.
            
            Args:
                template_generator: OpenFF template generator forcefield, 
                    'gaff', 'smirnoff' or 'espaloma'.
                
            Returns:
                Configured SmallMoleculeTemplateGenerator for ligand.
                
            Raises:
                ComplexProcessingError: If no complex is loaded or ligand not found.
        """
        self._check_ligand_molecule_set()
        
        if template_generator not in LIGAND_FORCE_FIELDS:
            raise ComplexProcessingError(f"Unknown template generator: {template_generator}. "
                                       f"Available: {list(LIGAND_FORCE_FIELDS.keys())}")
        
        try:
            ff = LIGAND_FORCE_FIELDS[template_generator]
            if template_generator == "gaff":
                generator = GAFFTemplateGenerator(self.ligand_openff_molecule, forcefield=ff)
            elif template_generator == "smirnoff":
                generator = SMIRNOFFTemplateGenerator(self.ligand_openff_molecule, forcefield=ff)
            elif template_generator == "espaloma":
                EspalomaGen = _get_espaloma_generator()
                generator = EspalomaGen(
                    self.ligand_openff_molecule, forcefield=ff,
                    template_generator_kwargs={'charge_method': 'nn'}
                )
            
            self.logger.info(f"Created {template_generator} template generator")
            return generator
            
        except Exception as e:
            raise ComplexProcessingError(f"Failed to create template generator: {e}") from e

    def get_complex(self) -> app.Modeller:
        """
            Get copy of current complex structure.
            
            Returns:
                Copy of the complex modeller.
                
            Raises:
                ComplexProcessingError: If no complex is loaded.
        """
        self._check_complex_loaded()
        return self._copy_modeller(self.complex_modeller)
    
    def set_complex(self, updated_modeller: app.Modeller) -> None:
        """
            Update internal complex with new Modeller (e.g., after minimization).
            
            Args:
                updated_modeller: Updated complex structure.
        """
        self.complex_modeller = self._copy_modeller(updated_modeller)
        self.logger.info("Complex updated")

    def get_ligand_openff_molecule(self) -> OpenFFMolecule:
        """
            Get the OpenFF molecule for the ligand.

            Returns:
                OpenFF molecule for the ligand.
        """
        return self.ligand_openff_molecule
    
    def set_ligand_openff_molecule(self, openff_molecule: OpenFFMolecule) -> None:
        """
            Set the OpenFF molecule for the ligand.

            Args:
                openff_molecule: OpenFF molecule to set for the ligand.
        """
        self.ligand_openff_molecule = openff_molecule

    def extract_ligand(self, keep_water: bool = False) -> app.Modeller:
        """
            Extract ligand-only structure from the loaded complex.
            
            Args:
                keep_water: Whether to keep water molecules with ligand.
                
            Returns:
                Ligand-only Modeller (copy with non-ligand atoms deleted).
                
            Raises:
                ComplexProcessingError: If no complex is loaded.
        """
        self._check_complex_loaded()
        ligand_modeller = self._copy_modeller(self.complex_modeller)
        
        atoms_to_delete = []
        for atom in ligand_modeller.topology.atoms():
            if atom.residue.chain.id == self.ligand_chain_id:
                continue
            elif keep_water and atom.residue.name in ['HOH', 'WAT', 'TIP3', 'TIP4', 'SPC']:
                continue
            else:
                atoms_to_delete.append(atom)
        
        ligand_modeller.delete(atoms_to_delete)
        return ligand_modeller
    
    def extract_target(self, keep_water: bool = True) -> app.Modeller:
        """
            Extract target-only structure from the loaded complex.
            
            Args:
                keep_water: Whether to keep water molecules with target (default: True).
                
            Returns:
                Target-only Modeller (copy with ligand deleted).
                
            Raises:
                ComplexProcessingError: If no complex is loaded.
        """
        self._check_complex_loaded()
        target_modeller = self._copy_modeller(self.complex_modeller)
        
        atoms_to_delete = []
        for atom in target_modeller.topology.atoms():
            if atom.residue.chain.id == self.ligand_chain_id:
                atoms_to_delete.append(atom)
            elif not keep_water and atom.residue.name in ['HOH', 'WAT', 'TIP3', 'TIP4', 'SPC']:
                atoms_to_delete.append(atom)
        
        target_modeller.delete(atoms_to_delete)
        return target_modeller
    
    def extract_pocket(
        self, 
        cutoff: float, 
        expand_residues: bool = True, 
        keep_water: bool = False,
        return_indices: bool = False
    ) -> Union[app.Modeller, List[int]]:
        """
            Extract binding pocket structure from the loaded complex.
            
            Args:
                cutoff: Distance cutoff in Angstroms for pocket definition.
                expand_residues: Whether to include complete residues in pocket.
                keep_water: Whether to keep water molecules in pocket (default: False).
                return_indices: Whether to return the indices of the pocket atoms.

            Returns:
                Pocket-only Modeller (copy with distant atoms deleted).
                If return_indices is True, returns list of atom indices instead.
                
            Raises:
                ComplexProcessingError: If no complex is loaded.
        """
        self._check_complex_loaded()
        
        pocket_modeller = self._copy_modeller(self.complex_modeller)
        if not keep_water:
            pocket_modeller.deleteWater()

        ligand_coords = self._get_ligand_coordinates(heavy_atoms_only=True)
        if not ligand_coords:
            raise ComplexProcessingError("No ligand atoms found for pocket extraction")
        
        non_ligand_atoms = []
        non_ligand_coords = []
        
        for atom in pocket_modeller.topology.atoms():
            if atom.residue.chain.id != self.ligand_chain_id:
                non_ligand_atoms.append(atom)
                pos = pocket_modeller.positions[atom.index].value_in_unit(unit.angstrom)
                non_ligand_coords.append([pos[0], pos[1], pos[2]])
        
        if not non_ligand_atoms:
            raise ComplexProcessingError("No non-ligand atoms found")
        
        ns = NeighborSearch(np.array(non_ligand_coords), logger=self.logger)
        pocket_indices = ns.search_multiple_within_cutoff(
            coords=np.array(ligand_coords),
            cutoff=cutoff
        )
        
        # Convert back to atom objects
        pocket_atoms = set(non_ligand_atoms[i] for i in pocket_indices)
        
        if expand_residues:
            pocket_residues = set(atom.residue for atom in pocket_atoms)
            expanded_pocket_atoms = set()
            for residue in pocket_residues:
                expanded_pocket_atoms.update(residue.atoms())
            pocket_atoms = expanded_pocket_atoms
        
        if return_indices:
            return sorted([atom.index for atom in pocket_atoms])
        else:
            atoms_to_delete = []
            for atom in pocket_modeller.topology.atoms():
                if atom not in pocket_atoms:
                    atoms_to_delete.append(atom)
            
            pocket_modeller.delete(atoms_to_delete)
            return pocket_modeller

    def write_complex(self, output_file: Path) -> None:
        """
            Write current complex to PDB file.
            
            Args:
                output_file: Output PDB file path.
                
            Raises:
                ComplexProcessingError: If no complex is loaded or write fails.
        """
        self._check_complex_loaded()
        self._write_structure(self.complex_modeller, output_file)
    
    def write_ligand(self, output_file: Path, keep_water: bool = False) -> None:
        """
            Extract ligand and write to PDB file.
            
            Args:
                output_file: Output PDB file path.
                keep_water: Whether to keep water molecules with ligand.
                
            Raises:
                ComplexProcessingError: If extraction or write fails.
        """
        ligand_modeller = self.extract_ligand(keep_water=keep_water)
        self._write_structure(ligand_modeller, output_file)
    
    def write_target(self, output_file: Path, keep_water: bool = False) -> None:
        """
            Extract target and write to PDB file.
            
            Args:
                output_file: Output PDB file path.
                keep_water: Whether to keep water molecules with target.
                
            Raises:
                ComplexProcessingError: If extraction or write fails.
        """
        target_modeller = self.extract_target(keep_water=keep_water)
        self._write_structure(target_modeller, output_file)
    
    def write_pocket(
        self, 
        output_file: Path, 
        cutoff: float, 
        expand_residues: bool = True, 
        keep_water: bool = False
    ) -> None:
        """
            Extract pocket and write to PDB file.
            
            Args:
                output_file: Output PDB file path.
                cutoff: Distance cutoff in Angstroms for pocket definition.
                expand_residues: Whether to include complete residues in pocket.
                keep_water: Whether to keep water molecules in pocket.
                
            Raises:
                ComplexProcessingError: If extraction or write fails.
        """
        pocket_modeller = self.extract_pocket(cutoff, expand_residues, keep_water)
        self._write_structure(pocket_modeller, output_file)

    def _check_complex_loaded(self) -> None:
        """Validate that complex is loaded, raise error if not."""
        if self.complex_modeller is None:
            raise ComplexProcessingError("No complex loaded. Call a load_from_* method first.")
    
    def _check_ligand_molecule_set(self) -> None:
        """Validate that ligand OpenFF molecule is set, raise error if not."""
        if self.ligand_openff_molecule is None:
            raise ComplexProcessingError("No ligand OpenFF molecule set. Call set_ligand_openff_molecule() first.")
    
    def _load_target(self, file_path: Path, fix: bool = False) -> app.Modeller:
        """
            Load the target file into OpenMM Modeller.

            Args:
                file_path: Path to the target file. PDB and PDBx are supported.
                fix: Whether to fix the structure by adding missing atoms and hydrogens.

            Returns:
                OpenMM Modeller object containing topology and positions.
        """
        try:
            target = PDBFixer(str(file_path))
            target.findMissingResidues()
            
            if target.missingResidues:
                self.logger.warning(
                    f"Missing residues found in target ((chain index, residue index): residues): " 
                    f"{target.missingResidues}"
                )

            for key, res_list in list(target.missingResidues.items()):
                if len(res_list) > 10:
                    self.logger.warning(
                        f"More than 10 missing residues in a row for chain {key[0]}, "
                        f"residue index {key[1]}. Skipping addition of these residues."
                    )
                    del target.missingResidues[key]

            if fix:
                target.findNonstandardResidues()
                target.replaceNonstandardResidues()
                target.findMissingAtoms()
                target.addMissingAtoms()

            modeller = app.Modeller(target.topology, target.positions)
            return modeller
        
        except Exception as e:
            raise ComplexProcessingError(f"Failed to load target from {file_path}: {e}") from e
    
    def _load_ligand(self, file_path: Path) -> OpenFFMolecule:
        """
            Load the ligand file into Openff Molecule.

            Args:
                file_path: Path to the ligand file. SDF, MOL2 and PDB are 
                    supported formats

            Returns:
                Ligand molecule as an OpenFFMolecule.

            Raises:
                ComplexProcessingError: If file cannot be loaded.
        """
        try:
            if file_path.suffix.lower() == '.sdf':
                ligand = Chem.MolFromMolFile(str(file_path), removeHs=False)
            elif file_path.suffix.lower() == '.mol2':
                ligand = Chem.MolFromMol2File(str(file_path), removeHs=False)
            elif file_path.suffix.lower() == '.pdb':
                ligand = Chem.MolFromPDBFile(str(file_path), removeHs=False)
            else:
                raise ComplexProcessingError(f"Unsupported ligand file format: {file_path.suffix}")
            
            return OpenFFMolecule.from_rdkit(ligand, allow_undefined_stereo=True)
        
        except Exception as e:
            raise ComplexProcessingError(f"Failed to load ligand from {file_path}: {e}") from e

    def _write_structure(
            self, 
            modeller: app.Modeller, 
            output_file: Path,
            format: Optional[str] = None
        ) -> None:
        """
            Write OpenMM Modeller to PDB file.
            
            Args:
                modeller: OpenMM Modeller object to write.
                output_file: Output file path.
                format: Output file format, "pdb", "pdbx", "cif", "sdf" and "mol2" 
                    are supported. If not given, it will be inferred from file extension.

            Raises:
                ComplexProcessingError: If file cannot be written.
        """
        format = format or output_file.suffix[1:].lower()
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'pdb':
                with open(output_file, 'w') as f:
                    app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
            elif format in ['pdbx', 'cif']:
                with open(output_file, 'w') as f:
                    app.PDBxFile.writeFile(modeller.topology, modeller.positions, f, entry=self.structure_id)
            elif format in ['sdf', 'mol2']:
                rdkit_mol = None
                
                # Try OpenFF conversion first if molecule is available
                if self.ligand_openff_molecule:
                    try:
                        openff_mol = OpenFFTopology.from_openmm(
                            modeller.getTopology(),
                            [self.ligand_openff_molecule],
                            modeller.getPositions()
                        )
                        openff_mol = OpenFFMolecule.from_topology(openff_mol)
                        rdkit_mol = openff_mol.to_rdkit()
                        self.logger.debug("Successfully converted using OpenFF molecule")
                    except Exception as e:
                        self.logger.warning(f"OpenFF conversion failed, falling back to PDB method: {e}")
                        rdkit_mol = None
                
                # Fallback to PDB conversion if OpenFF failed or not available
                if rdkit_mol is None:
                    try:
                        pdb_string = StringIO()
                        app.PDBFile.writeFile(modeller.topology, modeller.positions, pdb_string)
                        pdb_content = pdb_string.getvalue()
                        pdb_string.close()
                        
                        rdkit_mol = Chem.MolFromPDBBlock(
                            pdb_content, sanitize=False, removeHs=False, proximityBonding=False
                        )
                        
                        if rdkit_mol.GetNumBonds() == 0:
                            self.logger.warning(f"No bonding info in PDB, trying with proximity bonding")
                            rdkit_mol = Chem.MolFromPDBBlock(
                                pdb_content, sanitize=True, removeHs=True, proximityBonding=True
                            )
                        
                        if rdkit_mol:
                            rdb.DetermineBondOrders(rdkit_mol)
                            self.logger.debug("Successfully converted using PDB method with bond order determination")
                    except Exception as e:
                        raise ComplexProcessingError(f"Both OpenFF and PDB conversion methods failed: {e}")

                if not rdkit_mol:
                    raise ComplexProcessingError(f"Failed to convert structure to RDKit molecule: {self.structure_id}")
                
                # Validate molecule integrity - fail if multiple fragments
                num_fragments = len(Chem.GetMolFrags(rdkit_mol, sanitizeFrags=False))
                if num_fragments != 1:
                    raise ComplexProcessingError(f"Molecule has {num_fragments} fragments, cannot write as single molecule file")

                try:
                    if format == 'sdf':
                        Chem.MolToMolFile(rdkit_mol, str(output_file))
                    elif format == 'mol2':
                        Chem.MolToMol2File(rdkit_mol, str(output_file))
                except Exception as e:
                    raise ComplexProcessingError(f"Failed to write {format.upper()} file: {e}")
            else:
                raise ComplexProcessingError(f"Unsupported output file format: {format}")

        except Exception as e:
            raise ComplexProcessingError(f"Failed to write structure to {output_file}: {e}") from e
    
    def _combine_structures(
        self, 
        target: app.Modeller, 
        ligand: Union[app.Modeller, OpenFFMolecule]
    ) -> Tuple[app.Modeller, str]:
        """
            Combine target and ligand structures using OpenMM Modeller. Also, assign ligand chain ID.
            
            Args:
                target: Target structure as Modeller.
                ligand: Ligand structure as Modeller or OpenFF Molecule
                
            Returns:
                Combined complex as Modeller
                Assigned ligand chain id
        """
        # Create copy of target to avoid modifying original
        combined = self._copy_modeller(target)
        existing_chain_ids = {ch.id for ch in combined.topology.chains()}
        ligand_chain_id = [id for id in PDB_CHAIN_IDS if id not in existing_chain_ids][7]

        if isinstance(ligand, OpenFFMolecule):
            for atom in ligand.atoms:
                # NOTE: Not setting chain_id fails downstream processing.
                # Also, chain_id is updated when writing structures, so
                # these IDs might be different in the written files. 
                # There is no problem with residue_name.
                atom.metadata['chain_id'] = ligand_chain_id
                atom.metadata['residue_name'] = 'LIG'
            ligand_topology = ligand.to_topology()
            ligand_positions = ligand_topology.get_positions().to_openmm()
            ligand_topology = ligand_topology.to_openmm()
        else:
            if ligand.topology.getNumChains() != 1:
                raise ComplexProcessingError(
                    f"Ligand Modeller must contain exactly one chain, got {ligand.topology.getNumChains()}"
                )
            for chain in ligand.topology.chains():
                chain.id = ligand_chain_id
            ligand_topology = ligand.getTopology()
            ligand_positions = ligand.getPositions()

        combined.add(ligand_topology, ligand_positions)
        
        return combined, ligand_chain_id
    
    def _copy_modeller(self, modeller: app.Modeller) -> app.Modeller:
        """
            Create deep copy of OpenMM Modeller object.
            
            Args:
                modeller: Source Modeller to copy.
                
            Returns:
                Independent copy of the Modeller.
        """
        return app.Modeller(modeller.topology, modeller.positions)
    
    def _get_ligand_coordinates(self, heavy_atoms_only: bool = True) -> List[List[float]]:
        """
            Get ligand coordinates from the loaded complex for spatial searches.
            
            Args:
                heavy_atoms_only: Whether to exclude hydrogen atoms.
                
            Returns:
                List of [x, y, z] coordinates in Angstroms.
                
            Raises:
                ComplexProcessingError: If no complex is loaded.
        """
        self._check_complex_loaded()
        
        ligand_coords = []
        for atom in self.complex_modeller.topology.atoms():
            if atom.residue.chain.id == self.ligand_chain_id:
                if heavy_atoms_only and atom.element.symbol == 'H':
                    continue
                pos = self.complex_modeller.positions[atom.index].value_in_unit(unit.angstrom)
                ligand_coords.append([pos[0], pos[1], pos[2]])
        
        return ligand_coords

