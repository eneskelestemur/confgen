"""Molecule I/O: reading from various sources and writing conformer output."""
from __future__ import annotations

import json
import logging
import datetime
from pathlib import Path
from typing import Iterator

from rdkit import Chem

from confgen._typing import MolWithID

_logger = logging.getLogger(__name__)


class SDFWriterContext:
    """Streaming SDF writer that can be used as a context manager.

    Conformers are written to disk immediately via ``write_results``,
    avoiding the need to accumulate all results in memory.
    """

    def __init__(self, path: Path):
        self._path = path
        self._writer: Chem.SDWriter | None = None
        self.count = 0

    def __enter__(self) -> SDFWriterContext:
        self._writer = Chem.SDWriter(str(self._path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def write_results(self, results: list[dict]) -> None:
        """Write a batch of conformer result dicts to the SDF."""
        if self._writer is None:
            raise RuntimeError("SDFWriterContext is not open")
        for entry in results:
            mol: Chem.Mol = entry["mol"]
            conf_id: int = entry["conf_id"]
            name = f"{entry['mol_id']}__{entry['conf_tag']}"
            mol.SetProp("_Name", name)
            mol.SetProp("mol_id", entry["mol_id"])
            mol.SetProp("conf_tag", entry["conf_tag"])
            mol.SetProp("smiles", entry["smiles"])
            if entry.get("energy") is not None:
                mol.SetProp("energy", f"{entry['energy']:.6f}")
                mol.SetProp("energy_unit", entry.get("energy_unit", "kcal/mol"))
            mol.SetProp("forcefield", entry.get("forcefield", ""))
            if entry.get("original_name"):
                mol.SetProp("original_name", entry["original_name"])
            if entry.get("stereo_parent_id"):
                mol.SetProp("stereo_parent_id", entry["stereo_parent_id"])
            self._writer.write(mol, confId=conf_id)
            self.count += 1


def read_molecules(input_path: str) -> list[MolWithID]:
    """Read molecules from file or directory, returning (mol, id) pairs."""
    path = Path(input_path)
    if path.is_dir():
        return list(_read_directory(path))
    elif path.suffix.lower() in (".smi", ".smiles"):
        return list(_read_smiles_file(path))
    elif path.suffix.lower() == ".sdf":
        return list(_read_sdf_file(path))
    elif path.suffix.lower() == ".pdb":
        return list(_read_pdb_file(path))
    elif path.suffix.lower() == ".mol2":
        return list(_read_mol2_file(path))
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")


def _read_smiles_file(path: Path) -> Iterator[MolWithID]:
    """Read headerless whitespace-separated SMILES file (col1=SMILES, col2=optional ID)."""
    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            name = parts[1] if len(parts) > 1 else None
            try:
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                mol = None
            if mol is None:
                _logger.warning(f"Failed to parse SMILES on line {line_num}: {smiles}")
            yield mol, name


def _read_sdf_file(path: Path) -> Iterator[MolWithID]:
    """Read molecules from an SDF file."""
    suppl = Chem.SDMolSupplier(str(path), removeHs=False)
    for idx, mol in enumerate(suppl):
        name = None
        if mol is not None and mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            name = mol.GetProp("_Name").strip()
        if mol is None:
            _logger.warning(f"Failed to read molecule at index {idx} from {path.name}")
        yield mol, name


def _read_pdb_file(path: Path) -> Iterator[MolWithID]:
    """Read a single molecule from a PDB file."""
    mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
    name = path.stem
    if mol is None:
        _logger.warning(f"Failed to read PDB file: {path.name}")
    yield mol, name


def _read_mol2_file(path: Path) -> Iterator[MolWithID]:
    """Read a single molecule from a MOL2 file."""
    mol = Chem.MolFromMol2File(str(path), removeHs=False)
    name = path.stem
    if mol is None:
        _logger.warning(f"Failed to read MOL2 file: {path.name}")
    yield mol, name


def _read_directory(path: Path) -> Iterator[MolWithID]:
    """Read all supported structure files from a directory."""
    for fpath in sorted(path.iterdir()):
        if fpath.suffix.lower() == ".sdf":
            yield from _read_sdf_file(fpath)
        elif fpath.suffix.lower() == ".pdb":
            yield from _read_pdb_file(fpath)
        elif fpath.suffix.lower() == ".mol2":
            yield from _read_mol2_file(fpath)


def assign_mol_ids(
    mols: list[MolWithID],
) -> list[tuple[Chem.Mol, str]]:
    """Filter out failed molecules and assign sequential IDs where missing.

    Returns only valid (mol, id) tuples.
    """
    result = []
    idx = 0
    for mol, name in mols:
        if mol is None:
            continue
        mol_id = name if name else f"mol_{idx:06d}"
        result.append((mol, mol_id))
        idx += 1
    return result


def write_input_molecules_smi(
    molecules: list[tuple[Chem.Mol, str]],
    output_path: Path,
) -> None:
    """Write accepted molecules as a SMILES file with IDs."""
    with open(output_path, "w") as f:
        for mol, mol_id in molecules:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            f.write(f"{smi} {mol_id}\n")


def write_run_params(
    config_dict: dict,
    versions: dict[str, str],
    output_path: Path,
) -> None:
    """Write run parameters and software versions to JSON."""
    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "software_versions": versions,
        "parameters": config_dict,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def get_software_versions() -> dict[str, str]:
    """Collect versions of key dependencies."""
    import confgen

    versions: dict[str, str] = {"confgen": confgen.__version__}
    try:
        from rdkit import rdBase
        versions["rdkit"] = rdBase.rdkitVersion
    except Exception:
        pass
    try:
        import openmm
        versions["openmm"] = openmm.__version__
    except Exception:
        pass
    try:
        import openff.toolkit
        versions["openff-toolkit"] = openff.toolkit.__version__
    except Exception:
        pass
    try:
        import openmmforcefields
        versions["openmmforcefields"] = openmmforcefields.__version__
    except Exception:
        pass
    try:
        import tblite.interface
        versions["tblite"] = ".".join(str(v) for v in tblite.interface.library.get_version())
    except Exception:
        pass
    return versions
