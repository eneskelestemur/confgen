"""High-level pipeline: input -> curation -> generation -> minimization -> output."""
from __future__ import annotations

import logging
from pathlib import Path

from joblib import Parallel, delayed
from rdkit import Chem
from tqdm import tqdm

from confgen.config import ConfGenConfig
from confgen.constraints import build_coord_map, load_reference_mol
from confgen.curation import curate_molecule, enumerate_stereoisomers
from confgen.forcefield import ForceFieldProvider
from confgen.generator import ConformerGenerator
from confgen.minimizer import Minimizer
from confgen.mol_io import (
    SDFWriterContext,
    assign_mol_ids,
    get_software_versions,
    read_molecules,
    write_input_molecules_smi,
    write_run_params,
)

_logger = logging.getLogger(__name__)


class ConfGenPipeline:
    """Orchestrates the full conformer generation workflow."""

    def __init__(self, config: ConfGenConfig):
        config.validate()
        self.config = config
        self._setup_logging()

    def run(self) -> dict:
        """Execute the full pipeline and return summary statistics."""
        cfg = self.config
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Write run parameters ----
        versions = get_software_versions()
        write_run_params(cfg.to_dict(), versions, out_dir / "run_params.json")

        # ---- Read & curate ----
        raw = read_molecules(cfg.input)
        molecules = assign_mol_ids(raw)
        _logger.info(f"Read {len(raw)} entries, {len(molecules)} valid molecules")

        allowed = frozenset(cfg.allowed_elements) if cfg.allowed_elements else None
        curated: list[tuple[Chem.Mol, str]] = []
        for mol, mol_id in molecules:
            cm = curate_molecule(mol, cfg.max_heavy_atoms, allowed)
            if cm is not None:
                curated.append((cm, mol_id))
        _logger.info(f"{len(curated)} molecules passed curation")

        # ---- Stereo enumeration ----
        stereo_mols: list[tuple[Chem.Mol, str, str | None]] = []
        if cfg.enumerate_stereo:
            for mol, mol_id in curated:
                isomers = enumerate_stereoisomers(mol, cfg.max_stereo_isomers)
                for si, iso_mol in enumerate(isomers):
                    if len(isomers) > 1:
                        new_id = f"{mol_id}_s{si:02d}"
                        stereo_mols.append((iso_mol, new_id, mol_id))
                    else:
                        stereo_mols.append((iso_mol, mol_id, None))
        else:
            stereo_mols = [(m, mid, None) for m, mid in curated]

        # ---- Write input_molecules.smi ----
        smi_entries = [(m, mid) for m, mid, _ in stereo_mols]
        write_input_molecules_smi(smi_entries, out_dir / "input_molecules.smi")

        # ---- Setup components ----
        coord_map = self._build_constraints(stereo_mols)

        ff_provider = ForceFieldProvider(cfg.forcefield)
        generator = ConformerGenerator(
            n_confs=cfg.n_confs,
            rmsd_threshold=cfg.rmsd_threshold,
            seed=cfg.seed,
            num_threads=cfg.num_threads,
            coord_map=coord_map,
        )
        minimizer = Minimizer(
            ff_provider=ff_provider,
            max_iters=cfg.max_minimize_iters,
            num_threads=cfg.num_threads,
            platform=cfg.platform,
            seed=cfg.seed,
            solvent=cfg.solvent,
        )

        # ---- Generate, minimize & write ----
        sdf_path = out_dir / "conformers.sdf"
        stats = {
            "total_input": len(raw),
            "valid_parsed": len(molecules),
            "passed_curation": len(curated),
            "after_stereo": len(stereo_mols),
            "successful_molecules": 0,
            "failed_generation": 0,
            "total_conformers": 0,
        }

        with SDFWriterContext(sdf_path) as sdf_writer:
            if cfg.num_workers > 1:
                stats = self._run_parallel(
                    stereo_mols, generator, minimizer, ff_provider,
                    stats, sdf_writer,
                )
            else:
                stats = self._run_sequential(
                    stereo_mols, generator, minimizer, ff_provider,
                    stats, sdf_writer,
                )
            stats["total_conformers"] = sdf_writer.count

        _logger.info(
            f"Done: {stats['successful_molecules']} molecules, "
            f"{stats['total_conformers']} conformers written to {out_dir}"
        )
        return stats

    # ---- helpers ----

    def _flush_results(
        self,
        results: list[dict],
        sdf_writer: SDFWriterContext,
    ) -> None:
        """Apply energy-window filter (if configured) and write to SDF."""
        if not results:
            return
        if self.config.energy_window is not None:
            results = self._filter_by_energy_window(
                results, self.config.energy_window
            )
        sdf_writer.write_results(results)

    def _run_sequential(
        self,
        stereo_mols: list[tuple[Chem.Mol, str, str | None]],
        generator: ConformerGenerator,
        minimizer: Minimizer,
        ff_provider: ForceFieldProvider,
        stats: dict,
        sdf_writer: SDFWriterContext,
    ) -> dict:
        for mol, mol_id, parent_id in tqdm(
            stereo_mols, desc="Generating conformers", disable=_logger.level > logging.INFO
        ):
            results = _process_one_molecule(
                mol, mol_id, parent_id, generator, minimizer, ff_provider
            )
            if results:
                self._flush_results(results, sdf_writer)
                stats["successful_molecules"] += 1
            else:
                stats["failed_generation"] += 1
        return stats

    def _run_parallel(
        self,
        stereo_mols: list[tuple[Chem.Mol, str, str | None]],
        generator: ConformerGenerator,
        minimizer: Minimizer,
        ff_provider: ForceFieldProvider,
        stats: dict,
        sdf_writer: SDFWriterContext,
    ) -> dict:
        results_list = Parallel(n_jobs=self.config.num_workers, backend="loky")(
            delayed(_process_one_molecule)(
                mol, mol_id, parent_id, generator, minimizer, ff_provider
            )
            for mol, mol_id, parent_id in stereo_mols
        )

        for results in results_list:
            if results:
                self._flush_results(results, sdf_writer)
                stats["successful_molecules"] += 1
            else:
                stats["failed_generation"] += 1
        return stats

    def _build_constraints(
        self, stereo_mols: list[tuple[Chem.Mol, str, str | None]]
    ) -> dict[int, tuple[float, float, float]] | None:
        cfg = self.config
        if not cfg.constraint_smarts or not cfg.constraint_coords:
            return None

        ref_mol = load_reference_mol(cfg.constraint_coords)
        if ref_mol is None:
            _logger.error("Failed to load constraint reference; running unconstrained")
            return None

        # Use the first molecule to build the map (all should share the substructure)
        first_mol = stereo_mols[0][0] if stereo_mols else None
        if first_mol is None:
            return None

        return build_coord_map(Chem.AddHs(first_mol), cfg.constraint_smarts, ref_mol)

    @staticmethod
    def _filter_by_energy_window(
        results: list[dict], window_kcal: float
    ) -> list[dict]:
        """Keep only conformers within `window_kcal` of the per-molecule minimum."""
        from collections import defaultdict

        by_mol: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            by_mol[r["mol_id"]].append(r)

        filtered: list[dict] = []
        for mol_id, entries in by_mol.items():
            energies = [e["energy"] for e in entries if e.get("energy") is not None]
            if not energies:
                filtered.extend(entries)
                continue
            min_e = min(energies)
            for e in entries:
                if e.get("energy") is None or (e["energy"] - min_e) <= window_kcal:
                    filtered.append(e)
        return filtered

    def _setup_logging(self) -> None:
        level = getattr(logging, self.config.log_level, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        # Also log to file
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "run.log", mode="w")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logging.getLogger().addHandler(fh)


def _process_one_molecule(
    mol: Chem.Mol,
    mol_id: str,
    parent_id: str | None,
    generator: ConformerGenerator,
    minimizer: Minimizer,
    ff_provider: ForceFieldProvider,
) -> list[dict] | None:
    """Process a single molecule: generate conformers, minimize, return result dicts.

    This is a module-level function so joblib can pickle it.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    mol_h, conf_ids = generator.generate(mol)
    if mol_h is None or not conf_ids:
        return None

    energies = minimizer.minimize(mol_h, conf_ids)

    results: list[dict] = []
    for i, (cid, energy) in enumerate(energies):
        results.append(
            {
                "mol": mol_h,
                "mol_id": mol_id,
                "conf_tag": f"conf_{i:03d}",
                "conf_id": cid,
                "smiles": smiles,
                "energy": energy,
                "energy_unit": "kcal/mol",
                "forcefield": ff_provider.name,
                "original_name": mol_id,
                "stereo_parent_id": parent_id,
            }
        )
    return results
