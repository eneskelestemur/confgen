"""Configuration management: dataclass, YAML loading, CLI merge."""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Any

import yaml

from confgen._constants import ALL_FORCEFIELDS, ALL_SOLVENT_MODELS, GEN_BACKENDS, OPENMM_FORCEFIELDS

_logger = logging.getLogger(__name__)

_DEFAULTS_DIR = Path(__file__).resolve().parent.parent / "defaults"


@dataclass
class ConfGenConfig:
    """Full configuration for a confgen run."""

    # I/O
    input: str = ""
    output_dir: str = "confgen_output"

    # Generation
    gen_backend: str = "rdkit"        # rdkit | nvmolkit
    n_confs: int = 200
    rmsd_threshold: float = 1.5
    energy_window: float | None = None  # kcal/mol above minimum

    # Force field
    forcefield: str = "mmff"
    max_minimize_iters: int = 500

    # Solvation (None = vacuum)
    solvent: str | None = None

    # MD relaxation (OpenMM only)
    run_md: bool = False
    md_timestep_fs: float = 0.5       # femtoseconds
    md_temperature_k: float = 300.0   # Kelvin
    md_pressure_atm: float = 1.0      # atm (NPT, explicit solvent only)
    md_nvt_time_ps: float = 50.0      # NVT equilibration + NPT equilibration duration
    md_prod_time_ns: float = 0.1      # production MD duration

    # Stereochemistry
    enumerate_stereo: bool = False
    max_stereo_isomers: int = 32

    # Constraints
    constraint_smarts: str | None = None
    constraint_coords: str | None = None

    # Curation
    max_heavy_atoms: int = 100
    allowed_elements: list[str] | None = None

    # Computation
    gpu_streams: int = 1          # concurrent conformer simulations via MPS (OpenMM only)
    save_trajectories: bool = False  # write DCD + PDB per conformer to mdsims/ (run_md only)
    num_threads: int = 1
    platform: str = "CPU"
    seed: int = 42

    # Logging
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        self.gen_backend = self.gen_backend.lower()
        self.forcefield = self.forcefield.lower()
        self.platform = self.platform.upper()
        self.log_level = self.log_level.upper()
        if self.solvent:
            self.solvent = self.solvent.lower()

    def validate(self) -> None:
        """Raise ValueError for invalid settings."""
        if self.gen_backend not in GEN_BACKENDS:
            raise ValueError(
                f"Unknown gen_backend '{self.gen_backend}'. "
                f"Choose from: {sorted(GEN_BACKENDS)}"
            )
        if self.forcefield not in ALL_FORCEFIELDS:
            raise ValueError(
                f"Unknown forcefield '{self.forcefield}'. "
                f"Choose from: {sorted(ALL_FORCEFIELDS)}"
            )
        if self.solvent is not None and self.solvent not in ALL_SOLVENT_MODELS:
            raise ValueError(
                f"Unknown solvent '{self.solvent}'. "
                f"Choose from: {sorted(ALL_SOLVENT_MODELS)}"
            )
        if self.solvent is not None and self.forcefield not in OPENMM_FORCEFIELDS:
            raise ValueError(
                f"Solvation requires an OpenMM forcefield "
                f"({', '.join(sorted(OPENMM_FORCEFIELDS))}), "
                f"got '{self.forcefield}'"
            )
        if self.run_md and self.forcefield not in OPENMM_FORCEFIELDS:
            raise ValueError(
                f"run_md requires an OpenMM forcefield "
                f"({', '.join(sorted(OPENMM_FORCEFIELDS))}), "
                f"got '{self.forcefield}'"
            )
        if self.n_confs < 1:
            raise ValueError("n_confs must be >= 1")
        if self.rmsd_threshold <= 0:
            raise ValueError("rmsd_threshold must be > 0")
        if self.max_heavy_atoms < 1:
            raise ValueError("max_heavy_atoms must be >= 1")
        if self.gpu_streams < 1:
            raise ValueError("gpu_streams must be >= 1")
        if self.num_threads < 1:
            raise ValueError("num_threads must be >= 1")
        if self.platform not in ("CPU", "CUDA", "OPENCL", "HIP"):
            raise ValueError(f"Unknown platform '{self.platform}'")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (JSON-friendly)."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ConfGenConfig:
        """Load config from a YAML file, ignoring unknown keys."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid_keys = {fld.name for fld in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        unknown = set(data) - valid_keys
        if unknown:
            _logger.warning(f"Ignoring unknown config keys: {unknown}")
        return cls(**filtered)

    @classmethod
    def from_defaults(cls) -> ConfGenConfig:
        """Load from shipped default_config.yaml if it exists, else dataclass defaults."""
        default_path = _DEFAULTS_DIR / "default_config.yaml"
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls()

    def merge_cli_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply CLI overrides (non-None values only) on top of current config."""
        for key, value in overrides.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        self.__post_init__()
