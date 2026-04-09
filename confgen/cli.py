"""CLI entry point using Click with YAML config + CLI override support."""
from __future__ import annotations

from pathlib import Path

import click
import yaml

from confgen._constants import ALL_FORCEFIELDS, ALL_SOLVENT_MODELS
from confgen.config import ConfGenConfig

# ---- help text blocks ----

_FF_HELP = """\b
Force field for minimization.
RDKit (vacuum only):
  mmff          MMFF94 (default)
  uff           Universal Force Field
OpenMM (supports solvation):
  gaff          General Amber FF 2.11
  smirnoff      OpenFF (openff-2.2.1)
  espaloma      Espaloma ML (0.3.2)
tblite (vacuum only):
  gfn2-xtb      GFN2-xTB
  gfn1-xtb      GFN1-xTB
  ipea1-xtb     IPEA1-xTB"""

_SOLVENT_HELP = """\b
Solvent model (needs OpenMM FF).
Implicit (Generalized Born):
  implicit-obc1     implicit-obc2
  implicit-gbn      implicit-gbn2
  implicit-hct
Explicit (water box, 0.5 nm pad):
  explicit-tip3p    explicit-spce
  explicit-tip3pfb  explicit-opc
  explicit-tip4pew  explicit-opc3
  explicit-tip4pfb"""


@click.group()
@click.version_option(package_name="confgen")
def main() -> None:
    """confgen — physics-based small-molecule conformer generation."""


@main.command()
# ---- I/O ----
@click.option(
    "-i", "--input", "input_path", required=True,
    help="Input file (.smi, .sdf) or directory.",
)
@click.option(
    "-o", "--output-dir", default=None,
    help="Output directory.  [default: confgen_output]",
)
@click.option(
    "--config", "config_path", default=None,
    type=click.Path(exists=True),
    help="YAML config file (CLI flags override file values).",
)
# ---- Conformer generation ----
@click.option(
    "--n-confs", type=int, default=None,
    help="Number of ETKDG conformer attempts.  [default: 200]",
)
@click.option(
    "--rmsd-threshold", type=float, default=None,
    help="RMSD threshold (Å) for deduplication.  [default: 1.5]",
)
@click.option(
    "--energy-window", type=float, default=None,
    help="Keep conformers within this window (kcal/mol) of the minimum.  [default: off]",
)
# ---- Force field & solvation ----
@click.option(
    "--forcefield",
    type=click.Choice(sorted(ALL_FORCEFIELDS), case_sensitive=False),
    default=None,
    help=_FF_HELP,
)
@click.option(
    "--max-minimize-iters", type=int, default=None,
    help="Max minimization iterations.  [default: 500]",
)
@click.option(
    "--solvent",
    type=click.Choice(sorted(ALL_SOLVENT_MODELS), case_sensitive=False),
    default=None,
    help=_SOLVENT_HELP,
)
@click.option(
    "--run-md/--no-run-md", default=None,
    help="Run 0.1 ns MD before minimization (OpenMM only).  [default: off]",
)
# ---- Stereochemistry ----
@click.option(
    "--enumerate-stereo/--no-enumerate-stereo", default=None,
    help="Enumerate unspecified stereocenters.  [default: off]",
)
@click.option(
    "--max-stereo-isomers", type=int, default=None,
    help="Max stereoisomers per molecule.  [default: 32]",
)
# ---- Constraints ----
@click.option(
    "--constraint-smarts", default=None,
    help="SMARTS pattern for atoms fixed during embedding.",
)
@click.option(
    "--constraint-coords", default=None,
    type=click.Path(exists=True),
    help="Reference structure (.sdf/.mol2) with coordinates for constrained atoms.",
)
# ---- Curation ----
@click.option(
    "--max-heavy-atoms", type=int, default=None,
    help="Filter molecules with more heavy atoms.  [default: 100]",
)
# ---- Computation ----
@click.option(
    "--num-workers", type=int, default=None,
    help="Molecule-level parallelism via joblib.  [default: 1]",
)
@click.option(
    "--num-threads", type=int, default=None,
    help="Per-worker threads (RDKit batch opt / OpenMM).  [default: 1]",
)
@click.option(
    "--platform",
    type=click.Choice(["CPU", "CUDA", "OpenCL", "HIP"], case_sensitive=False),
    default=None,
    help="OpenMM platform.  [default: CPU]",
)
@click.option(
    "--seed", type=int, default=None,
    help="Random seed (-1 = non-deterministic).  [default: 42]",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Logging verbosity.  [default: INFO]",
)
def run(**kwargs: object) -> None:
    """Run conformer generation pipeline."""
    config_path = kwargs.pop("config_path")
    input_path = kwargs.pop("input_path")

    if config_path:
        cfg = ConfGenConfig.from_yaml(config_path)
    else:
        cfg = ConfGenConfig.from_defaults()

    # Always set input from CLI
    cfg.input = input_path

    # Merge CLI overrides (non-None values)
    overrides = {k: v for k, v in kwargs.items() if v is not None}
    cfg.merge_cli_overrides(overrides)
    cfg.validate()

    from confgen.pipeline import ConfGenPipeline

    pipeline = ConfGenPipeline(cfg)
    stats = pipeline.run()
    _print_stats(stats)


@main.command("show-config")
def show_config() -> None:
    """Print the default configuration as YAML."""
    cfg = ConfGenConfig.from_defaults()
    click.echo(yaml.dump(cfg.to_dict(), default_flow_style=False, sort_keys=False))


def _print_stats(stats: dict) -> None:
    click.echo("\n--- confgen summary ---")
    for key, val in stats.items():
        label = key.replace("_", " ").capitalize()
        click.echo(f"  {label}: {val}")


if __name__ == "__main__":
    main()
