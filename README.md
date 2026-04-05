# confgen

Physics-based small-molecule conformer generation using RDKit, OpenMM, OpenFF-Toolkit, and tblite.

## Features

- **ETKDG conformer embedding** with RMSD-based deduplication (RDKit)
- **Eight force fields** across three backends for energy minimization
- **Implicit and explicit solvation** via OpenMM Generalized Born and water-box models
- **Stereoisomer enumeration** for molecules with unspecified stereocenters
- **Substructure constraints** to fix atom positions during embedding
- **Parallel processing** at the molecule level (joblib) and per-worker threading
- **YAML config** with full CLI override support

## Installation

Create the conda environment and install:

```bash
mamba env create -f environment.yaml
mamba activate confgen
pip install -e .
```

## Quick start

### Basic run (MMFF, vacuum)

```bash
confgen run -i molecules.smi -o output/
```

### Specify a force field

```bash
# OpenFF SMIRNOFF (OpenMM backend)
confgen run -i molecules.smi --forcefield smirnoff

# GFN2-xTB tight-binding DFT (tblite backend)
confgen run -i molecules.smi --forcefield gfn2-xtb
```

### Solvation (requires OpenMM forcefield)

```bash
# Implicit Generalized Born
confgen run -i molecules.smi --forcefield gaff --solvent implicit-obc2

# Explicit TIP3P water box
confgen run -i molecules.smi --forcefield smirnoff --solvent explicit-tip3p
```

### Enumerate stereoisomers

```bash
confgen run -i molecules.smi --enumerate-stereo --max-stereo-isomers 8
```

### Energy window filter

```bash
confgen run -i molecules.smi --energy-window 10.0
```

### Use a YAML config file

```bash
confgen run -i molecules.smi --config my_config.yaml
```

CLI flags always override values from the config file. Print the default config with:

```bash
confgen show-config
```

## Force fields

| Force field | Backend | Description |
|-------------|---------|-------------|
| `mmff` *(default)* | RDKit | Merck Molecular Force Field 94 |
| `uff` | RDKit | Universal Force Field |
| `gaff` | OpenMM | General Amber Force Field 2.11 |
| `smirnoff` | OpenMM | Open Force Field (openff-2.2.1) |
| `espaloma` | OpenMM | Espaloma ML potential (0.3.2) |
| `gfn2-xtb` | tblite | GFN2-xTB tight-binding DFT |
| `gfn1-xtb` | tblite | GFN1-xTB tight-binding DFT |
| `ipea1-xtb` | tblite | IPEA1-xTB tight-binding DFT |

RDKit and tblite force fields run in vacuum only. Solvation requires an OpenMM force field (`gaff`, `smirnoff`, or `espaloma`).

## Solvent models

| Model | Type | Description |
|-------|------|-------------|
| `implicit-obc1` | Implicit | OBC1 (HCT radii) |
| `implicit-obc2` | Implicit | OBC2 (GBSA-OBC) |
| `implicit-gbn` | Implicit | GBn |
| `implicit-gbn2` | Implicit | GBn2 |
| `implicit-hct` | Implicit | HCT |
| `explicit-tip3p` | Explicit | TIP3P |
| `explicit-tip3pfb` | Explicit | TIP3P-FB |
| `explicit-tip4pew` | Explicit | TIP4P-Ew |
| `explicit-tip4pfb` | Explicit | TIP4P-FB |
| `explicit-spce` | Explicit | SPC/E |
| `explicit-opc` | Explicit | OPC |
| `explicit-opc3` | Explicit | OPC3 |

Explicit models add a periodic water box with 0.5 nm padding around the solute.

## Python API

```python
from confgen import ConfGenConfig, ConfGenPipeline

cfg = ConfGenConfig(
    input="molecules.smi",
    output_dir="results",
    n_confs=100,
    forcefield="smirnoff",
    solvent="implicit-obc2",
)
cfg.validate()

pipeline = ConfGenPipeline(cfg)
stats = pipeline.run()
print(stats)
```

## CLI reference

```
confgen run --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --input` | *(required)* | Input file (.smi, .sdf) or directory |
| `-o / --output-dir` | `confgen_output` | Output directory |
| `--config` | — | YAML config file |
| `--n-confs` | `200` | Number of ETKDG conformer attempts |
| `--rmsd-threshold` | `1.5` | RMSD threshold (Å) for deduplication |
| `--energy-window` | off | Energy window (kcal/mol) above minimum |
| `--forcefield` | `mmff` | Force field (see table above) |
| `--max-minimize-iters` | `500` | Max minimization iterations |
| `--solvent` | vacuum | Solvent model (see table above) |
| `--enumerate-stereo` | off | Enumerate unspecified stereocenters |
| `--max-stereo-isomers` | `32` | Max stereoisomers per molecule |
| `--constraint-smarts` | — | SMARTS for constrained atoms |
| `--constraint-coords` | — | Reference structure for constraints |
| `--max-heavy-atoms` | `100` | Skip molecules above this size |
| `--num-workers` | `1` | Molecule-level parallelism (joblib) |
| `--num-threads` | `1` | Per-worker threads (RDKit/OpenMM) |
| `--platform` | `CPU` | OpenMM platform (CPU, CUDA, OpenCL, HIP) |
| `--seed` | `42` | Random seed (-1 = non-deterministic) |
| `--log-level` | `INFO` | Logging verbosity |

## License

MIT
