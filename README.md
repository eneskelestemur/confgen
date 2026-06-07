# confgen

Physics-based small-molecule conformer generation using RDKit, OpenMM, OpenFF-Toolkit, and tblite.

## Features

- **ETKDG conformer embedding** with RMSD-based deduplication (RDKit or NVIDIA nvMolKit)
- **GPU-accelerated conformer generation** via nvMolKit (requires NVIDIA GPU, compute capability ≥7.0)
- **Eight force fields** across three backends for energy minimization
- **Implicit and explicit solvation** via OpenMM Generalized Born and water-box models
- **Multi-phase MD relaxation** (minimize → NVT → NPT → production → minimize, OpenMM only)
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

### GPU-accelerated conformer generation and minimization (nvMolKit)

```bash
confgen run -i molecules.smi --gen-backend nvmolkit --platform CUDA
```

When `--gen-backend nvmolkit` is set and the force field is `mmff` or `uff`, nvMolKit
also handles GPU-accelerated energy minimization — no extra flag needed.

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

### MD relaxation (OpenMM only)

```bash
confgen run -i molecules.smi --forcefield smirnoff --run-md
```

Advanced MD parameters (timestep, duration, temperature, pressure) can be set in a
YAML config file — see [MD Protocol Parameters](#md-protocol-parameters-config-only) below.

### GPU-accelerated MD with NVIDIA MPS

For small molecules the GPU is underutilised during a single OpenMM simulation.
NVIDIA MPS lets multiple conformer simulations share the GPU simultaneously.
Start MPS before the run and use `--gpu-streams` to control concurrency:

```bash
# Start MPS server (once per session, requires root or cuda group membership)
nvidia-cuda-mps-control -d

confgen run -i molecules.smi --forcefield smirnoff --run-md \
    --platform CUDA --gpu-streams 8

# Stop MPS after the run
echo quit | nvidia-cuda-mps-control
```

`--gpu-streams N` launches N conformer simulations concurrently on the GPU.
A value of 8–16 is a reasonable starting point for drug-like molecules on a
modern GPU; tune based on GPU memory and utilisation.

### Trajectory saving (debugging)

```bash
confgen run -i molecules.smi --forcefield smirnoff --run-md --save-trajectories
```

Creates `{output_dir}/mdsims/{mol_id}/` containing a topology PDB and DCD
trajectory for every conformer that underwent MD.  Off by default.

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

## Conformer generation backends

| Backend | Flag | Description |
|---------|------|-------------|
| `rdkit` *(default)* | `--gen-backend rdkit` | CPU-based ETKDG via RDKit |
| `nvmolkit` | `--gen-backend nvmolkit` | GPU-accelerated ETKDG via NVIDIA nvMolKit |

Both backends use the same ETKDGv3 algorithm and RMSD-based deduplication.
nvMolKit requires an NVIDIA GPU with compute capability ≥7.0 and CUDA driver ≥12.6.
Constrained embedding (`--constraint-smarts`) is not supported with the nvmolkit backend.
When using `nvmolkit` with `mmff` or `uff`, energy minimization is also GPU-accelerated.

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

## MD Protocol

When `--run-md` is enabled, each conformer undergoes the following protocol before
the final energy is recorded:

| Phase | Ensemble | Duration | Condition |
|-------|----------|----------|-----------|
| Initial minimize | — | until convergence | always |
| NVT equilibration | NVT, 300 K | 50 ps | `run_md` enabled |
| NPT equilibration | NPT, 300 K, 1 atm | 50 ps | `run_md` + explicit solvent |
| Production MD | NVT or NPT | 0.1 ns | `run_md` enabled |
| Final minimize | — | until convergence | `run_md` enabled |

NPT phases require a periodic box and are therefore skipped for vacuum and implicit solvent runs.

### MD Protocol Parameters (config only)

These parameters are available in the YAML config file only (not as CLI flags) to keep
the command line uncluttered. All are active only when `run_md: true`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `md_timestep_fs` | `0.5` | Integration timestep (femtoseconds) |
| `md_temperature_k` | `300.0` | Simulation temperature (Kelvin) |
| `md_pressure_atm` | `1.0` | Pressure for NPT barostat (atm) |
| `md_nvt_time_ps` | `50.0` | Duration of NVT and NPT equilibration (ps) |
| `md_prod_time_ns` | `0.1` | Production MD duration (ns) |

Example config snippet:

```yaml
run_md: true
md_timestep_fs: 2.0
md_temperature_k: 298.0
md_nvt_time_ps: 100.0
md_prod_time_ns: 0.5
```

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
| `--gen-backend` | `rdkit` | Conformer generation backend (`rdkit` or `nvmolkit`) |
| `--n-confs` | `200` | Number of ETKDG conformer attempts |
| `--rmsd-threshold` | `1.5` | RMSD threshold (Å) for deduplication |
| `--energy-window` | off | Energy window (kcal/mol) above minimum |
| `--forcefield` | `mmff` | Force field (see table above) |
| `--max-minimize-iters` | `500` | Max minimization iterations |
| `--solvent` | vacuum | Solvent model (see table above) |
| `--run-md` | off | Enable multi-phase MD relaxation (OpenMM only) |
| `--enumerate-stereo` | off | Enumerate unspecified stereocenters |
| `--max-stereo-isomers` | `32` | Max stereoisomers per molecule |
| `--constraint-smarts` | — | SMARTS for constrained atoms |
| `--constraint-coords` | — | Reference structure for constraints |
| `--max-heavy-atoms` | `100` | Skip molecules above this size |
| `--gpu-streams` | `1` | Concurrent conformer simulations via MPS (OpenMM + run_md) |
| `--save-trajectories` | off | Save DCD trajectory + topology PDB per conformer to `mdsims/` (run_md only) |
| `--num-threads` | `1` | Per-worker threads (RDKit/OpenMM) |
| `--platform` | `CPU` | OpenMM platform (CPU, CUDA, OpenCL, HIP) |
| `--seed` | `42` | Random seed (-1 = non-deterministic) |
| `--log-level` | `INFO` | Logging verbosity |

## License

MIT
