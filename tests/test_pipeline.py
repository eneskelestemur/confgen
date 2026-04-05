"""Integration tests for the pipeline (RDKit-only backend)."""
import json
from pathlib import Path

from confgen.config import ConfGenConfig
from confgen.pipeline import ConfGenPipeline


def test_pipeline_smi_to_sdf(tmp_path):
    """Full pipeline: SMILES file -> SDF output with MMFF."""
    smi_file = tmp_path / "input.smi"
    smi_file.write_text("CCO ethanol\nc1ccccc1 benzene\n")
    out_dir = tmp_path / "output"

    cfg = ConfGenConfig(
        input=str(smi_file),
        output_dir=str(out_dir),
        n_confs=10,
        rmsd_threshold=0.5,
        forcefield="mmff",
        max_minimize_iters=100,
        seed=42,
        log_level="WARNING",
    )

    pipeline = ConfGenPipeline(cfg)
    stats = pipeline.run()

    assert stats["successful_molecules"] == 2
    assert stats["total_conformers"] >= 2
    assert (out_dir / "conformers.sdf").exists()
    assert (out_dir / "input_molecules.smi").exists()
    assert (out_dir / "run_params.json").exists()

    params = json.loads((out_dir / "run_params.json").read_text())
    assert params["parameters"]["forcefield"] == "mmff"
    assert "rdkit" in params["software_versions"]


def test_pipeline_uff(tmp_path):
    smi_file = tmp_path / "input.smi"
    smi_file.write_text("CCO ethanol\n")
    out_dir = tmp_path / "output"

    cfg = ConfGenConfig(
        input=str(smi_file),
        output_dir=str(out_dir),
        n_confs=5,
        forcefield="uff",
        seed=42,
        log_level="WARNING",
    )

    pipeline = ConfGenPipeline(cfg)
    stats = pipeline.run()
    assert stats["successful_molecules"] == 1


def test_pipeline_stereo_enumeration(tmp_path):
    smi_file = tmp_path / "input.smi"
    smi_file.write_text("CC(O)C(N)C test_stereo\n")
    out_dir = tmp_path / "output"

    cfg = ConfGenConfig(
        input=str(smi_file),
        output_dir=str(out_dir),
        n_confs=5,
        rmsd_threshold=0.3,
        forcefield="mmff",
        enumerate_stereo=True,
        max_stereo_isomers=8,
        seed=42,
        log_level="WARNING",
    )

    pipeline = ConfGenPipeline(cfg)
    stats = pipeline.run()
    assert stats["after_stereo"] >= 2


def test_pipeline_energy_window(tmp_path):
    smi_file = tmp_path / "input.smi"
    smi_file.write_text("c1ccccc1 benzene\n")
    out_dir = tmp_path / "output"

    cfg = ConfGenConfig(
        input=str(smi_file),
        output_dir=str(out_dir),
        n_confs=20,
        rmsd_threshold=0.3,
        energy_window=5.0,
        forcefield="mmff",
        seed=42,
        log_level="WARNING",
    )

    pipeline = ConfGenPipeline(cfg)
    stats = pipeline.run()
    assert stats["total_conformers"] >= 1


def test_pipeline_curation_filters(tmp_path):
    """Molecules with disallowed elements should be filtered."""
    smi_file = tmp_path / "input.smi"
    smi_file.write_text("CCO ethanol\n[Fe] iron\n")
    out_dir = tmp_path / "output"

    cfg = ConfGenConfig(
        input=str(smi_file),
        output_dir=str(out_dir),
        n_confs=5,
        forcefield="mmff",
        seed=42,
        log_level="WARNING",
    )

    pipeline = ConfGenPipeline(cfg)
    stats = pipeline.run()
    assert stats["passed_curation"] == 1
