"""Tests for config module."""
import pytest
import yaml

from confgen.config import ConfGenConfig


def test_defaults():
    cfg = ConfGenConfig()
    assert cfg.forcefield == "mmff"
    assert cfg.seed == 42
    assert cfg.solvent is None
    assert cfg.n_confs == 200


def test_validation_good():
    cfg = ConfGenConfig(forcefield="uff", n_confs=10)
    cfg.validate()  # should not raise


def test_validation_bad_forcefield():
    cfg = ConfGenConfig(forcefield="unknown")
    with pytest.raises(ValueError, match="Unknown forcefield"):
        cfg.validate()


def test_validation_bad_solvent():
    cfg = ConfGenConfig(solvent="water")
    with pytest.raises(ValueError, match="Unknown solvent"):
        cfg.validate()


def test_validation_solvent_requires_openmm():
    cfg = ConfGenConfig(forcefield="mmff", solvent="implicit-obc2")
    with pytest.raises(ValueError, match="OpenMM forcefield"):
        cfg.validate()

    cfg2 = ConfGenConfig(forcefield="gfn2-xtb", solvent="explicit-tip3p")
    with pytest.raises(ValueError, match="OpenMM forcefield"):
        cfg2.validate()

    # Should NOT raise for an OpenMM forcefield
    cfg3 = ConfGenConfig(forcefield="gaff", solvent="implicit-obc2")
    cfg3.validate()


def test_validation_bad_n_confs():
    cfg = ConfGenConfig(n_confs=0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_from_yaml(tmp_path):
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(yaml.dump({"forcefield": "uff", "n_confs": 50, "seed": 123}))
    cfg = ConfGenConfig.from_yaml(yaml_path)
    assert cfg.forcefield == "uff"
    assert cfg.n_confs == 50
    assert cfg.seed == 123


def test_from_yaml_ignores_unknown(tmp_path):
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(yaml.dump({"forcefield": "mmff", "unknown_key": "value"}))
    cfg = ConfGenConfig.from_yaml(yaml_path)
    assert cfg.forcefield == "mmff"
    assert not hasattr(cfg, "unknown_key")


def test_merge_cli_overrides():
    cfg = ConfGenConfig(forcefield="mmff", n_confs=200)
    cfg.merge_cli_overrides({"forcefield": "uff", "n_confs": 50, "solvent": None})
    assert cfg.forcefield == "uff"
    assert cfg.n_confs == 50
    assert cfg.solvent is None


def test_to_dict():
    cfg = ConfGenConfig(forcefield="uff")
    d = cfg.to_dict()
    assert d["forcefield"] == "uff"
    assert "n_confs" in d


def test_from_defaults():
    cfg = ConfGenConfig.from_defaults()
    assert cfg.forcefield == "mmff"


def test_gpu_streams_default():
    cfg = ConfGenConfig()
    assert cfg.gpu_streams == 1


def test_save_trajectories_default():
    cfg = ConfGenConfig()
    assert cfg.save_trajectories is False


def test_gen_backend_default():
    cfg = ConfGenConfig()
    assert cfg.gen_backend == "rdkit"


def test_gen_backend_nvmolkit_validates():
    cfg = ConfGenConfig(gen_backend="nvmolkit")
    cfg.validate()  # should not raise


def test_gen_backend_invalid():
    cfg = ConfGenConfig(gen_backend="unknown_backend")
    with pytest.raises(ValueError, match="gen_backend"):
        cfg.validate()


def test_md_params_defaults():
    cfg = ConfGenConfig()
    assert cfg.md_timestep_fs == 0.5
    assert cfg.md_temperature_k == 300.0
    assert cfg.md_pressure_atm == 1.0
    assert cfg.md_nvt_time_ps == 50.0
    assert cfg.md_prod_time_ns == 0.1


def test_md_params_from_yaml(tmp_path):
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(yaml.dump({
        "md_timestep_fs": 2.0,
        "md_nvt_time_ps": 25.0,
        "md_prod_time_ns": 0.05,
        "md_temperature_k": 310.0,
    }))
    cfg = ConfGenConfig.from_yaml(yaml_path)
    assert cfg.md_timestep_fs == 2.0
    assert cfg.md_nvt_time_ps == 25.0
    assert cfg.md_prod_time_ns == 0.05
    assert cfg.md_temperature_k == 310.0
