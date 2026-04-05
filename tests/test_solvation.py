"""Tests for solvation module (unit tests only — no OpenMM system creation)."""
import pytest

from confgen.solvation import get_solvent_xmls, is_explicit


def test_vacuum():
    assert get_solvent_xmls(None) == []
    assert is_explicit(None) is False


def test_implicit_solvent():
    xmls = get_solvent_xmls("implicit-obc2")
    assert len(xmls) == 1
    assert "obc2" in xmls[0]
    assert is_explicit("implicit-obc2") is False


def test_explicit_solvent():
    xmls = get_solvent_xmls("explicit-tip3p")
    assert len(xmls) == 1
    assert "tip3p" in xmls[0]
    assert is_explicit("explicit-tip3p") is True


def test_unknown_solvent():
    with pytest.raises(ValueError, match="Unknown solvent"):
        get_solvent_xmls("invalid-model")


def test_all_implicit_models():
    from confgen._constants import IMPLICIT_SOLVENT_MODELS
    for key in IMPLICIT_SOLVENT_MODELS:
        xmls = get_solvent_xmls(key)
        assert len(xmls) == 1
        assert is_explicit(key) is False


def test_all_explicit_models():
    from confgen._constants import EXPLICIT_SOLVENT_MODELS
    for key in EXPLICIT_SOLVENT_MODELS:
        xmls = get_solvent_xmls(key)
        assert len(xmls) == 1
        assert is_explicit(key) is True
