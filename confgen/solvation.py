"""Solvation setup for OpenMM-based minimization (implicit and explicit)."""
from __future__ import annotations

import logging

from openmm import app, unit

from confgen._constants import EXPLICIT_SOLVENT_MODELS, IMPLICIT_SOLVENT_MODELS

_logger = logging.getLogger(__name__)


def get_solvent_xmls(solvent: str | None) -> list[str]:
    """Return the OpenMM XML file paths needed for the requested solvent model.

    For implicit solvent, the XML is applied as a force to the system.
    For explicit solvent, the water-model XML is loaded into the ForceField.

    Returns an empty list for vacuum (solvent=None).
    """
    if solvent is None:
        return []

    solvent = solvent.lower()

    if solvent in IMPLICIT_SOLVENT_MODELS:
        return [IMPLICIT_SOLVENT_MODELS[solvent]]

    if solvent in EXPLICIT_SOLVENT_MODELS:
        return [EXPLICIT_SOLVENT_MODELS[solvent]]

    raise ValueError(
        f"Unknown solvent model '{solvent}'. "
        f"Choose from: {sorted(list(IMPLICIT_SOLVENT_MODELS) + list(EXPLICIT_SOLVENT_MODELS))}"
    )


def is_explicit(solvent: str | None) -> bool:
    """Return True if the solvent model requires explicit water molecules."""
    if solvent is None:
        return False
    return solvent.lower() in EXPLICIT_SOLVENT_MODELS


def add_explicit_solvent(
    modeller: app.Modeller,
    forcefield: app.ForceField,
    solvent: str,
    padding_nm: float = 0.5,
    ionic_strength_molar: float = 0.0,
) -> None:
    """Add explicit water and optional ions to an OpenMM Modeller in-place.

    Args:
        modeller: OpenMM Modeller to solvate.
        forcefield: ForceField with the water model already loaded.
        solvent: Explicit solvent key (e.g. 'explicit-tip3p').
        padding_nm: Water box padding in nanometers.
        ionic_strength_molar: Ionic strength for neutralization (NaCl).
    """
    water_model_map = {
        "explicit-tip3p": "tip3p",
        "explicit-tip3pfb": "tip3pfb",
        "explicit-tip4pew": "tip4pew",
        "explicit-tip4pfb": "tip4pfb",
        "explicit-spce": "spce",
        "explicit-opc": "opc",
        "explicit-opc3": "opc3",
    }
    model = water_model_map.get(solvent.lower())
    if model is None:
        raise ValueError(f"Unknown explicit solvent model: {solvent}")

    modeller.addSolvent(
        forcefield,
        model=model,
        padding=padding_nm * unit.nanometer,
        ionicStrength=ionic_strength_molar * unit.molar,
    )
    _logger.info(
        f"Added explicit solvent ({model}) with {padding_nm} nm padding"
    )
