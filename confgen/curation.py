"""Molecule curation: validation, fragment handling, stereochemistry enumeration."""
from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

from confgen._constants import ORGANIC_ATOMS

_logger = logging.getLogger(__name__)


def curate_molecule(
    mol: Chem.Mol,
    max_heavy_atoms: int = 100,
    allowed_elements: frozenset[str] | None = None,
) -> Chem.Mol | None:
    """Validate and clean a molecule: keep largest fragment, check size and elements.

    Returns None if the molecule fails validation.
    """
    if mol is None:
        return None

    allowed = allowed_elements or ORGANIC_ATOMS

    # Keep the largest fragment (by heavy atom count)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

    if mol.GetNumHeavyAtoms() > max_heavy_atoms:
        _logger.debug(
            f"Skipping: {mol.GetNumHeavyAtoms()} heavy atoms > limit {max_heavy_atoms}"
        )
        return None

    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed:
            _logger.debug(f"Skipping: disallowed element {atom.GetSymbol()}")
            return None

    return mol


def enumerate_stereoisomers(
    mol: Chem.Mol,
    max_isomers: int = 32,
) -> list[Chem.Mol]:
    """Enumerate unspecified stereocenters, returning distinct stereoisomers.

    If the molecule has no unspecified stereo, a single-element list is returned.
    """
    opts = StereoEnumerationOptions(
        maxIsomers=max_isomers,
        tryEmbedding=False,
        onlyUnassigned=True,
        unique=True,
    )
    isomers = list(EnumerateStereoisomers(mol, options=opts))
    if not isomers:
        return [mol]
    return isomers
