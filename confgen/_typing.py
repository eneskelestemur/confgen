"""Shared type aliases for confgen."""
from __future__ import annotations

from typing import TypeAlias

from rdkit.Chem.rdchem import Mol

RDKitMol: TypeAlias = Mol
MolWithID: TypeAlias = tuple[RDKitMol | None, str | None]
