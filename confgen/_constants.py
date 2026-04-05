"""Constants used across confgen modules."""
from __future__ import annotations

ORGANIC_ATOMS: frozenset[str] = frozenset(
    {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
)

RDKIT_FORCEFIELDS: frozenset[str] = frozenset({"mmff", "uff"})

OPENMM_FORCEFIELDS: frozenset[str] = frozenset({"gaff", "smirnoff", "espaloma"})

TBLITE_METHODS: frozenset[str] = frozenset({"gfn2-xtb", "gfn1-xtb", "ipea1-xtb"})

ALL_FORCEFIELDS: frozenset[str] = RDKIT_FORCEFIELDS | OPENMM_FORCEFIELDS | TBLITE_METHODS

# Default SMIRNOFF force field (openff-2.2.1 is current default in openmmforcefields 0.15.1)
DEFAULT_SMIRNOFF_FF = "openff-2.2.1.offxml"
DEFAULT_GAFF_VERSION = "gaff-2.11"
DEFAULT_ESPALOMA_MODEL = "espaloma-0.3.2"

# Implicit solvent models available in OpenMM
IMPLICIT_SOLVENT_MODELS: dict[str, str] = {
    "implicit-obc1": "implicit/obc1.xml",
    "implicit-obc2": "implicit/obc2.xml",
    "implicit-gbn":  "implicit/gbn.xml",
    "implicit-gbn2": "implicit/gbn2.xml",
    "implicit-hct":  "implicit/hct.xml",
}

# Explicit water model XML files for OpenMM
EXPLICIT_SOLVENT_MODELS: dict[str, str] = {
    "explicit-tip3p":    "amber/tip3p_standard.xml",
    "explicit-tip3pfb":  "amber/tip3pfb_standard.xml",
    "explicit-tip4pew":  "amber/tip4pew_standard.xml",
    "explicit-tip4pfb":  "amber/tip4pfb_standard.xml",
    "explicit-spce":     "amber/spce_standard.xml",
    "explicit-opc":      "amber/opc_standard.xml",
    "explicit-opc3":     "amber/opc3_standard.xml",
}

ALL_SOLVENT_MODELS: frozenset[str] = frozenset(
    list(IMPLICIT_SOLVENT_MODELS) + list(EXPLICIT_SOLVENT_MODELS)
)

# tblite method name -> Calculator method string
TBLITE_METHOD_MAP: dict[str, str] = {
    "gfn2-xtb": "GFN2-xTB",
    "gfn1-xtb": "GFN1-xTB",
    "ipea1-xtb": "IPEA1-xTB",
}

# Energy conversion
HARTREE_TO_KCALMOL = 627.509474
KJ_TO_KCAL = 0.239006
