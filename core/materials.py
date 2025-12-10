from dataclasses import dataclass

@dataclass
class Material:
    name: str
    density_g_cm3: float
    zeff_hint: float | None = None

MATERIALS = {
    "WATER": Material("WATER", 1.0, 7.42),
    "SOFT_TISSUE": Material("SOFT_TISSUE", 1.06, 7.6),
    "BONE_CORTICAL": Material("BONE_CORTICAL", 1.85, 13.8),
    "BONE_TRABECULAR": Material("BONE_TRABECULAR", 1.20, 12.0),
    "ENAMEL": Material("ENAMEL", 2.90, 16.0),
    "DENTIN": Material("DENTIN", 2.10, 14.0),
    "PMMA": Material("PMMA", 1.19, 6.5),
}

def get_material(name: str) -> Material:
    return MATERIALS[name]
