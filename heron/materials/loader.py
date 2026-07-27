"""Load and query the material table (CLAUDE.md §7).

Turns ``table.yaml`` + ``aliases.yaml`` into a queryable object. Physics engines
ask it for per-material scalars (emissivity, temperature prior); Layer A asks it
to resolve free-text VLM labels to seed classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_DIR = Path(__file__).parent
DEFAULT_TABLE = _DIR / "table.yaml"
DEFAULT_ALIASES = _DIR / "aliases.yaml"


@dataclass(frozen=True)
class Material:
    """One material's cross-instrument behavior. See table.yaml for meaning."""

    name: str
    emissivity_lwir: float
    lwir_behavior: str
    temp_mean_c: float
    temp_sigma_c: float
    zones: dict[str, float]
    xray_class: str
    nir: str
    kirlian_activity: float
    transmissivity_lwir: float

    @classmethod
    def from_entry(cls, name: str, e: dict) -> "Material":
        tp = e.get("temp_prior_c", {})
        return cls(
            name=name,
            emissivity_lwir=float(e.get("emissivity_lwir", 0.92)),
            lwir_behavior=str(e.get("lwir_behavior", "emissive")),
            temp_mean_c=float(tp.get("mean", 24.0)),
            temp_sigma_c=float(tp.get("sigma", 4.0)),
            zones={k: float(v) for k, v in e.get("zones", {}).items()},
            xray_class=str(e.get("xray_class", "soft")),
            nir=str(e.get("nir", "neutral")),
            kirlian_activity=float(e.get("kirlian_activity", 0.2)),
            transmissivity_lwir=float(e.get("transmissivity_lwir", 0.0)),
        )


@dataclass(frozen=True)
class MaterialTable:
    materials: dict[str, Material]
    aliases: dict[str, str]

    DEFAULT = "unknown"

    def get(self, name: str) -> Material:
        return self.materials.get(name, self.materials[self.DEFAULT])

    def names(self) -> list[str]:
        return list(self.materials.keys())

    def resolve_label(self, label: str) -> str:
        """Map a free-text label (VLM/grounding) to a seed-class material name.

        Exact material name wins; then exact alias; then substring alias match;
        else the default class. Case-insensitive.
        """
        key = label.strip().lower()
        if key in self.materials:
            return key
        if key in self.aliases:
            return self.aliases[key]
        for alias, target in self.aliases.items():
            if alias in key:
                return target
        return self.DEFAULT

    # --- scalar projections for physics ------------------------------------
    def emissivity_map(self) -> dict[str, float]:
        return {n: m.emissivity_lwir for n, m in self.materials.items()}

    def temp_mean_map(self) -> dict[str, float]:
        return {n: m.temp_mean_c for n, m in self.materials.items()}

    def reflective_map(self) -> dict[str, float]:
        """1.0 for reflective materials, 0.5 for mixed, 0.0 for emissive."""
        weight = {"reflective": 1.0, "mixed": 0.5, "emissive": 0.0}
        return {n: weight.get(m.lwir_behavior, 0.0) for n, m in self.materials.items()}

    # Relative X-ray linear-attenuation coefficient per mass class (stylized,
    # not calibrated) — higher = more opaque to X-rays (CLAUDE.md §3.2, §7).
    XRAY_MU = {
        "air": 0.03,
        "fabric": 0.15,
        "soft": 0.55,
        "dense": 0.9,
        "bone": 1.4,
        "metal": 2.6,
    }

    def xray_mu_map(self) -> dict[str, float]:
        """material name -> relative X-ray attenuation coefficient."""
        return {n: self.XRAY_MU.get(m.xray_class, 0.55) for n, m in self.materials.items()}

    def nir_map(self) -> dict[str, str]:
        """material name -> NIR behavior string (foliage_glow|skin_translucent|
        dye_transparent|dark|neutral)."""
        return {n: m.nir for n, m in self.materials.items()}

    def kirlian_map(self) -> dict[str, float]:
        """material name -> corona-seeding propensity [0,1]."""
        return {n: m.kirlian_activity for n, m in self.materials.items()}

    def transmissivity_map(self) -> dict[str, float]:
        """material name -> LWIR transmissivity: how much of the warm body
        BEHIND a surface shows through it (fabric/hair are semi-transparent)."""
        return {n: m.transmissivity_lwir for n, m in self.materials.items()}


@lru_cache(maxsize=4)
def load_material_table(
    table_path: str | Path = DEFAULT_TABLE,
    aliases_path: str | Path = DEFAULT_ALIASES,
) -> MaterialTable:
    table_data = yaml.safe_load(Path(table_path).read_text())
    materials = {
        name: Material.from_entry(name, entry)
        for name, entry in table_data["materials"].items()
    }
    if MaterialTable.DEFAULT not in materials:
        raise ValueError(f"material table must define a '{MaterialTable.DEFAULT}' class")
    aliases_data = yaml.safe_load(Path(aliases_path).read_text())
    aliases = {k.lower(): v for k, v in aliases_data.get("aliases", {}).items()}
    return MaterialTable(materials=materials, aliases=aliases)
