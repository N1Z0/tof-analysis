"""Ion library, custom species, and styling metadata."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from tof_analysis.physics import IonSpecies, argon_ions, holmium_ions, methane_fragment_name, methane_fragments

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "custom_ions.json"

GROUP_COLORS = {
    "Argon": "#059669",
    "Methane": "#7c3aed",
    "Holmium": "#d97706",
    "Restgas": "#e11d48",
    "Misc": "#64748b",
    "Custom": "#9333ea",
}

GROUP_ORDER = ["Argon", "Methane", "Holmium", "Restgas", "Custom", "Misc"]


@dataclass(frozen=True)
class AnalysisSpecies:
    """Selectable species row in the analysis workbench (charge range expands to ion names)."""

    key: str
    label: str
    group: str
    color: str
    q_min: int = 1
    q_max: int = 8
    fixed_ion: str | None = None
    name_pattern: str | None = None
    charge_names: dict[int, str] | None = None

    def ion_names(self, q_min: int | None = None, q_max: int | None = None) -> list[str]:
        if self.fixed_ion:
            return [self.fixed_ion]
        lo = max(self.q_min, int(q_min if q_min is not None else self.q_min))
        hi = min(self.q_max, int(q_max if q_max is not None else self.q_max))
        if lo > hi:
            return []
        names: list[str] = []
        pattern = self.name_pattern or f"{self.key}{{q}}+"
        for q in range(lo, hi + 1):
            if self.charge_names and q in self.charge_names:
                names.append(self.charge_names[q])
            else:
                names.append(pattern.format(q=q))
        return names


def _methane_charge_names(formula: str, *, q_max: int = 8) -> dict[int, str]:
    return {q: methane_fragment_name(formula, q) for q in range(1, q_max + 1)}


def analysis_species_catalog(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> list[AnalysisSpecies]:
    """Species rows shown in the analysis ion checklist."""
    c = GROUP_COLORS
    species: list[AnalysisSpecies] = [
        AnalysisSpecies("Ar", "Ar", "Argon", c["Argon"], q_max=8, name_pattern="Ar{q}+"),
        AnalysisSpecies("H_m", "H", "Methane", c["Methane"], q_max=8, charge_names=_methane_charge_names("H")),
        AnalysisSpecies("C_m", "C", "Methane", c["Methane"], q_max=8, charge_names=_methane_charge_names("C")),
        AnalysisSpecies("CH", "CH", "Methane", c["Methane"], q_max=8, charge_names=_methane_charge_names("CH")),
        AnalysisSpecies("CH2", "CH2", "Methane", c["Methane"], q_max=8, charge_names=_methane_charge_names("CH2")),
        AnalysisSpecies("CH3", "CH3", "Methane", c["Methane"], q_max=8, charge_names=_methane_charge_names("CH3")),
        AnalysisSpecies("CH4", "CH4", "Methane", c["Methane"], q_max=8, charge_names=_methane_charge_names("CH4")),
        AnalysisSpecies("Ho", "Ho", "Holmium", c["Holmium"], q_max=8, name_pattern="Ho{q}+"),
        AnalysisSpecies("H", "H", "Restgas", c["Restgas"], q_min=1, q_max=1, fixed_ion="H+"),
        AnalysisSpecies("C", "C", "Restgas", c["Restgas"], q_max=8, name_pattern="C{q}+"),
        AnalysisSpecies("N", "N", "Restgas", c["Restgas"], q_max=8, name_pattern="N{q}+"),
        AnalysisSpecies("O", "O", "Restgas", c["Restgas"], q_max=8, name_pattern="O{q}+"),
        AnalysisSpecies("D", "D", "Misc", c["Misc"], q_min=1, q_max=1, fixed_ion="D+"),
        AnalysisSpecies("He", "He", "Misc", c["Misc"], q_min=1, q_max=2, charge_names={1: "He+", 2: "He2+"}),
        AnalysisSpecies("Ne", "Ne", "Misc", c["Misc"], q_min=1, q_max=1, fixed_ion="Ne+"),
        AnalysisSpecies("CO", "CO", "Misc", c["Misc"], q_min=1, q_max=1, fixed_ion="CO+"),
        AnalysisSpecies("CO2", "CO₂", "Misc", c["Misc"], q_min=1, q_max=1, fixed_ion="CO2+"),
        AnalysisSpecies("N2", "N₂", "Misc", c["Misc"], q_min=1, q_max=1, fixed_ion="N2+_mol"),
    ]
    for entry in load_custom_entries(config_path):
        species.append(
            AnalysisSpecies(
                key=f"custom_{entry.prefix}",
                label=entry.prefix,
                group=entry.group or "Custom",
                color=entry.color,
                q_min=entry.charge_min,
                q_max=entry.charge_max,
                name_pattern=f"{entry.prefix}{{q}}+",
            )
        )
    return species


def expand_species_selection(
    selections: list[tuple[AnalysisSpecies, int, int]],
    registry: dict[str, IonDefinition] | None = None,
) -> list[IonDefinition]:
    """Expand checked species + charge ranges into registry ion definitions."""
    registry = registry or full_ion_registry()
    seen: set[str] = set()
    ions: list[IonDefinition] = []
    for species, q_min, q_max in selections:
        for name in species.ion_names(q_min, q_max):
            if name in registry and name not in seen:
                seen.add(name)
                ions.append(registry[name])
    return ions


@dataclass
class IonDefinition:
    name: str
    mass_amu: float
    charge: int
    color: str
    group: str

    @property
    def species(self) -> IonSpecies:
        return IonSpecies(self.name, self.mass_amu, self.charge)


@dataclass
class CustomIonEntry:
    prefix: str
    mass_amu: float
    charge_min: int = 1
    charge_max: int = 1
    color: str = "#9333ea"
    group: str = "Custom"

    def expand(self) -> list[IonDefinition]:
        ions: list[IonDefinition] = []
        for q in range(self.charge_min, self.charge_max + 1):
            name = f"{self.prefix}{q}+"
            ions.append(
                IonDefinition(
                    name=name,
                    mass_amu=self.mass_amu,
                    charge=q,
                    color=self.color,
                    group=self.group,
                )
            )
        return ions

    @classmethod
    def from_dict(cls, data: dict) -> "CustomIonEntry":
        return cls(
            prefix=str(data["prefix"]),
            mass_amu=float(data["mass_amu"]),
            charge_min=int(data.get("charge_min", 1)),
            charge_max=int(data.get("charge_max", data.get("charge_min", 1))),
            color=str(data.get("color", GROUP_COLORS["Custom"])),
            group=str(data.get("group", "Custom")),
        )


def _ions_with_group(ions: list[IonSpecies], group: str, color: str | None = None) -> list[IonDefinition]:
    c = color or GROUP_COLORS.get(group, GROUP_COLORS["Misc"])
    return [IonDefinition(i.name, i.mass_amu, i.charge, c, group) for i in ions]


def restgas_ions(*, max_charge: int = 8) -> list[IonDefinition]:
    """H, C, N, O charge states typical in vacuum rest gas."""
    color = GROUP_COLORS["Restgas"]
    ions = [IonDefinition("H+", 1.007825, 1, color, "Restgas")]
    for q in range(1, max_charge + 1):
        ions.append(IonDefinition(f"C{q}+", 12.0, q, color, "Restgas"))
        ions.append(IonDefinition(f"N{q}+", 14.003074, q, color, "Restgas"))
        ions.append(IonDefinition(f"O{q}+", 15.999, q, color, "Restgas"))
    return ions


def builtin_registry() -> dict[str, IonDefinition]:
    registry: dict[str, IonDefinition] = {}
    for group, ions in [
        ("Argon", _ions_with_group(argon_ions(max_charge=8), "Argon")),
        ("Methane", _ions_with_group(methane_fragments(max_charge=8), "Methane")),
        ("Holmium", _ions_with_group(holmium_ions(max_charge=8), "Holmium")),
        ("Restgas", restgas_ions()),
    ]:
        for ion in ions:
            registry[ion.name] = ion
    from tof_analysis.physics import _other_ions

    for sp in _other_ions():
        if sp.name not in registry:
            registry[sp.name] = IonDefinition(
                sp.name, sp.mass_amu, sp.charge, GROUP_COLORS["Misc"], "Misc"
            )
    return registry


def load_custom_entries(path: Path | str = DEFAULT_CONFIG_PATH) -> list[CustomIonEntry]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [CustomIonEntry.from_dict(entry) for entry in data.get("entries", [])]


def save_custom_entries(entries: list[CustomIonEntry], path: Path | str = DEFAULT_CONFIG_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entries": [asdict(e) for e in entries]}
    path.write_text(json.dumps(payload, indent=2))
    return path


def full_ion_registry(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, IonDefinition]:
    registry = builtin_registry()
    for entry in load_custom_entries(config_path):
        for ion in entry.expand():
            registry[ion.name] = ion
    return registry


def group_ion_names(group: str, registry: dict[str, IonDefinition] | None = None) -> list[str]:
    registry = registry or full_ion_registry()
    return sorted(name for name, ion in registry.items() if ion.group == group)


def all_ion_names(registry: dict[str, IonDefinition] | None = None) -> list[str]:
    """All ions sorted by group then name."""
    registry = registry or full_ion_registry()

    def sort_key(name: str) -> tuple[int, str]:
        group = registry[name].group
        try:
            order = GROUP_ORDER.index(group)
        except ValueError:
            order = len(GROUP_ORDER)
        return (order, name)

    return sorted(registry.keys(), key=sort_key)


def ion_checkbox_label(name: str, ion: IonDefinition) -> str:
    """Display label for the ion checklist (no tag for miscellaneous ions)."""
    if ion.group in {"Misc"}:
        return name
    return f"{name}  ({ion.group})"


def additional_ion_names(registry: dict[str, IonDefinition] | None = None) -> list[str]:
    """Deprecated alias — returns all ions."""
    return all_ion_names(registry)


def ion_species_list(registry: dict[str, IonDefinition] | None = None) -> list[IonSpecies]:
    registry = registry or full_ion_registry()
    return [ion.species for ion in registry.values()]


def flat_ion_options(registry: dict[str, IonDefinition] | None = None) -> list[str]:
    return ["—"] + sorted((registry or full_ion_registry()).keys())


def sanitize_calibration_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", name.strip())
    return cleaned.strip("_") or "calibration"
