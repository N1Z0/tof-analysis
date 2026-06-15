"""TOF flight-time physics and ion species library."""

from __future__ import annotations

from dataclasses import dataclass

ELEMENTARY_CHARGE = 1.602_176_634e-19  # C
ATOMIC_MASS_UNIT = 1.660_539_066_60e-27  # kg


@dataclass(frozen=True)
class IonSpecies:
    name: str
    mass_amu: float
    charge: int = 1

    @property
    def mz(self) -> float:
        return self.mass_amu / self.charge

    @property
    def sqrt_mz(self) -> float:
        return self.mz**0.5


def flight_time(
    mass_amu: float,
    charge: int,
    length_m: float,
    voltage_V: float,
    t0_s: float = 0.0,
) -> float:
    """Return flight time in seconds for a drift length and acceleration voltage."""
    mass_kg = mass_amu * ATOMIC_MASS_UNIT
    charge_C = charge * ELEMENTARY_CHARGE
    velocity = (2.0 * charge_C * voltage_V / mass_kg) ** 0.5
    return length_m / velocity + t0_s


def flight_time_us(
    mass_amu: float,
    charge: int,
    length_m: float,
    voltage_V: float,
    t0_us: float = 0.0,
) -> float:
    return flight_time(mass_amu, charge, length_m, voltage_V, t0_s=t0_us * 1e-6) * 1e6


def sqrt_mz(mass_amu: float, charge: int = 1) -> float:
    return (mass_amu / charge) ** 0.5


def ion_library() -> list[IonSpecies]:
    """All built-in and custom ions from the project registry."""
    from tof_analysis.ion_config import ion_species_list

    return ion_species_list()


def argon_ions(*, max_charge: int = 8, mass_amu: float = 39.948) -> list[IonSpecies]:
    return [IonSpecies(f"Ar{q}+", mass_amu, q) for q in range(1, max_charge + 1)]


def methane_fragment_name(formula: str, charge: int) -> str:
    """Build a methane-group ion name; q=1 keeps legacy names (CH2+ is methyl, not CH²⁺)."""
    if charge == 1:
        return f"{formula}+"
    if formula in {"C", "CH", "CH2", "CH3", "CH4"}:
        return f"({formula}){charge}+"
    return f"{formula}{charge}+"


def methane_fragments(*, max_charge: int = 8) -> list[IonSpecies]:
    bases = [
        ("H", 1.007825),
        ("C", 12.0),
        ("CH", 13.018),
        ("CH2", 14.026),
        ("CH3", 15.034),
        ("CH4", 16.042),
    ]
    ions: list[IonSpecies] = []
    for formula, mass in bases:
        for charge in range(1, max_charge + 1):
            ions.append(IonSpecies(methane_fragment_name(formula, charge), mass, charge))
    return ions


def holmium_ions(*, max_charge: int = 8, mass_amu: float = 164.930) -> list[IonSpecies]:
    """Holmium ions expected in laser-ablation shots."""
    return [IonSpecies(f"Ho{q}+", mass_amu, q) for q in range(1, max_charge + 1)]


def _other_ions() -> list[IonSpecies]:
    entries = [
        ("D+", 2.014102, 1),
        ("He+", 4.002602, 1),
        ("He2+", 4.002602, 2),
        ("C2+", 12.0, 2),
        ("C3+", 12.0, 3),
        ("C4+", 12.0, 4),
        ("N+", 14.003074, 1),
        ("O+", 15.999, 1),
        ("O2+", 15.999, 2),
        ("O3+", 15.999, 3),
        ("O4+", 15.999, 4),
        ("O5+", 15.999, 5),
        ("O6+", 15.999, 6),
        ("O7+", 15.999, 7),
        ("O8+", 15.999, 8),
        ("Ne+", 20.1797, 1),
        ("CO+", 28.0101, 1),
        ("CO2+", 44.0095, 1),
        ("N2+_mol", 28.0134, 1),
    ]
    return [IonSpecies(name=n, mass_amu=m, charge=z) for n, m, z in entries]
