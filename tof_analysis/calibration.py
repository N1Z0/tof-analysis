"""Calibrate TOF traces against ion (q, m) assignments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.optimize import linear_sum_assignment

from tof_analysis.detection import Dip
from tof_analysis.physics import ATOMIC_MASS_UNIT, ELEMENTARY_CHARGE, IonSpecies, sqrt_mz


@dataclass
class Calibration:
    length_m: float
    voltage_V: float
    t0_us: float
    k_us_per_sqrt_mz: float
    rmse_us: float
    n_points: int

    def predict_time_us(self, ion: IonSpecies) -> float:
        return self.k_us_per_sqrt_mz * ion.sqrt_mz + self.t0_us

    def predict_time_us_from_mz(self, mass_amu: float, charge: int = 1) -> float:
        return self.k_us_per_sqrt_mz * sqrt_mz(mass_amu, charge) + self.t0_us

    def effective_length_m(self) -> float:
        """Recover drift length from k = L * sqrt(u/(2e)) / sqrt(V)."""
        factor = (ATOMIC_MASS_UNIT / (2.0 * ELEMENTARY_CHARGE)) ** 0.5
        return self.k_us_per_sqrt_mz * 1e-6 * (self.voltage_V**0.5) / factor


@dataclass(frozen=True)
class Assignment:
    dip: Dip
    ion: IonSpecies
    residual_us: float


def fit_calibration(
    assignments: list[tuple[Dip, IonSpecies]],
    *,
    voltage_V: float,
    length_m: float | None = None,
) -> Calibration:
    """
    Fit t = k * sqrt(m/z) + t0 from manual dip-ion pairings.

    If length_m is supplied, k is constrained by the drift length and only t0 is free.
    Otherwise both k and t0 are fit (useful when voltage or length are uncertain).
    """
    if len(assignments) < 2:
        raise ValueError("Need at least two dip-ion assignments for a calibration fit")

    x = np.array([ion.sqrt_mz for _, ion in assignments], dtype=float)
    y = np.array([dip.time_us for dip, _ in assignments], dtype=float)

    if length_m is not None:
        factor = (length_m / (voltage_V**0.5)) * (ATOMIC_MASS_UNIT / (2.0 * ELEMENTARY_CHARGE)) ** 0.5
        k = factor * 1e6
        t0 = float(np.mean(y - k * x))
    else:
        design = np.column_stack([x, np.ones_like(x)])
        coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        k, t0 = float(coeffs[0]), float(coeffs[1])
        length_m = (k * 1e-6) * (voltage_V**0.5) / (ATOMIC_MASS_UNIT / (2.0 * ELEMENTARY_CHARGE)) ** 0.5

    predicted = k * x + t0
    rmse = float(np.sqrt(np.mean((predicted - y) ** 2)))
    return Calibration(
        length_m=float(length_m),
        voltage_V=float(voltage_V),
        t0_us=t0,
        k_us_per_sqrt_mz=k,
        rmse_us=rmse,
        n_points=len(assignments),
    )


def suggest_ion_assignments(
    dips: list[Dip],
    candidate_ions: list[IonSpecies],
    *,
    voltage_V: float,
    length_m: float,
    t0_us: float = 0.0,
    max_dips: int = 12,
    max_ions: int = 12,
) -> list[Assignment]:
    """
    Match detected dips to candidate ions by minimizing |t_dip - t_theory|.

    Uses the Hungarian algorithm when counts differ; brute-force permutations for small sets.
    """
    if not dips or not candidate_ions:
        return []

    dips = dips[:max_dips]
    ions = candidate_ions[:max_ions]
    k = (
        (length_m / (voltage_V**0.5))
        * (ATOMIC_MASS_UNIT / (2.0 * ELEMENTARY_CHARGE)) ** 0.5
        * 1e6
    )

    dip_times = np.array([d.time_us for d in dips])
    ion_times = np.array([k * ion.sqrt_mz + t0_us for ion in ions])
    cost = np.abs(dip_times[:, None] - ion_times[None, :])

    n_dips, n_ions = cost.shape
    use_brute_force = n_dips <= 6 and n_ions <= 10 and n_dips <= n_ions
    if use_brute_force:
        best_perm = None
        best_score = np.inf
        for ion_subset in permutations(range(n_ions), n_dips):
            score = sum(cost[i, j] for i, j in enumerate(ion_subset))
            if score < best_score:
                best_score = score
                best_perm = ion_subset
        assert best_perm is not None
        return [
            Assignment(
                dip=dips[i],
                ion=ions[best_perm[i]],
                residual_us=float(dip_times[i] - ion_times[best_perm[i]]),
            )
            for i in range(n_dips)
        ]

    n = max(n_dips, n_ions)
    padded = np.full((n, n), cost.max() * 10.0)
    padded[:n_dips, :n_ions] = cost
    row_ind, col_ind = linear_sum_assignment(padded)
    assignments: list[Assignment] = []
    for row, col in zip(row_ind, col_ind):
        if row >= n_dips or col >= n_ions:
            continue
        assignments.append(
            Assignment(
                dip=dips[row],
                ion=ions[col],
                residual_us=float(dip_times[row] - ion_times[col]),
            )
        )
    assignments.sort(key=lambda a: a.dip.time_us)
    return assignments
