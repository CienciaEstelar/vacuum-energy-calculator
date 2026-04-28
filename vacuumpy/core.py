"""
vacuumpy.core
=============

Pedagogical toolkit for UV-regularized vacuum energy density and ideal
Casimir force calculations in flat spacetime QFT.

Scope and limitations
---------------------
This module implements the standard textbook calculation of the zero-point
energy density of a free scalar field, regularized by a smooth UV cutoff
filter f(omega/omega_c):

    rho_vac(omega_c) = (hbar / 2*pi^2 * c^3) * int_0^inf omega^3 f(-omega/omega_c) d(omega)

The cutoff frequency omega_c is treated as a **free model parameter**, not a
fundamental physical scale. We do NOT claim that any particular value of
omega_c resolves the cosmological constant problem; reproducing the observed
vacuum energy density by tuning omega_c is a dimensional consistency check,
not a physical prediction.

The module also provides the ideal (T=0, perfectly smooth, perfectly
conducting) Casimir pressure between parallel plates, with optional
perturbative corrections for finite temperature and surface roughness. These
corrections are the standard leading-order expressions; they are not a
substitute for the full Lifshitz-theory treatment used in precision Casimir
metrology.

References
----------
- Milonni, P. W. (1994). The Quantum Vacuum. Academic Press.
- Casimir, H. B. G. (1948). Proc. K. Ned. Akad. Wet. 51, 793.
- Bordag, M., Klimchitskaya, G. L., Mohideen, U., & Mostepanenko, V. M.
  (2009). Advances in the Casimir Effect. Oxford University Press.
- Lamoreaux, S. K. (1997). Phys. Rev. Lett. 78, 5.

License: MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from scipy.integrate import quad

__all__ = [
    "Filter",
    "VacuumDensityResult",
    "calculate_vacuum_density",
    "calculate_casimir_pressure",
    "calculate_casimir_pressure_corrected",
    "thermal_correction_factor",
    "roughness_correction_factor",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------
H_PLANCK: float = 6.626_070_15e-34   # Planck constant [J s]
HBAR: float = H_PLANCK / (2 * math.pi)
C_LIGHT: float = 299_792_458.0        # [m/s]
K_B: float = 1.380_649e-23            # [J/K]

# Derived prefactors used in the integrand and Casimir formula.
_PREFAC_RHO: float = HBAR / (2.0 * math.pi**2 * C_LIGHT**3)
_PREFAC_CASIMIR: float = (math.pi**2 / 240.0) * HBAR * C_LIGHT


# ---------------------------------------------------------------------------
# UV regulators
# ---------------------------------------------------------------------------
class Filter(Enum):
    """
    UV regulator (cutoff filter) families.

    Each filter f(x) is a smooth, monotonically decreasing function of
    x = omega / omega_c that suppresses high-frequency modes. The choice of
    filter is a modelling decision; physical observables that depend on the
    detailed shape of f are regulator-dependent and therefore unphysical
    unless renormalized away.
    """

    EXP = "exp"
    GAUSS = "gauss"
    LORENTZ = "lorentz"

    def function(self) -> Callable[[float], float]:
        """Return f(x) where x >= 0 is omega/omega_c."""
        if self is Filter.EXP:
            return lambda x: math.exp(-x)
        if self is Filter.GAUSS:
            return lambda x: math.exp(-(x**2))
        if self is Filter.LORENTZ:
            return lambda x: 1.0 / (1.0 + x**2)
        raise ValueError(f"Unknown filter: {self}")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VacuumDensityResult:
    """Container for the regularized vacuum energy density calculation."""

    value: float        # [J/m^3]
    error: float        # quadrature error estimate [J/m^3]
    nu_c_hz: float      # input cutoff frequency [Hz]
    filter_name: str

    def __str__(self) -> str:
        return (
            f"rho_vac = {self.value:.3e} +/- {self.error:.1e} J/m^3 "
            f"(nu_c = {self.nu_c_hz:.3e} Hz, filter = {self.filter_name})"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _check_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")


# ---------------------------------------------------------------------------
# Vacuum energy density
# ---------------------------------------------------------------------------
def calculate_vacuum_density(
    nu_c_hz: float,
    filter_type: str | Filter = Filter.EXP,
    n_cutoffs: float = 50.0,
) -> VacuumDensityResult:
    """
    Compute the regularized zero-point energy density of a real scalar field.

    Parameters
    ----------
    nu_c_hz : float
        Cutoff frequency [Hz]. Treated as a free model parameter.
    filter_type : str or Filter
        UV regulator family (default: exponential).
    n_cutoffs : float
        Upper integration limit, expressed in units of omega_c. The default
        n_cutoffs=50 is sufficient for all implemented filters because their
        contribution beyond ~20 omega_c is negligible at double precision.

    Returns
    -------
    VacuumDensityResult
        Computed density, quadrature error, and input parameters.

    Notes
    -----
    For an exponential cutoff the integral has a closed form:

        rho_vac = (hbar / 2 pi^2 c^3) * 6 * omega_c^4

    so any choice of nu_c reproducing the observed vacuum energy density
    (~5.4e-10 J/m^3) is, by construction, consistent with the dimensional
    analysis of holographic dark energy models (Cohen, Kaplan & Nelson,
    Phys. Rev. Lett. 82, 4971 (1999)). The fact that some choice of nu_c
    yields the observed value is therefore not a prediction.
    """
    _check_positive(nu_c_hz, "nu_c_hz")
    _check_positive(n_cutoffs, "n_cutoffs")

    if isinstance(filter_type, str):
        filter_type = Filter(filter_type)

    f = filter_type.function()
    omega_c = 2.0 * math.pi * nu_c_hz
    omega_max = n_cutoffs * omega_c

    def integrand(omega: float) -> float:
        return _PREFAC_RHO * omega**3 * f(omega / omega_c)

    value, error = quad(
        integrand,
        0.0,
        omega_max,
        limit=2000,
        epsabs=1e-14,
        epsrel=1e-12,
    )

    if not np.isfinite(value):
        logger.warning(
            "Non-finite result at nu_c=%.3e Hz, filter=%s",
            nu_c_hz,
            filter_type.value,
        )

    return VacuumDensityResult(
        value=value,
        error=error,
        nu_c_hz=nu_c_hz,
        filter_name=filter_type.value,
    )


# ---------------------------------------------------------------------------
# Casimir pressure
# ---------------------------------------------------------------------------
def calculate_casimir_pressure(d_m: float) -> float:
    """
    Ideal Casimir pressure between parallel perfectly conducting plates at T=0.

    Parameters
    ----------
    d_m : float
        Plate separation [m]. Must be > 0.

    Returns
    -------
    float
        Pressure [Pa]. Negative sign denotes attraction.

        P = - pi^2 hbar c / (240 d^4)

    Notes
    -----
    This is the textbook Casimir result. It does not include finite-conductivity
    or geometry corrections; for precision metrology use the Lifshitz formula
    with measured optical data of the plate material.
    """
    _check_positive(d_m, "d_m")
    return -_PREFAC_CASIMIR / d_m**4


def thermal_correction_factor(d_m: float, temp_k: float) -> float:
    """
    Leading-order thermal correction factor C_th(d, T) for the Casimir pressure.

    Valid in the low-temperature regime k_B T d / (hbar c) << 1. In that
    regime the thermal correction is exponentially small:

        C_th ~ 1 + (2/pi) beta * exp(-beta),  beta = hbar c / (k_B T d)

    Outside this regime one must use the full Matsubara sum.

    Parameters
    ----------
    d_m : float
        Plate separation [m].
    temp_k : float
        Temperature [K]. T=0 returns 1 exactly.
    """
    _check_positive(d_m, "d_m")
    if temp_k < 0:
        raise ValueError(f"temp_k must be non-negative, got {temp_k!r}")
    if temp_k == 0.0:
        return 1.0

    beta = HBAR * C_LIGHT / (K_B * temp_k * d_m)
    if beta < 1.0:
        logger.warning(
            "Low-T expansion may be inaccurate: beta=%.3f < 1 "
            "(d=%.3e m, T=%.3e K). Use full Matsubara sum instead.",
            beta,
            d_m,
            temp_k,
        )
    if beta > 100.0:
        return 1.0
    return 1.0 + (2.0 / math.pi) * beta * math.exp(-beta)


def roughness_correction_factor(d_m: float, roughness_rms_m: float) -> float:
    """
    Perturbative roughness correction to the ideal Casimir pressure.

    Leading-order expansion in (sigma/d) for stochastic surface roughness:

        C_rough = 1 + 6 * (sigma/d)^2 + O((sigma/d)^4)

    This is the lowest-order result of Genet, Lambrecht & Reynaud (2003);
    it is reliable only for sigma/d << 0.1.

    Parameters
    ----------
    d_m : float
        Plate separation [m].
    roughness_rms_m : float
        RMS surface roughness [m].
    """
    _check_positive(d_m, "d_m")
    if roughness_rms_m < 0:
        raise ValueError(
            f"roughness_rms_m must be non-negative, got {roughness_rms_m!r}"
        )
    ratio = roughness_rms_m / d_m
    if ratio > 0.1:
        logger.warning(
            "Perturbative expansion likely invalid: sigma/d=%.3f > 0.1",
            ratio,
        )
    return 1.0 + 6.0 * ratio**2


def calculate_casimir_pressure_corrected(
    d_m: float,
    temp_k: float = 0.0,
    roughness_rms_m: float = 0.0,
) -> float:
    """
    Casimir pressure with leading-order thermal and roughness corrections.

    Parameters
    ----------
    d_m : float
        Plate separation [m].
    temp_k : float
        Temperature [K] (default 0).
    roughness_rms_m : float
        RMS surface roughness [m] (default 0).

    Returns
    -------
    float
        Corrected pressure [Pa]. Negative = attractive.

    Notes
    -----
    The two corrections are applied multiplicatively, which is correct only
    at leading order. For precision work use a full numerical Lifshitz
    calculation with the appropriate dielectric data and measured surface
    profiles.
    """
    p_ideal = calculate_casimir_pressure(d_m)
    c_th = thermal_correction_factor(d_m, temp_k)
    c_rough = roughness_correction_factor(d_m, roughness_rms_m)
    return p_ideal * c_th * c_rough
