"""
Tests for vacuumpy.core.

These tests verify:
1. Closed-form check of the exponential cutoff integral.
2. Casimir pressure sign and magnitude at known scales.
3. Limiting behaviour of thermal and roughness corrections.
4. Input validation.
"""

import math

import numpy as np
import pytest

from vacuumpy.core import (
    HBAR,
    C_LIGHT,
    Filter,
    calculate_casimir_pressure,
    calculate_casimir_pressure_corrected,
    calculate_vacuum_density,
    roughness_correction_factor,
    thermal_correction_factor,
)


# ---------------------------------------------------------------------------
# Vacuum density
# ---------------------------------------------------------------------------
class TestVacuumDensity:
    def test_exponential_closed_form(self):
        """
        For f(x) = exp(-x), the integral has the closed form
            rho = (hbar / 2 pi^2 c^3) * 6 * omega_c^4
        Verify numerical quadrature reproduces this to high precision.
        """
        nu_c = 1e12
        omega_c = 2.0 * math.pi * nu_c
        expected = (HBAR / (2.0 * math.pi**2 * C_LIGHT**3)) * 6.0 * omega_c**4

        result = calculate_vacuum_density(nu_c, Filter.EXP)
        assert result.value == pytest.approx(expected, rel=1e-8)
        assert result.error < 1e-6 * abs(result.value)

    def test_gaussian_scaling(self):
        """rho should scale as omega_c^4 for any fixed-shape filter."""
        r1 = calculate_vacuum_density(1e12, Filter.GAUSS)
        r2 = calculate_vacuum_density(2e12, Filter.GAUSS)
        # doubling nu_c => factor of 16
        assert r2.value / r1.value == pytest.approx(16.0, rel=1e-6)

    def test_lorentzian_smaller_than_exp(self):
        """At fixed nu_c, the Lorentzian filter has heavier tail
        than the exponential, so rho_lorentz > rho_exp."""
        nu_c = 1e12
        rho_exp = calculate_vacuum_density(nu_c, Filter.EXP).value
        rho_lor = calculate_vacuum_density(nu_c, Filter.LORENTZ).value
        assert rho_lor > rho_exp

    def test_filter_enum_and_string_equivalent(self):
        nu_c = 1e12
        a = calculate_vacuum_density(nu_c, "exp").value
        b = calculate_vacuum_density(nu_c, Filter.EXP).value
        assert a == pytest.approx(b, rel=1e-12)

    @pytest.mark.parametrize("nu_c", [-1.0, 0.0, float("inf"), float("nan")])
    def test_invalid_nu_c_raises(self, nu_c):
        with pytest.raises(ValueError):
            calculate_vacuum_density(nu_c, Filter.EXP)


# ---------------------------------------------------------------------------
# Casimir pressure
# ---------------------------------------------------------------------------
class TestCasimirPressure:
    def test_ideal_known_value_at_1um(self):
        """
        P = -pi^2 hbar c / (240 d^4)
        At d = 1 um: P = -1.30e-3 Pa  (Bordag et al. 2009, eq. 2.11)
        """
        d = 1e-6
        p = calculate_casimir_pressure(d)
        expected = -(math.pi**2 / 240.0) * HBAR * C_LIGHT / d**4
        assert p == pytest.approx(expected, rel=1e-12)
        assert p == pytest.approx(-1.30e-3, rel=1e-2)

    def test_attractive(self):
        assert calculate_casimir_pressure(1e-6) < 0

    def test_inverse_fourth_power_scaling(self):
        p1 = calculate_casimir_pressure(1e-6)
        p2 = calculate_casimir_pressure(2e-6)
        # halving distance => 16x stronger
        assert p1 / p2 == pytest.approx(16.0, rel=1e-12)

    def test_zero_distance_raises(self):
        with pytest.raises(ValueError):
            calculate_casimir_pressure(0.0)


# ---------------------------------------------------------------------------
# Thermal correction
# ---------------------------------------------------------------------------
class TestThermalCorrection:
    def test_zero_temperature_is_unity(self):
        assert thermal_correction_factor(1e-6, 0.0) == 1.0

    def test_low_T_correction_small(self):
        """At T=4 K, d=1 um: beta ~ 6900, correction is exp-suppressed."""
        c = thermal_correction_factor(1e-6, 4.0)
        assert c == pytest.approx(1.0, abs=1e-100)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            thermal_correction_factor(1e-6, -1.0)


# ---------------------------------------------------------------------------
# Roughness correction
# ---------------------------------------------------------------------------
class TestRoughnessCorrection:
    def test_zero_roughness_is_unity(self):
        assert roughness_correction_factor(1e-6, 0.0) == 1.0

    def test_quadratic_in_sigma_over_d(self):
        d = 1e-6
        sigma1 = 1e-8   # sigma/d = 0.01
        sigma2 = 2e-8   # sigma/d = 0.02
        c1 = roughness_correction_factor(d, sigma1) - 1.0
        c2 = roughness_correction_factor(d, sigma2) - 1.0
        # quadratic => factor 4 between corrections
        assert c2 / c1 == pytest.approx(4.0, rel=1e-12)

    def test_negative_roughness_raises(self):
        with pytest.raises(ValueError):
            roughness_correction_factor(1e-6, -1e-9)


# ---------------------------------------------------------------------------
# Combined pressure
# ---------------------------------------------------------------------------
class TestCorrectedPressure:
    def test_no_corrections_matches_ideal(self):
        d = 1e-6
        p1 = calculate_casimir_pressure(d)
        p2 = calculate_casimir_pressure_corrected(d, 0.0, 0.0)
        assert p1 == pytest.approx(p2, rel=1e-12)

    def test_roughness_increases_magnitude(self):
        """Roughness correction factor > 1, so |P| increases."""
        d = 1e-6
        p_ideal = calculate_casimir_pressure(d)
        p_rough = calculate_casimir_pressure_corrected(d, 0.0, 1e-8)
        assert abs(p_rough) > abs(p_ideal)
