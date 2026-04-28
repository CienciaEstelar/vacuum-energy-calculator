"""
VacuumPy: A pedagogical toolkit for UV-regularized vacuum energy and
Casimir pressure calculations.

See vacuumpy.core for the public API.
"""

from vacuumpy.core import (
    Filter,
    VacuumDensityResult,
    calculate_casimir_pressure,
    calculate_casimir_pressure_corrected,
    calculate_vacuum_density,
    roughness_correction_factor,
    thermal_correction_factor,
)

__version__ = "0.1.0"

__all__ = [
    "Filter",
    "VacuumDensityResult",
    "calculate_vacuum_density",
    "calculate_casimir_pressure",
    "calculate_casimir_pressure_corrected",
    "thermal_correction_factor",
    "roughness_correction_factor",
    "__version__",
]
