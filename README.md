# vacuum_energy

High-precision, reproducible Python package for computing vacuum energy density (\( \rho_{\text{vac}} \)) and Casimir force per unit area, supporting the manuscript *"h como ciclo y evento elemental: Reinterpretación conceptual de la constante de Planck"* (arXiv:). Version 9.2.1 includes enhanced numerical robustness, thermal corrections, and publication-ready figures to address reviewer feedback.

## Installation

Install the `vacuum_energy` package via PyPI:


pip install vacuum-energy==9.2.1

Alternatively, install from source with dependencies:
bash
pip install -r requirements.txt
Usage

The package provides a command-line interface (CLI) for computing ( \rho_{\text{vac}} ), Casimir force, and generating figures. Examples:

    Calculate ( \rho_{\text{vac}} ) with an exponential filter (( \nu_c = 7.275 \times 10^{11} ) Hz):
    

python vacuum_energy.py --nu-c 7.275e11 --filter exp

Generate convergence plot for ( \rho_{\text{vac}} ):
bash
python vacuum_energy.py --plot convergence --nu-c 7.275e11

Compare Casimir force (with and without thermal correction) for plate separations 0.05–2 mm:
bash
python vacuum_energy.py --plot casimir_abs --d-min 0.05 --d-max 2.0

Generate all figures with verbose logging:
bash
python vacuum_energy.py --generate-all -vv

Show version:
bash

    python vacuum_energy.py --version

API

The package provides functions for vacuum energy and Casimir force calculations:
python
from vacuum_energy import (
    calculate_vacuum_density,
    calculate_casimir_force,
    calculate_casimir_force_thermal,
    calculate_lamb_shift_correction,
    validate_nu_c
)

# Vacuum energy density (J m⁻³)
rho = calculate_vacuum_density(nu_c_hz=7.275e11, filter_type="exp").value

# Casimir force per unit area (Pa, negative = attractive)
F = calculate_casimir_force(d_m=0.5e-3, nu_c_hz=7.275e11)

# Casimir force with thermal correction (T=4 K)
F_thermal = calculate_casimir_force_thermal(d_m=0.5e-3, nu_c_hz=7.275e11, temp_k=4.0)

# Lamb shift correction (relative to standard QED)
lamb_corr = calculate_lamb_shift_correction(nu_c_hz=7.275e11)

# Relative deviation of ρ_vac from observed value
deviation = validate_nu_c(nu_c_hz=7.275e11)
Features

    Computations: High-precision ( \rho_{\text{vac}} ), Casimir force (with thermal correction), and Lamb shift correction.
    Plots: Publication-ready figures (300 DPI, LaTeX style) for Casimir force comparison, sensitivity analysis, ( \rho_{\text{vac}} ) vs. ( \nu_c ), absolute Casimir force, and convergence analysis.
    Filters: Supports exponential, Gaussian, Lorentzian, and non-local UV suppression filters.
    Parallelism: Optional joblib for faster computations.
    Export: Raw data in CSV/TXT for reproducibility.
    Self-tests: Built-in checks for ( \rho_{\text{vac}} ), Casimir force sign, and Lamb shift correction.

Dependencies

    Required: NumPy ≥ 1.21, SciPy ≥ 1.7
    Optional: Matplotlib ≥ 3.8 (for --plot), joblib ≥ 1.1 (for parallelism), ipywidgets ≥ 8 (for interactive plots)

Exceptions

    VacuumEnergyError: Base error for package issues.
    InputError: Invalid physical parameters (e.g., non-positive values).
    MissingDependencyError: Missing optional dependencies for plotting or parallelism.

Contributing

Contributions are welcome! Please submit issues or pull requests to the GitHub repository. Ensure code adheres to PEP 8 and includes tests in the tests/ directory.
License

MIT License © 2025 Juan Galaz & collaborators. If used in academic work, please cite:
text
Galaz, J. (2025). h como ciclo y evento elemental: Reinterpretación conceptual de la constante de Planck. arXiv:forthcoming
