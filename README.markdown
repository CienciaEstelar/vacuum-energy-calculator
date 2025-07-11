```markdown
# Vacuum Energy Calculator (v8.0.0)

**Scientific-Grade Plus**: A reproducible Python tool for calculating the vacuum energy density (ρ_vac) and Casimir force per unit area (F_Casimir), designed for publication-ready outputs and academic research.

This tool is optimized for theoretical physics computations, featuring high-precision numerical integration, publication-quality visualizations, and robust error handling. It supports command-line usage, Jupyter notebooks, and PyPI packaging.

---

## Features

- **Accurate Physics Computations**: Calculates ρ_vac and F_Casimir using `scipy.quad` with high precision (EPSABS = 1e-40).
- **Publication-Ready Plots**: Generates 300 DPI PDF figures with LaTeX-like typography, exported alongside raw data in `.txt` and `.csv`.
- **Optional Parallelization**: Uses `joblib.Parallel` for efficient computation, with serial fallback if the dependency is absent.
- **Interactive Visualization**: Jupyter/VS Code-compatible widgets via `ipywidgets` for dynamic exploration.
- **Quick Tests**: Built-in `selftest()` verifies reference values and force signs.
- **Intuitive CLI**: Self-descriptive flags (`--generate-all`, `--version`) with incremental verbosity (`-v`, `-vv`).
- **PyPI-Ready**: Packaged with `pyproject.toml` and `vacuum_energy` entry-point.

---

## Installation

### Option 1: From PyPI (coming soon)
```bash
pip install vacuum_energy
```

### Option 2: From Source
```bash
git clone https://github.com/your-username/vacuum-energy.git
cd vacuum-energy
pip install -r requirements.txt
```

### Option 3: Install Wheel
```bash
pip install vacuum_energy-8.0.0-py3-none-any.whl
```

---

## Quick Start

1. **Calculate ρ_vac with Gaussian filter (ν_c = 10¹² Hz)**:
   ```bash
   python vacuum_energy.py --nu-c 1e12 --filtro gauss
   ```

2. **Generate Casimir force comparison (0.1–1 mm)**:
   ```bash
   python vacuum_energy.py --plot casimir_comp --d-min 0.1 --d-max 1.0
   ```

3. **Generate all standard figures with verbose logging**:
   ```bash
   python vacuum_energy.py --generate-all -vv
   ```

4. **Check version**:
   ```bash
   python vacuum_energy.py --version
   ```

5. **Use in Jupyter**:
   ```python
   from vacuum_energy import interactive_plot
   interactive_plot(nu_c=7.275e11, filtro="gauss")
   ```

---

## API Usage

```python
from vacuum_energy import calcular_rho_vacio, calcular_fuerza_casimir

# Vacuum energy density (J/m³)
rho, err = calcular_rho_vacio(7.275e11, filtro="exp")
print(f"ρ_vac = {rho:.3e} ± {err:.1e} J/m³")

# Casimir force per unit area (Pa)
F = calcular_fuerza_casimir(0.5e-3, 7.275e11, filtro="exp")
print(f"F_Casimir = {F:.3e} Pa")
```

---

## Requirements

- **Python**: ≥ 3.8
- **Mandatory**:
  - NumPy ≥ 1.21
  - SciPy ≥ 1.7
- **Optional**:
  - Matplotlib ≥ 3.8 (for `--plot`)
  - joblib ≥ 1.1 (for parallelization)
  - ipywidgets ≥ 8.0 (for Jupyter widgets)

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Exceptions

- `VacuumEnergyError`: Base exception for package errors.
- `InputError`: Raised for invalid physical parameters (e.g., negative distances or frequencies).
- `MissingDependencyError`: Raised when optional dependencies are required but not installed.

---

## Testing

Run basic self-tests:
```bash
python vacuum_energy.py -vv
```

For comprehensive testing, use `pytest`:
```bash
pytest tests/test_vacuum_energy.py -v
```

---

## Citation

If you use this software in your research, please cite:
> Galaz, J. et al. (2025). "Reinterpretando la Constante de Planck: El Ciclo Elemental y su Impacto en la Densidad de Energía del Vacío." *arXiv:2507.12345*.

---

## License

© 2025 Juan Galaz & collaborators. Released under the [MIT License](LICENSE).

---

## Development

To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-addition`).
3. Commit changes (`git commit -m "Add awesome feature"`).
4. Push to the branch (`git push origin feature/awesome-addition`).
5. Open a pull request.

To build and test locally:
```bash
python -m build
pip install dist/vacuum_energy-8.0.0-py3-none-any.whl
pytest tests/
```

---

## Acknowledgments

- Built with inspiration from the Planck 2018 cosmological data and CODATA 2018 constants.
- Thanks to the open-source community for NumPy, SciPy, Matplotlib, and more.

```