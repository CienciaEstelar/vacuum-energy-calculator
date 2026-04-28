# VacuumPy

A pedagogical Python toolkit for UV-regularized vacuum energy density and Casimir pressure calculations in flat-spacetime QFT.

## What this is

A small, well-tested, honestly-documented reference implementation of three textbook calculations:

1. The regularized zero-point energy density of a free scalar field with a smooth UV cutoff (`exp`, `Gaussian`, `Lorentzian`).
2. The ideal Casimir pressure between parallel plates at T=0.
3. Leading-order thermal and roughness corrections to the Casimir pressure.

It is intended for students of QFT and cosmology, for use in lectures and notebooks, and as a clean starting point for more sophisticated work (e.g. full Lifshitz-theory Casimir calculations, finite-temperature field theory).

## What this is *not*

This code does **not**:

- Solve the cosmological constant problem.
- Predict the mass of any axion-like particle.
- Constitute a new theory of gravity, dark energy, or the vacuum.

The cutoff frequency `nu_c` is treated as a **free model parameter**. Choosing `nu_c` such that the regularized density reproduces the observationally inferred vacuum energy density (~5.4e-10 J/m^3) is a dimensional consistency check, not a physical prediction. The same dimensional logic underlies holographic dark energy (Cohen, Kaplan & Nelson, *Phys. Rev. Lett.* **82**, 4971, 1999); we do not improve on that argument.

## Installation

```bash
git clone <repo>
cd vacuumpy
pip install -e .
```

Dependencies: `numpy`, `scipy`. Optional: `matplotlib` (examples), `pytest` (tests).

## Usage

```python
from vacuumpy import Filter, calculate_vacuum_density, calculate_casimir_pressure

# Vacuum energy density at nu_c = 1 THz with exponential cutoff
result = calculate_vacuum_density(1e12, Filter.EXP)
print(result)
# rho_vac = 7.66e-08 +/- 1.0e-19 J/m^3 (nu_c = 1.000e+12 Hz, filter = exp)

# Ideal Casimir pressure at 1 micron separation
P = calculate_casimir_pressure(1e-6)
print(f"{P:.3e} Pa")  # -1.301e-03 Pa
```

See `examples/` for a complete script.

## Tests

```bash
pytest tests/ -v
```

Twenty unit tests covering closed-form integrals, scaling laws, sign conventions, limiting behaviour, and input validation.

## References

- Milonni, P. W. (1994). *The Quantum Vacuum*. Academic Press.
- Bordag, M. et al. (2009). *Advances in the Casimir Effect*. Oxford University Press.
- Casimir, H. B. G. (1948). *Proc. K. Ned. Akad. Wet.* **51**, 793.
- Cohen, A. G., Kaplan, D. B., & Nelson, A. E. (1999). *Phys. Rev. Lett.* **82**, 4971.
- Genet, C., Lambrecht, A., & Reynaud, S. (2003). *Phys. Rev. A* **67**, 043811.

## License

MIT.

## Citation

If you use this code, please cite it via the Zenodo DOI listed in `CITATION.cff`.
