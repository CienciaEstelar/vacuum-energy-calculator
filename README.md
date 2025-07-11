# vacuum_energy

## Installation

To install the `vacuum_energy` package, you can use pip:

```
pip install vacuum_energy-8.0-py3-none-any.whl
```

Alternatively, you can install the development environment by running:

```
pip install -r requirements.txt
```

## Usage

Here are some examples of how to use the `vacuum_energy` package:

1. Calculate the vacuum energy density (ρ_vac) with a Gaussian filter (ν_c = 1 × 10¹² Hz):

   ```
   python vacuum_energy.py --nu-c 1e12 --filtro gauss
   ```

2. Compare the Casimir force in the range of 0.1–1 mm:

   ```
   python vacuum_energy.py --plot casimir_comp --d-min 0.1 --d-max 1.0
   ```

3. Generate all the figures with verbose logging:

   ```
   python vacuum_energy.py --generate-all -vv
   ```

4. Show the version and exit:

   ```
   python vacuum_energy.py --version
   ```

## API

The package provides a minimal API with the following functions:

```python
from vacuum_energy import calcular_rho_vacio, calcular_fuerza_casimir

rho, err = calcular_rho_vacio(7.275e11)       # ρ_vac in J m⁻³
F = calcular_fuerza_casimir(0.5e-3, 7.275e11) # force in Pa
```

## Dependencies

The package requires the following dependencies:

- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Matplotlib ≥ 3.8 (optional for `--plot`)
- joblib ≥ 1.1 (optional)
- ipywidgets ≥ 8 (optional)

## Exceptions

The package can raise the following exceptions:

- `ValueError` – for parameters outside the physical range.
- `RuntimeError` – for missing required dependencies.

## Contributing

Contributions to the `vacuum_energy` package are welcome. Please submit any issues or pull requests to the project's repository.

## License

This project is licensed under the MIT License. If you use this package, please cite the reference `arXiv:update`.

## Testing

You can run the package's self-tests by calling the `run_tests()` function, which verifies the reference values and the sign of the force.
