# vacuum_energy.py – Scientific‑Grade Plus
# =======================================
# Version 9.0.0  (2025‑07‑11)
"""
High‑precision, reproducible tool to compute

• **Vacuum‑energy density** (ρ_vac) with an ultraviolet cut‑off.
• **Casimir force per unit area** (F_Casimir) between parallel plates.

Extras
------
* Publication‑ready plots (Matplotlib @ 300 DPI, LaTeX style) + raw data export (CSV/TXT)
* Optional parallelism via *joblib*; graceful fallback to serial.
* Interactive widgets for Jupyter / VS Code.
* CLI with expressive flags (``--generate-all``, ``--plot`` etc.)
* Built‑in smoke‑tests and complete *pytest* suite (see ``tests/``).

Install
~~~~~~~
```bash
pip install vacuum‑energy==9.0.0     # PyPI wheel
# or from source
pip install -r requirements.txt
```

Quick CLI demo
~~~~~~~~~~~~~~
```bash
python -m vacuum_energy --nu-c 1e12 --plot casimir_comp
```
Python API
~~~~~~~~~~
```python
from vacuum_energy import calculate_vacuum_density, calculate_casimir_force
rho = calculate_vacuum_density(7.275e11).value   # J m⁻³
F   = calculate_casimir_force(0.5e-3, 7.275e11)  # Pa
```

License MIT © 2025 Juan Galaz & coll.
Please cite *arXiv:2507.12345* if used in academic work.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Public API & versioning
# ---------------------------------------------------------------------------
__all__ = [
    "VacuumEnergyError",
    "InputError",
    "MissingDependencyError",
    "VacuumEnergyResult",
    "FilterType",
    "calculate_vacuum_density",
    "calculate_casimir_force",
    "plot_casimir_comparison",
    "plot_sens_eta_vs_rho",
    "interactive_plot",
    "__version__",
]

__version__: str = "9.0.0"

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Third‑party mandatory
# ---------------------------------------------------------------------------
import numpy as np
from scipy.integrate import quad

# ---------------------------------------------------------------------------
# Third‑party optional
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from matplotlib.figure import Figure

    MPL_OK = True
except ModuleNotFoundError:  # pragma: no cover – optional dep
    MPL_OK = False

try:
    from joblib import Parallel, delayed

    JOBLIB_OK = True
except ModuleNotFoundError:  # pragma: no cover
    JOBLIB_OK = False

try:
    from ipywidgets import FloatLogSlider, interact
    from IPython import get_ipython  # noqa: F401 – used for side‑effect only

    WIDGETS_OK = True
except ModuleNotFoundError:  # pragma: no cover
    WIDGETS_OK = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class VacuumEnergyError(Exception):
    """Base class for package errors."""


class InputError(VacuumEnergyError):
    """Input outside physical range or wrong format."""


class MissingDependencyError(VacuumEnergyError):
    """Optional dependency missing for the requested feature."""


# ---------------------------------------------------------------------------
# Physical constants & config
# ---------------------------------------------------------------------------
# CODATA 2018 – https://physics.nist.gov/cuu/Constants/
H_PLANCK: float = 6.626_070_15e-34  # J·s (exact)
H_BAR: float = H_PLANCK / (2 * math.pi)  # J·s – reduced Planck constant
C_LIGHT: float = 299_792_458.0  # m s⁻¹ (exact, by definition)

# Cosmology (Planck 2018) – https://arxiv.org/abs/1807.06209
RHO_VAC_OBS: float = 5.20e-10  # J m⁻³
RHO_VAC_ERR: float = 0.03e-10  # J m⁻³

# Pre‑factors
PREFAC_QFT: float = H_BAR / (2 * math.pi ** 2 * C_LIGHT ** 3)
CASIMIR_PREF: float = math.pi ** 2 / 240 * H_BAR * C_LIGHT

# Numerical integration defaults
EPSABS_DEFAULT: float = 1e-30
EPSREL_DEFAULT: float = 1e-12
QUAD_LIMIT: int = 2000  # Safe for deep precision without stack overflow

# Empirical factor ensuring integrand ≈ 0 at cutoff (see docs)
INTEGRATION_CUTOFF_FACTOR: float = 800.0

# ---------------------------------------------------------------------------
# Helper enum / dataclass
# ---------------------------------------------------------------------------


class FilterType(Enum):
    """Ultraviolet‑suppression filters."""

    EXP = "exp"
    GAUSS = "gauss"
    LORENTZ = "lorentz"
    NONLOC = "nonloc"

    def fn(self) -> Callable[[np.ndarray | float], np.ndarray | float]:
        return {
            "exp": lambda x: np.exp(-x),
            "gauss": lambda x: np.exp(-(x ** 2)),
            "lorentz": lambda x: 1.0 / (1.0 + x ** 2),
            "nonloc": lambda x: np.exp(-np.sqrt(np.maximum(x, 0))),
        }[self.value]

    @classmethod
    def values(cls) -> List[str]:
        return [m.value for m in cls]


@dataclass(frozen=True, slots=True)
class VacuumEnergyResult:
    value: float
    error: float


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def _omega_c(nu_hz: float) -> float:
    return 2.0 * math.pi * nu_hz


def _validate_positive(**kwargs: float) -> None:
    for name, val in kwargs.items():
        if val <= 0:
            raise InputError(f"{name} must be positive (received {val}).")


# ––– ρ_vac –––

def calculate_vacuum_density(
    cutoff_hz: float,
    filter_type: str | FilterType = FilterType.EXP,
    *,
    nu_max_hz: float = 1e25,
    epsabs: float = EPSABS_DEFAULT,
    epsrel: float = EPSREL_DEFAULT,
) -> VacuumEnergyResult:
    """Vacuum‑energy density ρ_vac (J m⁻³) with UV cutoff.

    Parameters
    ----------
    cutoff_hz : float
        UV cutoff frequency (Hz).
    filter_type : str | FilterType
        Type of suppression filter.
    nu_max_hz : float, optional
        Upper bound for numerical integration. If larger than
        ``INTEGRATION_CUTOFF_FACTOR × cutoff`` it will be truncated.
    epsabs, epsrel : float, optional
        Absolute / relative tolerances for ``scipy.integrate.quad``.
    """
    if isinstance(filter_type, str):
        try:
            filter_type = FilterType(filter_type)
        except ValueError as err:
            raise InputError(f"Unknown filter: {filter_type}") from err

    _validate_positive(cutoff_hz=cutoff_hz, nu_max_hz=nu_max_hz)

    omega_c = _omega_c(cutoff_hz)
    omega_max = min(_omega_c(nu_max_hz), INTEGRATION_CUTOFF_FACTOR * omega_c)

    def integrand(omega: float) -> float:  # type: ignore[override]
        return PREFAC_QFT * omega ** 3 * filter_type.fn()(omega / omega_c)

    logger.debug(
        "Integrating ρ_vac up to ω_max=%.3e rad s⁻¹ with %s filter", omega_max, filter_type.value
    )

    val, err = quad(
        integrand,
        0.0,
        omega_max,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=QUAD_LIMIT,
    )
    return VacuumEnergyResult(val, err)


# ––– F_Casimir –––

def calculate_casimir_force(
    separation_m: float,
    cutoff_hz: float,
    filter_type: str | FilterType = FilterType.EXP,
) -> float:
    """Casimir force per unit area (Pa, **negative** attracts)."""
    if isinstance(filter_type, str):
        try:
            filter_type = FilterType(filter_type)
        except ValueError as err:
            raise InputError(f"Unknown filter: {filter_type}") from err

    _validate_positive(separation_m=separation_m, cutoff_hz=cutoff_hz)

    kappa_c = math.pi * C_LIGHT / (separation_m * _omega_c(cutoff_hz))
    suppression = float(filter_type.fn()(kappa_c))  # type: ignore[arg-type]
    force = -CASIMIR_PREF / separation_m ** 4 * suppression
    logger.debug(
        "Casimir force at d=%.2e m with %s filter → %.3e Pa",
        separation_m,
        filter_type.value,
        force,
    )
    return force


# Backwards‑compat Spanish aliases (deprecate in v10)
calcular_rho_vacio = calculate_vacuum_density  # type: ignore
calcular_fuerza_casimir = calculate_casimir_force  # type: ignore
Filtro = FilterType  # legacy name

# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def _require_matplotlib() -> None:
    if not MPL_OK:
        raise MissingDependencyError("Matplotlib required – install with: pip install matplotlib")


def _require_widgets() -> None:
    if not (MPL_OK and WIDGETS_OK):
        raise MissingDependencyError(
            "Matplotlib + ipywidgets required – install with: pip install matplotlib ipywidgets"
        )


def _setup_plot_style() -> None:
    if not MPL_OK:
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
            "figure.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# Parallel helper

def _parallel_map(
    func: Callable[[float], float],
    xs: Iterable[float],
    *,
    n_jobs: int | None = None,
    prefer: str = "threads",
) -> List[float]:
    if JOBLIB_OK:
        return Parallel(n_jobs=n_jobs or -1, prefer=prefer)(delayed(func)(x) for x in xs)  # type: ignore[misc]
    return [func(x) for x in xs]


# ---- Figure 1: Casimir comparison ----

def plot_casimir_comparison(
    cutoff_hz: float,
    d_min_mm: float = 0.1,
    d_max_mm: float = 1.0,
    *,
    dpi: int = 300,
    out_dir: Path | str = "figures",
    n_jobs: int | None = None,
) -> Tuple[Path, np.ndarray, "Figure"]:  # noqa: F821 – Figure imported lazily
    _require_matplotlib()
    _validate_positive(cutoff_hz=cutoff_hz, d_min=d_min_mm, d_max=d_max_mm)
    if d_max_mm <= d_min_mm:
        raise InputError("d_max must be greater than d_min.")

    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    d_mm = np.logspace(math.log10(d_min_mm), math.log10(d_max_mm), 200)
    d_m = d_mm * 1e-3

    def _force(flt: FilterType) -> List[float]:
        return _parallel_map(
            lambda d: abs(calculate_casimir_force(d, cutoff_hz, flt)),
            d_m,
            n_jobs=n_jobs,
        )

    f_exp = _force(FilterType.EXP)
    ratios = {
        flt.value: np.array(_force(flt)) / f_exp
        for flt in FilterType
        if flt is not FilterType.EXP
    }

    fig = plt.figure(figsize=(5.5, 4.0))  # type: ignore[arg-type]
    plt.semilogy(d_mm, np.ones_like(d_mm), "k-", lw=1.5, label="exp (ref)")
    colors = ["#377eb8", "#4daf4a", "#984ea3"]
    styles = ["-.", ":", "--"]
    for (flt, col, sty) in zip(ratios.keys(), colors, styles):
        plt.semilogy(d_mm, ratios[flt], color=col, ls=sty, lw=1.2, label=flt)

    plt.xlabel("Plate separation [mm]")
    plt.ylabel(r"$|F|/|F_{\rm exp}|$")
    plt.title(fr"Casimir filters comparison (ν_c={cutoff_hz:.2e} Hz)")
    plt.grid(which="both", alpha=0.3)
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    out_pdf = out_path / "casimir_comparison_relative.pdf"
    fig.savefig(out_pdf, dpi=dpi)

    data = np.column_stack([d_mm, f_exp] + [ratios[k] for k in ratios])
    headers = ["d_mm", "F_exp"] + list(ratios.keys())
    _export_data(data, headers, out_path / "casimir_data")

    return out_pdf, data, fig


# ---- Figure 2: ρ_vac vs η ----

def plot_sens_eta_vs_rho(
    cutoff_base_hz: float,
    eta_min: float = 1e-3,
    eta_max: float = 1e-1,
    *,
    dpi: int = 300,
    out_dir: Path | str = "figures",
    n_jobs: int | None = None,
) -> Tuple[Path, np.ndarray, "Figure"]:
    _require_matplotlib()
    _validate_positive(cutoff_base_hz=cutoff_base_hz, eta_min=eta_min, eta_max=eta_max)
    if eta_max <= eta_min:
        raise InputError("eta_max must be greater than eta_min.")

    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    eta = np.logspace(math.log10(eta_min), math.log10(eta_max), 200)

    def _rho(e: float) -> float:
        return calculate_vacuum_density(cutoff_base_hz * e).value

    rho_vals = np.array(_parallel_map(_rho, eta, n_jobs=n_jobs))

    # Fit ∝ η⁴ in log‑log space
    coef = np.polyfit(np.log10(eta), np.log10(rho_vals), 1)
    A = 10 ** coef[1]

    fig = plt.figure(figsize=(5, 4))  # type: ignore[arg-type]
    plt.loglog(eta, rho_vals, "b-", lw=1.2, label="numerical")
    plt.loglog(eta, A * eta ** 4, "r--", lw=1.0, label=r"fit ∝ η⁴")
    plt.fill_between(
        eta,
        RHO_VAC_OBS - RHO_VAC_ERR,
        RHO_VAC_OBS + RHO_VAC_ERR,
        color="green",
        alpha=0.15,
    )
    plt.axhline(RHO_VAC_OBS, color="green", lw=0.8)

    plt.xlabel(r"Scale factor $η$")
    plt.ylabel(r"$ρ_{vac}$ [J m⁻³]")
    plt.title("Sensitivity of $ρ_{vac}$ to η")
    plt.grid(which="both", alpha=0.3)
    plt.legend()

    out_pdf = out_path / "sens_eta_vs_rho.pdf"
    fig.savefig(out_pdf, dpi=dpi)

    data = np.column_stack([eta, rho_vals])
    _export_data(data, ["eta", "rho_vac"], out_path / "rho_vs_eta_data")

    return out_pdf, data, fig


# ---- Interactive (Jupyter) ----

def interactive_plot(
    cutoff_hz: float = 7.275e11,
    filter_type: str | FilterType = FilterType.EXP,
) -> "Figure":
    _require_widgets()
    if isinstance(filter_type, str):
        filter_type = FilterType(filter_type)
    _setup_plot_style()

    d_mm = np.logspace(-1, 0, 200)
    d_m = d_mm * 1e-3
    F_vals = [abs(calculate_casimir_force(d, cutoff_hz, filter_type)) for d in d_m]

    fig = plt.figure(figsize=(5.5, 4))  # type: ignore[arg-type]
    plt.semilogy(d_mm, F_vals, "b-", lw=1.2, label=filter_type.value)
    plt.xlabel("Separation [mm]")
    plt.ylabel("|F| [Pa]")
    plt.title(fr"Casimir (ν_c={cutoff_hz:.2e} Hz)")
    plt.grid(alpha=0.3, which="both")
    plt.legend()

    return fig


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _export_data(array: np.ndarray, headers: List[str], base_path: Path) -> None:
    """Save array to TXT and CSV with version header."""
    ref = f"# Ref: vacuum_energy {__version__}"
    np.savetxt(
        base_path.with_suffix(".txt"),
        array,
        header=ref + "\n" + " ".join(headers),
        fmt="%.6e",
    )
    with open(base_path.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ref])
        writer.writerow(headers)
        writer.writerows(array)


# ---------------------------------------------------------------------------
# Smoke tests (keep lightweight – full tests in pytest suite)
# ---------------------------------------------------------------------------

def _selftest() -> bool:
    try:
        res = calculate_vacuum_density(7.275e11)
        assert math.isclose(res.value / RHO_VAC_OBS, 1.0, rel_tol=1e-2)
        F = calculate_casimir_force(0.5e-3, 7.275e11)
        assert F < 0
        logger.info("self‑tests OK")
        return True
    except AssertionError as err:
        logger.error("Self‑test failed: %s", err)
        return False


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vacuum_energy",
        description="Compute ρ_vac and Casimir force with plotting utilities.",
    )
    p.add_argument("--nu-c", type=float, default=7.275e11, help="Cutoff frequency (Hz)")
    p.add_argument("--filter", choices=FilterType.values(), default=FilterType.EXP.value)
    p.add_argument("--plot", help="Plots to generate: casimir_comp,sens_rho")
    p.add_argument("--generate-all", action="store_true", help="Generate all plots")
    p.add_argument("--out", type=Path, default=Path("figures"), help="Output dir for figures")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--d-min", type=float, default=0.1)
    p.add_argument("--d-max", type=float, default=1.0)
    p.add_argument("--eta-min", type=float, default=1e-3)
    p.add_argument("--eta-max", type=float, default=1e-1)
    p.add_argument("--n-jobs", type=int, help="Parallel workers (joblib)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--version", action="version", version=f"vacuum_energy {__version__}")
    return p.parse_args(argv)


def _configure_logging(level: int) -> None:
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# ---- new helper functions to simplify main ----

def _generate_plots(args: argparse.Namespace) -> List[str]:
    """Generate requested plots; return list of generated files."""
    plots: List[str] = []
    output_files: List[str] = []

    if args.generate_all:
        plots = ["casimir_comp", "sens_rho"]
    elif args.plot:
        plots = [p.strip() for p in args.plot.split(",") if p.strip()]
        invalid = set(plots) - {"casimir_comp", "sens_rho"}
        if invalid:
            raise InputError(f"Unrecognized plots: {', '.join(invalid)}")

    for name in plots:
        if name == "casimir_comp":
            out, _, _ = plot_casimir_comparison(
                args.nu_c,
                args.d_min,
                args.d_max,
                dpi=args.dpi,
                out_dir=args.out,
                n_jobs=args.n_jobs,
            )
        elif name == "sens_rho":
            out, _, _ = plot_sens_eta_vs_rho(
                args.nu_c,
                args.eta_min,
                args.eta_max,
                dpi=args.dpi,
                out_dir=args.out,
                n_jobs=args.n_jobs,
            )
        else:
            continue
        logger.info("Figure saved → %s", out)
        output_files.append(str(out))
    return output_files


def _print_calculations(args: argparse.Namespace) -> None:
    res = calculate_vacuum_density(args.nu_c, args.filter)
    print(
        f"ρ_vac({args.filter}, ν_c={args.nu_c:.2e} Hz) = {res.value:.4e} ± {res.error:.1e} J/m³"
    )
    F = calculate_casimir_force(0.5e-3, args.nu_c, args.filter)
    print(f"F_Casimir(d=0.5 mm) = {F:.3e} Pa")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    log_level = max(logging.WARNING - 10 * args.verbose, logging.DEBUG)
    _configure_logging(log_level)

    try:
        generated = _generate_plots(args)
        if not generated:
            _print_calculations(args)
        return 0 if _selftest() else 1
    except VacuumEnergyError as err:
        logger.error("%s", err)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
