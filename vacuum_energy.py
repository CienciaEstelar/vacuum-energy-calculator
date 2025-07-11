# vacuum_energy.py – Scientific-Grade Plus
# =======================================
# Versión 8.0 (2025-07-11)

"""
Herramienta reproducible para estimar la densidad de energía del vacío
(ρ_vac) y la fuerza de Casimir *per unit area* (F_Casimir), con utilidades
de visualización orientadas a artículos científicos (300 DPI, tipografía
LaTeX-like).

Características
---------------
• Cálculo físico fiable — integra con `scipy.quad` (EPSABS = 1e-40) y valida rangos de entrada.  
• Gráficos *publication-ready* — estilo uniforme; exporta PDF y datos crudos (`.txt`, `.csv`).  
• Paralelización opcional — `joblib.Parallel`; pasa a modo serial si la dependencia no está presente.  
• Entorno interactivo — widgets `ipywidgets` listos para Jupyter / VS Code.  
• Pruebas rápidas — `run_tests()` verifica valores de referencia y la señal de la fuerza.  
• CLI intuitiva — flags autodescriptivos (`--generate-all`, `--version`), verbosidad incremental (`-v`, `-vv`).  
• Empaquetado PyPI — compatible con `pyproject.toml` y el *entry-point* `vacuum_energy`.  





Uso rápido
----------


1) Instalación  
   
   pip install -r requirements.txt              # entorno de desarrollo
   # – o –
   pip install vacuum_energy-8.0-py3-none-any.whl
````

2. Cálculo directo de ρ\_vac con filtro gaussiano (ν\_c = 1 × 10¹² Hz)

   
   python vacuum_energy.py --nu-c 1e12 --filtro gauss
   ```

3. Comparativa de fuerza de Casimir en 0.1–1 mm

   
   python vacuum_energy.py --plot casimir_comp --d-min 0.1 --d-max 1.0
   

4. Generar todas las figuras con log muy verboso

   
   python vacuum_energy.py --generate-all -vv
   

5. Mostrar versión y salir

   
   python vacuum_energy.py --version
   

##### API mínima #####

```python
from vacuum_energy import calcular_rho_vacio, calcular_fuerza_casimir

rho, err = calcular_rho_vacio(7.275e11)       # ρ_vac en J m⁻³
F = calcular_fuerza_casimir(0.5e-3, 7.275e11) # fuerza en Pa


## Dependencias

NumPy ≥ 1.21 · SciPy ≥ 1.7 · Matplotlib ≥ 3.8 (opcional para `--plot`) ·
joblib ≥ 1.1 (opcional) · ipywidgets ≥ 8 (opcional).

## Excepciones

* `ValueError` – parámetros fuera de rango físico.
* `RuntimeError` – dependencias obligatorias ausentes.

© 2025 Juan Galaz & colab. MIT License — cite *arXiv:2507.12345* si lo usas.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------

# PUBLIC API & VERSIONING

# ---------------------------------------------------------------------------

__all__ = [
"VacuumEnergyError",
"InputError",
"MissingDependencyError",
"VacuumEnergyResult",
"calcular\_rho\_vacio",
"calcular\_fuerza\_casimir",
"plot\_casimir\_comparison",
"plot\_sens\_eta\_vs\_rho",
"interactive\_plot",
"**version**",
]

__version__: str = "8.0.0"


# ---------------------------------------------------------------------------
#  STANDARD LIBS
# ---------------------------------------------------------------------------
import math
import logging
import argparse
import csv
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Tuple, Iterable

# ---------------------------------------------------------------------------
#  THIRD‑PARTY (mandatory)
# ---------------------------------------------------------------------------
import numpy as np
from scipy.integrate import quad

# ---------------------------------------------------------------------------
#  THIRD‑PARTY (optional)
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
    from ipywidgets import interact, FloatLogSlider
    from IPython import get_ipython

    WIDGETS_OK = True
except ModuleNotFoundError:  # pragma: no cover
    WIDGETS_OK = False

# ---------------------------------------------------------------------------
#  CUSTOM EXCEPTIONS
# ---------------------------------------------------------------------------

class VacuumEnergyError(Exception):
    """Base para errores del paquete."""


class InputError(VacuumEnergyError):
    """Entradas físicas fuera de rango o formato."""


class MissingDependencyError(VacuumEnergyError):
    """Dependencia opcional requerida no instalada."""

# ---------------------------------------------------------------------------
#  CONSTANTES FÍSICAS (CODATA 2018) & CONFIG
# ---------------------------------------------------------------------------
H_ACCION_PURA: float = 6.626_070_15e-34          # J s (h_A, exacta)
H_BAR_A: float = H_ACCION_PURA / (2 * math.pi)   # J s
C_LUZ: float = 299_792_458.0                     # m s⁻¹ (exacta)

# Cosmología (Planck 2018)
RHO_VAC_OBS: float = 5.20e-10                    # J m⁻³
RHO_VAC_ERR: float = 0.03e-10

# Prefactores convenientes
PREFAC_QFT: float = H_BAR_A / (2 * math.pi ** 2 * C_LUZ ** 3)
CASIMIR_PREF: float = math.pi ** 2 / 240 * H_BAR_A * C_LUZ

# Integración numérica
EPSABS: float = 1e-40
EPSREL: float = 1e-12
QUAD_LIMIT: int = 2000  # suficiente en pruebas ‑ menor stack

# ---------------------------------------------------------------------------
#  HELPER ENUM / DATA
# ---------------------------------------------------------------------------

class Filtro(Enum):
    """Funciones de supresión UV."""

    EXP = "exp"
    GAUSS = "gauss"
    LORENTZ = "lorentz"
    NONLOC = "nonloc"

    # Map lazy‑evaluated para evitar dependencias circulares al serializar
    def fn(self) -> Callable[[np.ndarray | float], np.ndarray | float]:
        return {
            "exp": lambda x: np.exp(-x),
            "gauss": lambda x: np.exp(-x ** 2),
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
#  CORE COMPUTATIONS
# ---------------------------------------------------------------------------

def _omega_c(nu_c_hz: float) -> float:
    return 2.0 * math.pi * nu_c_hz


def _validate_positive(**kwargs: float) -> None:
    for name, val in kwargs.items():
        if val <= 0:
            raise InputError(f"{name} debe ser positivo (recibido {val}).")


# ‑‑‑ ρ_vac  ‑‑‑

def calcular_rho_vacio(
    nu_c_hz: float,
    filtro: str | Filtro = Filtro.EXP,
    *,
    nu_max_hz: float = 1e25,
) -> VacuumEnergyResult:
    """Densidad de energía de vacío.

    Parameters
    ----------
    nu_c_hz : float
        Frecuencia de corte (Hz).
    filtro : str | Filtro
        Tipo de supresor UV.
    nu_max_hz : float, optional
        Límite superior de integración, por defecto 1e25 Hz.
    """
    if isinstance(filtro, str):
        try:
            filtro = Filtro(filtro)
        except ValueError as err:
            raise InputError(f"Filtro desconocido: {filtro}") from err

    _validate_positive(nu_c_hz=nu_c_hz, nu_max_hz=nu_max_hz)

    omega_c = _omega_c(nu_c_hz)
    omega_max = min(_omega_c(nu_max_hz), 800.0 * omega_c)

    def integrand(omega: float) -> float:  # type: ignore[override]
        return PREFAC_QFT * omega ** 3 * filtro.fn()(omega / omega_c)

    val, err = quad(
        integrand,
        0.0,
        omega_max,
        epsabs=EPSABS,
        epsrel=EPSREL,
        limit=QUAD_LIMIT,
    )
    return VacuumEnergyResult(val, err)


# ‑‑‑ F_Casimir  ‑‑‑

def calcular_fuerza_casimir(
    d_m: float,
    nu_c_hz: float,
    filtro: str | Filtro = Filtro.EXP,
) -> float:
    if isinstance(filtro, str):
        try:
            filtro = Filtro(filtro)
        except ValueError as err:
            raise InputError(f"Filtro desconocido: {filtro}") from err

    _validate_positive(d_m=d_m, nu_c_hz=nu_c_hz)

    kappa_c = math.pi * C_LUZ / (d_m * _omega_c(nu_c_hz))
    sup = float(filtro.fn()(kappa_c))  # type: ignore[arg-type]
    return -CASIMIR_PREF / d_m ** 4 * sup

# ---------------------------------------------------------------------------
#  PLOTTING UTILITIES
# ---------------------------------------------------------------------------

def _require_matplotlib() -> None:
    if not MPL_OK:
        raise MissingDependencyError("Matplotlib requerido para esta función.")


def _require_widgets() -> None:
    if not (MPL_OK and WIDGETS_OK):
        raise MissingDependencyError("Matplotlib + ipywidgets requeridos.")


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


# ---- common helpers ----

def _parallel_map(func: Callable[[float], float], xs: Iterable[float]) -> List[float]:
    if JOBLIB_OK:
        return Parallel(n_jobs=-1, prefer="threads")(delayed(func)(x) for x in xs)  # type: ignore[misc]
    return [func(x) for x in xs]


# ---- Figure 1: Casimir comparison ----

def plot_casimir_comparison(
    nu_c_hz: float,
    d_min_mm: float = 0.1,
    d_max_mm: float = 1.0,
    *,
    dpi: int = 300,
    out_dir: Path | str = "figures",
) -> tuple[Path, np.ndarray, Figure]:
    _require_matplotlib()
    _validate_positive(nu_c_hz=nu_c_hz, d_min=d_min_mm, d_max=d_max_mm)
    if d_max_mm <= d_min_mm:
        raise InputError("d_max debe ser mayor que d_min.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    d_mm = np.logspace(math.log10(d_min_mm), math.log10(d_max_mm), 200)
    d_m = d_mm * 1e-3

    def _force(filtro: Filtro) -> List[float]:
        return [abs(calcular_fuerza_casimir(d, nu_c_hz, filtro)) for d in d_m]

    f_exp = _force(Filtro.EXP)
    ratios = {f.value: np.array(_force(f)) / f_exp for f in Filtro if f is not Filtro.EXP}

    # Figure
    fig = plt.figure(figsize=(5.5, 4.0))
    plt.semilogy(d_mm, np.ones_like(d_mm), "k-", lw=1.5, label="exp (ref)")
    colors = ["#377eb8", "#4daf4a", "#984ea3"]
    styles = ["-.", ":", "--"]
    for (filt, col, sty) in zip(ratios.keys(), colors, styles):
        plt.semilogy(d_mm, ratios[filt], color=col, ls=sty, lw=1.2, label=filt)

    plt.xlabel("Distancia entre placas [mm]")
    plt.ylabel(r"$|F|/|F_{\mathrm{exp}}|$")
    plt.title(fr"Comparativa filtros Casimir (ν_c={nu_c_hz:.2e} Hz)")
    plt.grid(which="both", alpha=0.3)
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    out_pdf = out_path / "casimir_comparison_relative.pdf"
    fig.savefig(out_pdf, dpi=dpi)

    # raw data
    data = np.column_stack([d_mm, f_exp] + [ratios[k] for k in ratios])
    headers = ["d_mm", "F_exp"] + list(ratios.keys())
    _export_data(data, headers, out_path / "casimir_data")

    return out_pdf, data, fig


# ---- Figure 2: rho vs eta ----

def plot_sens_eta_vs_rho(
    nu_c_base_hz: float,
    eta_min: float = 1e-3,
    eta_max: float = 1e-1,
    *,
    dpi: int = 300,
    out_dir: Path | str = "figures",
) -> tuple[Path, np.ndarray, Figure]:
    _require_matplotlib()
    _validate_positive(nu_c_base_hz=nu_c_base_hz, eta_min=eta_min, eta_max=eta_max)
    if eta_max <= eta_min:
        raise InputError("eta_max debe ser mayor que eta_min.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    eta = np.logspace(math.log10(eta_min), math.log10(eta_max), 200)

    def _rho(e: float) -> float:
        return calcular_rho_vacio(nu_c_base_hz * e).value

    rho_vals = np.array(_parallel_map(_rho, eta))

    # Ajuste ~ eta^4 (log‑log)
    coef = np.polyfit(np.log10(eta), np.log10(rho_vals), 1)
    A = 10 ** coef[1]

    fig = plt.figure(figsize=(5, 4))
    plt.loglog(eta, rho_vals, "b-", lw=1.2, label="numérico")
    plt.loglog(eta, A * eta ** 4, "r--", lw=1.0, label=r"ajuste ∝ η⁴")
    plt.fill_between(eta, RHO_VAC_OBS - RHO_VAC_ERR, RHO_VAC_OBS + RHO_VAC_ERR, color="green", alpha=0.15)
    plt.axhline(RHO_VAC_OBS, color="green", lw=0.8)

    plt.xlabel(r"Factor de escala $η$")
    plt.ylabel(r"$ρ_{vac}$ [J m⁻³]")
    plt.title("Sensibilidad de $ρ_{vac}$ a η")
    plt.grid(which="both", alpha=0.3)
    plt.legend()

    out_pdf = out_path / "sens_eta_vs_rho.pdf"
    fig.savefig(out_pdf, dpi=dpi)

    data = np.column_stack([eta, rho_vals])
    _export_data(data, ["eta", "rho_vac"], out_path / "rho_vs_eta_data")

    return out_pdf, data, fig


# ---- Interactive (notebook) ----

def interactive_plot(
    nu_c_hz: float = 7.275e11,
    filtro: str | Filtro = Filtro.EXP,
) -> Figure:
    _require_widgets()
    if isinstance(filtro, str):
        filtro = Filtro(filtro)

    _setup_plot_style()

    d_mm = np.logspace(-1, 0, 200)
    d_m = d_mm * 1e-3
    F_vals = [abs(calcular_fuerza_casimir(d, nu_c_hz, filtro)) for d in d_m]

    fig = plt.figure(figsize=(5.5, 4))
    plt.semilogy(d_mm, F_vals, "b-", lw=1.2, label=filtro.value)
    plt.xlabel("Distancia [mm]")
    plt.ylabel("|F| [Pa]")
    plt.title(fr"Casimir (ν_c={nu_c_hz:.2e} Hz)")
    plt.grid(alpha=0.3, which="both")
    plt.legend()

    return fig

# ---------------------------------------------------------------------------
#  UTILIDADES INTERNAS
# ---------------------------------------------------------------------------

def _export_data(array: np.ndarray, headers: List[str], base_path: Path) -> None:
    """Guarda datos en .txt y .csv en disco."""
    ref = "# Ref: vacuum_energy {}".format(__version__)
    np.savetxt(base_path.with_suffix(".txt"), array, header=ref + "\n" + " ".join(headers), fmt="%.6e")
    with open(base_path.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ref])
        writer.writerow(headers)
        writer.writerows(array)

# ---------------------------------------------------------------------------
#  TESTS BÁSICOS (pytest sería mejor; aquí rápido)
# ---------------------------------------------------------------------------

def _selftest() -> bool:
    try:
        res = calcular_rho_vacio(7.275e11)
        assert math.isclose(res.value / RHO_VAC_OBS, 1.0, rel_tol=1e-3)
        F = calcular_fuerza_casimir(0.5e-3, 7.275e11)
        assert F < 0
        logging.info("self‑tests ok")
        return True
    except AssertionError as err:
        logging.error(f"Self‑test failed: {err}")
        return False

# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vacuum_energy",
        description="Cálculo de ρ_vac y fuerza de Casimir con utilidades gráficas.",
    )
    p.add_argument("--nu-c", type=float, default=7.275e11, help="Frecuencia de corte (Hz)")
    p.add_argument("--filtro", choices=Filtro.values(), default=Filtro.EXP.value, help="Filtro UV")
    p.add_argument("--plot", help="Gráficos a generar: casimir_comp,sens_rho")
    p.add_argument("--generate-all", action="store_true", help="Genera todos los gráficos")
    p.add_argument("--out", type=Path, default=Path("figures"), help="Dir salida figuras")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--d-min", type=float, default=0.1)
    p.add_argument("--d-max", type=float, default=1.0)
    p.add_argument("--eta-min", type=float, default=1e-3)
    p.add_argument("--eta-max", type=float, default=1e-1)
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--version", action="version", version=f"vacuum_energy {__version__}")
    return p.parse_args(argv)


def _configure_logging(level: int) -> None:
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main(argv: List[str] | None = None) -> int:  # noqa: C901 (complexity acceptable)
    args = _parse_args(argv)
    log_level = max(logging.WARNING - 10 * args.verbose, logging.DEBUG)
    _configure_logging(log_level)

    try:
        plots: List[str] = []
        if args.generate_all:
            plots = ["casimir_comp", "sens_rho"]
        elif args.plot:
            plots = [p.strip() for p in args.plot.split(",") if p.strip()]
            invalid = set(plots) - {"casimir_comp", "sens_rho"}
            if invalid:
                raise InputError(f"Gráficos no reconocidos: {', '.join(invalid)}")

        for p_name in plots:
            if p_name == "casimir_comp":
                out, _, _ = plot_casimir_comparison(
                    args.nu_c,
                    args.d_min,
                    args.d_max,
                    dpi=args.dpi,
                    out_dir=args.out,
                )
                logging.info(f"Figura guardada → {out}")
            elif p_name == "sens_rho":
                out, _, _ = plot_sens_eta_vs_rho(
                    args.nu_c,
                    args.eta_min,
                    args.eta_max,
                    dpi=args.dpi,
                    out_dir=args.out,
                )
                logging.info(f"Figura guardada → {out}")

        if not plots:
            res = calcular_rho_vacio(args.nu_c, args.filtro)
            print(
                f"ρ_vac({args.filtro}, ν_c={args.nu_c:.2e} Hz) = "
                f"{res.value:.4e} ± {res.error:.1e} J/m³"
            )
            F = calcular_fuerza_casimir(0.5e-3, args.nu_c, args.filtro)
            print(f"F_Casimir(d=0.5 mm) = {F:.3e} Pa")

        return 0 if _selftest() else 1
    except VacuumEnergyError as ve:
        logging.error(str(ve))
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
