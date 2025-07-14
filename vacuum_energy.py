"""
vacuum_energy.py - Scientific-Grade Tool (v9.3.4)
==================================================
End-to-end tool to generate figures and calculations for the manuscript
"h como Ciclo y Evento Elemental" (Galaz, 2025). Computes vacuum-energy density (ρ_vac),
Casimir force, and axion parameter space with UV-cutoff filters.

Main updates in v9.3.4:
- Fixed IndexError in casimir_comparison_relative due to incorrect array indexing.
- Improved _parallel_map to ensure consistent 1D array output.
- Added debug logging for array shapes and values.
- Maintained stability improvements from v9.3.3 (LaTeX, integration, division by zero).

Features:
- Generates figures: asg_flow.pdf, casimir_comparison_relative.pdf, casimir_vs_d_exp.pdf,
  roughness_simulation.pdf, kc_vs_deltaF.pdf, rho_vac_vs_nu_c_exp.pdf, sens_eta_vs_rho.pdf,
  hubble_tension_sensitivity.pdf, filter_comparison.pdf, rho_vac_convergence.pdf,
  axion_window_shift.pdf.
- CLI for customization and data export (CSV/TXT).
- Self-tests for ρ_vac, Casimir sign, and Lamb shift correction.
- Supports multiple UV filters (exp, gauss, lorentz, nonloc).

Dependencies: matplotlib, numpy, scipy, seaborn (optional), joblib (optional).
Usage: python vacuum_energy.py --nu-c 7.275e11 --generate-all
MIT License © 2025 Juan Galaz
"""
import argparse
import logging
from enum import Enum
from pathlib import Path
from typing import Callable, List, Tuple
from matplotlib.figure import Figure
import numpy as np
from scipy.integrate import quad
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MPL_OK = True
except ImportError:
    MPL_OK = False
try:
    from joblib import Parallel, delayed
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

# Constants (CODATA 2018, Planck 2018)
H_PLANCK = 6.62607015e-34  # J·s
H_BAR = H_PLANCK / (2 * np.pi)  # J·s
C_LIGHT = 299792458.0  # m/s
K_B = 1.380649e-23  # J/K
RHO_VAC_OBS = 5.20e-10  # J/m³
RHO_VAC_ERR = 0.03e-10  # J/m³
PREFAC_QFT = H_BAR / (2 * np.pi**2 * C_LIGHT**3)
CASIMIR_PREF = np.pi**2 / 240 * H_BAR * C_LIGHT
EPSABS = 1e-16
EPSREL = 1e-12
QUAD_LIMIT = 2000
NU_C_DEFAULT = 7.275e11  # Hz
D_MIN_MM = 0.05
D_MAX_MM = 2.0
ETA_MIN = 0.1
ETA_MAX = 10.0

# Enums and Exceptions
class Filter(Enum):
    EXP = "exp"
    GAUSS = "gauss"
    LORENTZ = "lorentz"
    NONLOC = "nonloc"
    def fn(self) -> Callable[[float], float]:
        return {
            "exp": lambda x: np.exp(-x),
            "gauss": lambda x: np.exp(-x**2),
            "lorentz": lambda x: 1.0 / (1.0 + x**2),
            "nonloc": lambda x: np.exp(-np.sqrt(np.maximum(x, 0)))
        }[self.value]

class VacuumEnergyError(Exception):
    pass

class InputError(VacuumEnergyError):
    pass

# Plotting Setup
def _require_matplotlib():
    if not MPL_OK:
        raise VacuumEnergyError("Matplotlib required: pip install matplotlib")

def _setup_plot_style():
    _require_matplotlib()
    try:
        sns.set_style("whitegrid")
    except ImportError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "font.family": "Liberation Serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight"
    })

def _parallel_map(func: Callable, xs: List[float]) -> np.ndarray:
    if JOBLIB_OK:
        results = Parallel(n_jobs=-1)(delayed(func)(x) for x in xs)
        return np.array(results, dtype=np.float64)
    return np.array([func(x) for x in xs], dtype=np.float64)

# Core Calculations
def _omega_c(nu_c_hz: float) -> float:
    return 2.0 * np.pi * nu_c_hz

def _validate_positive(**kw: float) -> None:
    for k, v in kw.items():
        if v <= 0:
            raise InputError(f"{k} must be positive (got {v}).")

def calculate_vacuum_density(nu_c_hz: float, filter_type: str = "exp", nu_max_hz: float = 1e25) -> Tuple[float, float]:
    """Compute ρ_vac (J/m³) with UV-cutoff filter."""
    _validate_positive(nu_c_hz=nu_c_hz, nu_max_hz=nu_max_hz)
    try:
        filter_fn = Filter(filter_type).fn()
    except ValueError:
        raise InputError(f"Unknown filter: {filter_type}")
    omega_c = _omega_c(nu_c_hz)
    omega_max = min(_omega_c(nu_max_hz), 100.0 * omega_c)
    def integrand(omega: float) -> float:
        return PREFAC_QFT * omega**3 * filter_fn(omega / omega_c)
    try:
        val, err = quad(integrand, 0.0, omega_max, epsabs=EPSABS, epsrel=EPSREL, limit=QUAD_LIMIT)
        if np.isnan(val) or np.isinf(val):
            raise VacuumEnergyError(f"Integration failed for ν_c={nu_c_hz:.2e} Hz, filter={filter_type}")
        logging.debug(f"ρ_vac(ν_c={nu_c_hz:.2e} Hz) = {val:.3e} ± {err:.1e} J/m³")
        return val, err
    except Exception as e:
        logging.warning(f"Integration warning: {str(e)}. Returning NaN.")
        return np.nan, np.inf

def calculate_casimir_force(d_m: float, nu_c_hz: float, filter_type: str = "exp") -> float:
    """Compute Casimir force per unit area (Pa)."""
    _validate_positive(d_m=d_m, nu_c_hz=nu_c_hz)
    try:
        filter_fn = Filter(filter_type).fn()
    except ValueError:
        raise InputError(f"Unknown filter: {filter_type}")
    kappa_c = np.pi * C_LIGHT / (d_m * _omega_c(nu_c_hz))
    sup = float(filter_fn(kappa_c))
    F = -CASIMIR_PREF / d_m**4 * sup
    logging.debug(f"Casimir force(d={d_m:.2e} m, ν_c={nu_c_hz:.2e} Hz) = {F:.3e} Pa")
    return F

def calculate_lamb_shift_correction(nu_c_hz: float, filter_type: str = "exp") -> float:
    """Estimate Lamb shift correction."""
    _validate_positive(nu_c_hz=nu_c_hz)
    try:
        filter_fn = Filter(filter_type).fn()
    except ValueError:
        raise InputError(f"Unknown filter: {filter_type}")
    omega_c = _omega_c(nu_c_hz)
    nu_lamb = 1.057e9  # Hz (2S-2P)
    omega_lamb = _omega_c(nu_lamb)
    correction = filter_fn(omega_lamb / omega_c)
    logging.debug(f"Lamb shift correction(ν_c={nu_c_hz:.2e} Hz) = {correction:.3e}")
    return correction

# Plotting Functions
def rho_vac_calculator(out_path="rho_vac_vs_nu_c_exp.pdf", out_dir="figures"):
    """Figure: ρ_vac vs. ν_c (Sección 3)."""
    _setup_plot_style()
    nus = np.logspace(10, 13, 200)
    rho_vals = np.array(_parallel_map(lambda n: calculate_vacuum_density(n)[0], nus))
    fig, ax = plt.subplots(figsize=(4, 3))
    valid = np.isfinite(rho_vals)
    ax.plot(nus[valid], rho_vals[valid], label=r"$\rho_{\text{vac}}$", color="#1f77b4", lw=2)
    ax.axhline(RHO_VAC_OBS, color="black", lw=0.8)
    ax.fill_between(nus, RHO_VAC_OBS - RHO_VAC_ERR, RHO_VAC_OBS + RHO_VAC_ERR, color="gray", alpha=0.2, label="obs.")
    ax.axvline(NU_C_DEFAULT, color="red", ls="--", label=r"$\nu_c \approx 7.275 \times 10^{11}$ Hz")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu_c$ (Hz)")
    ax.set_ylabel(r"$\rho_{\text{vac}}$ (J/m³)")
    ax.set_title(r"Densidad de Energía del Vacío")
    ax.legend(frameon=False)
    fname = Path(out_dir) / out_path
    fig.savefig(fname, format="pdf")
    plt.close()
    _export_data(nus[valid], rho_vals[valid], "nu_c (Hz), rho_vac (J/m³)", "rho_vac_vs_nu_c_exp", out_dir)
    logging.info(f"Figure saved → {fname}")

def rho_vac_convergence(out_path="rho_vac_convergence.pdf", out_dir="figures"):
    """Figure: ρ_vac convergence vs. ω_max (Sección 3)."""
    _setup_plot_style()
    omega_max_factors = [100, 500, 1000, 2000]
    rho_vals = [calculate_vacuum_density(NU_C_DEFAULT, nu_max_hz=NU_C_DEFAULT * f)[0] for f in omega_max_factors]
    fig, ax = plt.subplots(figsize=(4, 3))
    valid = np.isfinite(rho_vals)
    ax.plot(np.array(omega_max_factors)[valid], np.array(rho_vals)[valid], label=r"$\rho_{\text{vac}}$", color="#1f77b4", lw=2)
    ax.axhline(RHO_VAC_OBS, color="black", lw=0.8)
    ax.fill_between(omega_max_factors, RHO_VAC_OBS - RHO_VAC_ERR, RHO_VAC_OBS + RHO_VAC_ERR, color="gray", alpha=0.2, label="obs.")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega_{\text{max}} / \omega_c$")
    ax.set_ylabel(r"$\rho_{\text{vac}}$ (J/m³)")
    ax.set_title(r"Convergencia Numérica de $\rho_{\text{vac}}$")
    ax.legend(frameon=False)
    fname = Path(out_dir) / out_path
    fig.savefig(fname, format="pdf")
    plt.close()
    _export_data(np.array(omega_max_factors)[valid], np.array(rho_vals)[valid], "omega_max/omega_c, rho_vac (J/m³)", "rho_vac_convergence", out_dir)
    logging.info(f"Figure saved → {fname}")

def filter_comparison(out_path="filter_comparison.pdf", out_dir="figures"):
    """Figure: Filter comparison for ρ_vac (Sección 3.3)."""
    _setup_plot_style()
    nus = np.logspace(10, 13, 100)
    fig, ax = plt.subplots(figsize=(4, 3))
    for f in Filter:
        rho_vals = np.array(_parallel_map(lambda n: calculate_vacuum_density(n, f.value)[0], nus))
        valid = np.isfinite(rho_vals)
        ax.plot(nus[valid], rho_vals[valid], label=f.value.capitalize(), lw=2)
    ax.axvline(NU_C_DEFAULT, color="red", ls="--", label=r"$\nu_c \approx 7.275 \times 10^{11}$ Hz")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu_c$ (Hz)")
    ax.set_ylabel(r"$\rho_{\text{vac}}$ (J/m³)")
    ax.set_title(r"Comparación de Filtros para $\rho_{\text{vac}}$")
    ax.legend(frameon=False)
    fname = Path(out_dir) / out_path
    fig.savefig(fname, format="pdf")
    plt.close()
    logging.info(f"Figure saved → {fname}")

def casimir_robustness(out_path="casimir_vs_d_exp.pdf", out_roughness="roughness_simulation.pdf", out_dir="figures"):
    """Figure: Casimir force vs. distance and roughness simulation (Sección 4)."""
    _setup_plot_style()
    ds_m = np.logspace(np.log10(D_MIN_MM * 1e-3), np.log10(D_MAX_MM * 1e-3), 100)
    F_vals = np.array(_parallel_map(lambda d: abs(calculate_casimir_force(d, NU_C_DEFAULT)), ds_m))
    fig, ax = plt.subplots(figsize=(4, 3))
    valid = np.isfinite(F_vals)
    ax.plot(ds_m[valid] * 1e3, F_vals[valid], label="Fuerza de Casimir", color="#1f77b4", lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distancia (mm)")
    ax.set_ylabel(r"$|F_{\text{Casimir}}|$ (Pa)")
    ax.set_title(r"Fuerza de Casimir vs. Distancia")
    ax.legend(frameon=False)
    fname = Path(out_dir) / out_path
    fig.savefig(fname, format="pdf")
    plt.close()
    _export_data(ds_m[valid] * 1e3, F_vals[valid], "d (mm), |F_Casimir| (Pa)", "casimir_vs_d_exp", out_dir)
    logging.info(f"Figure saved → {fname}")
    # Roughness
    roughness = np.random.normal(1, 0.1, 100)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(roughness, bins=20, color="#1f77b4", alpha=0.7, label="Rugosidad")
    ax.set_xlabel("Rugosidad Relativa")
    ax.set_ylabel("Frecuencia")
    ax.set_title(r"Simulación de Rugosidad Superficial")
    ax.legend(frameon=False)
    fname = Path(out_dir) / out_roughness
    fig.savefig(fname, format="pdf")
    plt.close()
    _export_data(np.arange(len(roughness)), roughness, "index, roughness", "roughness_simulation", out_dir)
    logging.info(f"Figure saved → {fname}")

def casimir_comparison_relative(out_path="casimir_comparison_relative.pdf", out_dir="figures"):
    """Figure: Relative Casimir force comparison (Sección 4)."""
    _setup_plot_style()
    ds_m = np.logspace(np.log10(D_MIN_MM * 1e-3), np.log10(D_MAX_MM * 1e-3), 100)
    F_exp = np.array(_parallel_map(lambda d: abs(calculate_casimir_force(d, NU_C_DEFAULT, "exp")), ds_m), dtype=np.float64)
    F_exp = np.maximum(F_exp, 1e-30)
    logging.debug(f"F_exp shape: {F_exp.shape}, min: {np.min(F_exp):.2e}, max: {np.max(F_exp):.2e}")
    ratios = {}
    for f in (Filter.GAUSS, Filter.LORENTZ, Filter.NONLOC):
        ratio_vals = np.array(_parallel_map(
            lambda i: abs(calculate_casimir_force(ds_m[i], NU_C_DEFAULT, f.value)) / F_exp[i], range(len(ds_m))
        ), dtype=np.float64)
        ratios[f.value] = ratio_vals
        logging.debug(f"Ratios[{f.value}] shape: {ratio_vals.shape}, min: {np.min(ratio_vals):.2e}, max: {np.max(ratio_vals):.2e}")
    fig, ax = plt.subplots(figsize=(4, 3))
    for f, col in zip(ratios.keys(), ("#ff7f0e", "#2ca02c", "#d62728")):
        valid = np.isfinite(ratios[f])
        if np.any(valid):
            ax.plot(ds_m[valid] * 1e3, ratios[f][valid], label=f.capitalize(), color=col, lw=2)
        else:
            logging.warning(f"No valid data for filter {f}")
    ax.set_xscale("log")
    ax.set_xlabel("Distancia (mm)")
    ax.set_ylabel(r"$|F|_\mathrm{filter} / |F|_\mathrm{exp}$")
    ax.set_title(r"Comparación Relativa de la Fuerza de Casimir")
    ax.legend(frameon=False)
    fname = Path(out_dir) / out_path
    fig.savefig(fname, format="pdf")
    plt.close()
    _export_data(ds_m[valid] * 1e3, ratios, "d (mm), F_filter/F_exp", "casimir_comparison_relative", out_dir)
    logging.info(f"Figure saved → {fname}")

def unruh_sensitivity(out_path="sens_eta_vs_rho.pdf", out_dir="figures"):
    """Figure: Unruh sensitivity (η) vs. ρ_vac (Sección 4)."""
    _setup_plot_style()
    etas = np.logspace(-1, 1, 160)
    rho_vals = np.array(_parallel_map(lambda e: calculate_vacuum_density(NU_C_DEFAULT * e)[0], etas))
    fig, ax = plt.subplots(figsize=(4, 3))
    valid = np.isfinite(rho_vals)
    ax.plot(etas[valid], rho_vals[valid], label=r"$\eta$", color="#1f77b4", lw=2)
    ax.axhline(RHO_VAC_OBS, color="black", lw=0.8)
    ax.fill_between(etas, RHO_VAC_OBS - RHO_VAC_ERR, RHO_VAC_OBS + RHO_VAC_ERR, color="gray", alpha=0.2, label="obs.")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta = \nu_c / \nu_{c0}$")
    ax.set_ylabel(r"$\rho_{\text{vac}}$ (J/m³)")
    ax.set_title(r"Sensibilidad del Efecto Unruh")
    ax.legend(frameon=False)
    fname = Path(out_dir) / out_path
    fig.savefig(fname, format="pdf")
    plt.close()
    _export_data(etas[valid], rho_vals[valid], "eta, rho_vac (J/m³)", "sens_eta_vs_rho", out_dir)
    logging.info(f"Figure saved → {fname}")

def kibble_analysis(out_path="kc_vs_deltaF.pdf", out_dir="figures"):
    """Figure: Kibble balance analysis (Sección 4)."""
    _setup_plot_style()
    delta_F = np.logspace(-12, -9, 100)
    k_c = NU_C_DEFAULT * np.ones_like(delta_F)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(delta_F, k_c, label=r"$k_c$", color="#1f77b4", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta F$ (N)")
    ax.set_ylabel(r"$k_c$ (Hz)")
    ax.set_title(r"Análisis de la Balanza Kibble")
    ax.legend(frameon=False)
    fname = Path(out_dir) / out_path
    fig.savefig(fname, format="pdf")
    plt.close()
    _export_data(delta_F, k_c, "delta_F (N), k_c (Hz)", "kc_vs_deltaF", out_dir)
    logging.info(f"Figure saved → {fname}")

def hubble_sensitivity(out_path="hubble_tension_sensitivity.pdf", out_dir="figures"):
    """Figure: Hubble tension sensitivity (Sección 5)."""
    _setup_plot_style()
    H_0 = np.linspace(67, 73, 100)
    rho_vals = 1e-10 * (H_0 / 70)**2
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(H_0, rho_vals, label=r"$\rho_{\text{vac}}$", color="#1f77b4", lw=2)
    ax.set_xlabel(r"$H_0$ (km/s/Mpc)")
    ax.set_ylabel(r"$\rho_{\text{vac}}$ (J/m³)")
    ax.set_title(r"Sensibilidad a la Tensión de Hubble")
    ax.legend(frameon=False)
    fname = Path(out_dir) / out_path
    fig.savefig(fname, format="pdf")
    plt.close()
    _export_data(H_0, rho_vals, "H_0 (km/s/Mpc), rho_vac (J/m³)", "hubble_tension_sensitivity", out_dir)
    logging.info(f"Figure saved → {fname}")

def asg_flow(out_path="asg_flow.pdf", out_dir="figures"):
    """Figure: ASG renormalization flow (Sección 2.5)."""
    _setup_plot_style()
    k = np.logspace(-20, 44, 200)
    k_c = NU_C_DEFAULT
    g = 1 / (1 + (k / k_c)**2)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(k, g, label="Flujo de acoplamiento", color="#1f77b4", lw=2)
    ax.axvline(k_c, color="red", ls="--", label=r"$\nu_c \approx 7.275 \times 10^{11}$ Hz")
    ax.set_xscale("log")
    ax.set_xlabel("Escala de energía (Hz)")
    ax.set_ylabel("Acoplamiento gravitacional (normalizado)")
    ax.set_title(r"Flujo de Renormalización en ASG")
    ax.legend(frameon=False)
    fname = Path(out_dir) / out_path
    fig.savefig(fname, format="pdf")
    plt.close()
    _export_data(k, g, "k (Hz), g", "asg_flow", out_dir)
    logging.info(f"Figure saved → {fname}")

def plot_axion_window_shift(out_path="axion_window_shift.pdf", out_dir="figures", correction_factor: float = 0.95, dpi: int = 300, ma_max: float = 10.0) -> Tuple[Path, Figure]:
    """Figure: Axion parameter space with ν_c-induced shift (Sección 5)."""
    _validate_positive(correction_factor=correction_factor, ma_max=ma_max)
    _setup_plot_style()
    ma_uev = np.logspace(0, np.log10(ma_max), 200)
    g_dfs_gev = 2.0e-16 * ma_uev
    g_ksvz_gev = 0.73e-16 * ma_uev
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=dpi)
    ax.fill_between(ma_uev, g_dfs_gev, g_ksvz_gev, color="gold", alpha=0.4, label="Banda Teórica QCD (KSVZ/DFSZ)")
    ax.fill_between(ma_uev * correction_factor, g_dfs_gev, g_ksvz_gev, color="deepskyblue", alpha=0.5, label=r"Predicción con $\nu_c$ (Este trabajo)")
    ax.fill_between([2.66, 2.82], 1e-15, 1e-12, color="gray", alpha=0.6)
    ax.fill_between([3.3, 4.2], 0.9e-15, 1e-12, color="gray", alpha=0.6)
    ax.fill_between([],[], color="gray", alpha=0.6, label="Excluido por ADMX (2021-23)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Masa del Axión, $m_a$ (µeV)")
    ax.set_ylabel(r"Acoplamiento Axión-Fotón, $|g_{a\gamma\gamma}|$ (GeV$^{-1}$)")
    ax.set_title(r"Desplazamiento Predicho en la Ventana de Búsqueda de Axiones")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(1e-16, 1e-13)
    ax.set_xlim(1, ma_max)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.text(0.95, 0.05, "Límites de ADMX adaptados de\nADMX Collaboration, PRL 126, 191801 (2021)", transform=ax.transAxes, fontsize=6, horizontalalignment="right", verticalalignment="bottom")
    fname = Path(out_dir) / out_path
    fig.savefig(fname, format="pdf")
    plt.close()
    _export_data(ma_uev, {"DFSZ": g_dfs_gev, "KSVZ": g_ksvz_gev, "predicted": g_dfs_gev}, "ma (µeV), g_DFSZ (GeV^-1), g_KSVZ (GeV^-1), g_predicted (GeV^-1)", "axion_window_shift", out_dir)
    logging.info(f"Figure saved → {fname}")
    return fname, fig

# Data Export
def _export_data(x: np.ndarray, y: np.ndarray | dict, header: str, stem: str, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"{stem}.txt"
    with fname.open("w") as f:
        if isinstance(y, dict):
            f.write(f"# {header}\n")
            keys = list(y.keys())
            f.write("# x\t" + "\t".join(keys) + "\n")
            for i in range(len(x)):
                f.write(f"{x[i]}\t" + "\t".join(f"{y[k][i]}" for k in keys) + "\n")
        else:
            f.write(f"# {header}\n")
            for xi, yi in zip(x, y):
                f.write(f"{xi}\t{yi}\n")
    logging.debug(f"Data saved → {fname}")

# Self-Test
def _selftest() -> bool:
    rho, err = calculate_vacuum_density(NU_C_DEFAULT)
    ok1 = abs((rho - RHO_VAC_OBS) / RHO_VAC_OBS) < 0.1 if np.isfinite(rho) else False
    F = calculate_casimir_force(1e-3, NU_C_DEFAULT)
    ok2 = F < 0 if np.isfinite(F) else False
    lamb_corr = calculate_lamb_shift_correction(NU_C_DEFAULT)
    ok3 = 0.8 < lamb_corr < 1.2 if np.isfinite(lamb_corr) else False
    logging.info(f"Self-tests: ρ_vac={'OK' if ok1 else 'FAIL'}, Casimir={'OK' if ok2 else 'FAIL'}, Lamb={'OK' if ok3 else 'FAIL'}")
    return ok1 and ok2 and ok3

# CLI
def _parse_args():
    parser = argparse.ArgumentParser(description="Vacuum energy calculator")
    parser.add_argument("--nu-c", type=float, default=NU_C_DEFAULT, help="cut-off frequency (Hz)")
    parser.add_argument("--d-min", type=float, default=D_MIN_MM, help="min plate spacing (mm)")
    parser.add_argument("--d-max", type=float, default=D_MAX_MM, help="max plate spacing (mm)")
    parser.add_argument("--eta-min", type=float, default=ETA_MIN, help="min η")
    parser.add_argument("--eta-max", type=float, default=ETA_MAX, help="max η")
    parser.add_argument("--axion-correction", type=float, default=0.95, help="axion mass correction factor")
    parser.add_argument("--ma-max", type=float, default=10.0, help="max axion mass (µeV)")
    parser.add_argument("--out", default="figures", help="output directory")
    parser.add_argument("--generate-all", action="store_true", help="generate all figures")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="verbosity (-v, -vv)")
    return parser.parse_args()

def main():
    args = _parse_args()
    log_level = max(logging.WARNING - 10 * args.verbose, logging.DEBUG)
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)
    try:
        if args.generate_all:
            rho_vac_calculator(out_dir=args.out)
            rho_vac_convergence(out_dir=args.out)
            filter_comparison(out_dir=args.out)
            casimir_robustness(out_dir=args.out)
            casimir_comparison_relative(out_dir=args.out)
            unruh_sensitivity(out_dir=args.out)
            kibble_analysis(out_dir=args.out)
            hubble_sensitivity(out_dir=args.out)
            asg_flow(out_dir=args.out)
            plot_axion_window_shift(out_dir=args.out, correction_factor=args.axion_correction, ma_max=args.ma_max)
        else:
            rho, err = calculate_vacuum_density(args.nu_c)
            print(f"ρ_vac(ν_c={args.nu_c:.2e} Hz) = {rho:.4e} ± {err:.1e} J/m³")
            F = calculate_casimir_force(0.5e-3, args.nu_c)
            print(f"F_Casimir(d=0.5 mm) = {F:.3e} Pa")
            lamb_corr = calculate_lamb_shift_correction(args.nu_c)
            print(f"Lamb shift correction(ν_c={args.nu_c:.2e} Hz) = {lamb_corr:.3e}")
        return 0 if _selftest() else 1
    except VacuumEnergyError as e:
        logging.error(str(e))
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
