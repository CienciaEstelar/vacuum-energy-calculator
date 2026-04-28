"""
Example: how the regularized vacuum energy density depends on the choice
of UV cutoff frequency and filter shape.

This script demonstrates two pedagogical points:

1. For any smooth UV regulator with a single scale omega_c, dimensional
   analysis fixes rho_vac to scale as omega_c^4. The numerical prefactor
   depends on the filter shape (~6 for exponential, ~sqrt(pi)/2 for
   Gaussian, etc.).

2. Reproducing the observationally inferred vacuum energy density
   (rho_obs ~ 5.4e-10 J/m^3) by tuning omega_c is a dimensional
   consistency check, not a solution to the cosmological constant problem.
   The same dimensional logic underlies holographic dark energy (Cohen,
   Kaplan & Nelson, PRL 82, 4971, 1999), where the IR cutoff is set by
   the Hubble scale.

Run:
    python examples/vacuum_density_scan.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from vacuumpy import Filter, calculate_vacuum_density

# Observationally inferred vacuum energy density (Planck 2018 cosmology,
# rho_Lambda = Omega_Lambda * rho_crit ~ 5.4e-10 J/m^3).
RHO_OBS = 5.4e-10  # J/m^3


def main() -> None:
    nu_c_grid = np.logspace(10, 14, 200)  # 10 GHz to 100 THz

    fig, ax = plt.subplots(figsize=(7, 5))

    for filt, label in [
        (Filter.EXP, "exponential"),
        (Filter.GAUSS, "Gaussian"),
        (Filter.LORENTZ, "Lorentzian"),
    ]:
        rho = np.array(
            [calculate_vacuum_density(nu, filt).value for nu in nu_c_grid]
        )
        ax.loglog(nu_c_grid, rho, label=label)

    ax.axhline(
        RHO_OBS,
        color="k",
        linestyle="--",
        linewidth=1,
        label=r"$\rho_{\rm obs} \approx 5.4 \times 10^{-10}\ {\rm J/m^3}$",
    )

    ax.set_xlabel(r"Cutoff frequency $\nu_c$ [Hz]")
    ax.set_ylabel(r"Regularized vacuum energy density $\rho_{\rm vac}$ [J/m$^3$]")
    ax.set_title(r"Filter dependence of $\rho_{\rm vac}(\nu_c)$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("vacuum_density_scan.png", dpi=150)
    print("Saved vacuum_density_scan.png")


if __name__ == "__main__":
    main()
