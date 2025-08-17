# Vacuum Energy Toolkit (v22.0) 🌌  
*Cálculo científico de densidad de energía del vacío, fuerza de Casimir y fenómenos QG*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)  
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

> **Versión 22.0**: Reescritura completa con validación manual. Incluye **correcciones físicas**, **análisis estadístico robusto** y **13 figuras de calidad publicación**.



🔍 Características Clave

    ρ_vac(ν_c): Densidad de energía del vacío con 4 modelos de cutoff (exp, gauss, lorentz, nonloc).

    Fuerza de Casimir: Modelo ideal + correcciones (rugosidad, temperatura).

    Validación experimental: χ² contra datos de Lamoreaux (1997).

    Conexiones QED: Cálculo de correcciones a g-2 y Lamb shift.

    Visualización: 13 gráficos científicos (PDF/PNG 300-600 DPI).

    Paralelización: Acelerado con Joblib para multi-core.


---

## 🚀 Instalación  
```bash  
git clone https://github.com/tu-usuario/vacuum-energy.git  
cd vacuum-energy  
pip install -r requirements.txt  # numpy, scipy, matplotlib, joblib  



# Auto-prueba + cálculos básicos  
python vacuum_energy.py --nu-c 7.275e11  

# Generar TODAS las figuras  
python vacuum_energy.py --generate-all-plots --out ./figuras  

# Gráfico específico (ej: análisis χ²)  
python vacuum_energy.py --plot chi2_analysis --dpi 600

from vacuum_energy import (  
    calculate_vacuum_density, calculate_casimir_force_corrected,  
    calculate_chi2_casimir, LAMOREAUX_VALIDATION_DATA  
)  

# Densidad de energía del vacío  
ρ = calculate_vacuum_density(nu_c_hz=7.275e11)  
print(f"ρ_vac = {ρ.value:.3e} ± {ρ.error:.1e} J/m³")  

# Fuerza de Casimir corregida (d=1µm, T=4K)  
F = calculate_casimir_force_corrected(d_m=1e-6, nu_c_hz=7.275e11)  
print(f"F_corr = {F:.3e} Pa")  

# Validación χ² contra datos experimentales  
χ² = calculate_chi2_casimir(nu_c_hz=7.275e11, data=LAMOREAUX_VALIDATION_DATA, use_corrections=True)  
print(f"χ² reducido = {χ²:.2f}")  
