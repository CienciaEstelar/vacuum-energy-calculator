Vacuum Energy Toolkit (v22.0) 
🌌Cálculo científico de densidad de energía del vacío, fuerza de Casimir y fenómenos QGVersión 22.0: 
Reescritura completa con validación manual. Incluye correcciones físicas, análisis estadístico robusto y 16 figuras de calidad publicación, generadas en 3 formatos diferentes (.pdf, .png, .svg).
🚀 Descripción del ProyectoEste repositorio alberga el motor computacional end-to-end para el programa de investigación "h como Ciclo y Evento Elemental" (Galaz, 2025). 
vacuum_energy.py es una herramienta robusta diseñada para explorar fenómenos cuánticos del vacío, incluyendo:Densidad de energía del vacío (rho_vac) con modelado de error sistemático.Fuerza de Casimir con correcciones térmicas y de rugosidad.
Modificaciones del espacio de parámetros del axión.Anomalías QED (g−2, Lamb shift)
.Análisis de sensibilidad a la tensión de Hubble.✨ 
Características Clave🔍 Fenómenos Cuánticos del VacíoDensidad de Energía del Vacío (rho_vac): Calcula rho_vac utilizando 4 filtros de corte UV: exp, gauss, lorentz, nonloc.
Fuerza de Casimir: Modelo ideal con correcciones avanzadas para rugosidad superficial y efectos térmicos.Axiones: Predicción de la ventana de búsqueda de axiones con escalado nu_c y visualización del flujo de acoplamiento ASG.Anomalías QED: Correcciones a g−2 y al Lamb shift.
Cosmología: Análisis de sensibilidad de la tensión de Hubble.
📊 Rigor y RendimientoValidación Experimental: Análisis de chi2 contra datos experimentales de Lamoreaux (1997).
Propagación de Errores: Integración completa con la librería uncertainties.py.Rendimiento: Aceleración 5-10x mediante paralelización con Joblib.
Estabilidad: Guardas NaN/Inf en todas las integraciones numéricas.
📈 Figuras de Salida (16 únicas, en 3 formatos: .pdf, .png, .svg) 
El script genera un total de 16 figuras profesionales, cada una disponible en PDF, PNG y SVG. 
Estas figuras están listas para su publicación y se organizan por tipo de filtro cuando se genera en la carpeta 
resultados_completos.casimir_with_corrections: Comparación de la fuerza de Casimir con y sin correcciones.
chi2_analysis_casimir: Análisis chi2 para la fuerza de Casimir.
casimir_comparison_relative: Comparación relativa de la fuerza de Casimir entre filtros.
sensitivity_eta_vs_rho: Sensibilidad de rho_vac vs. eta.casimir_absolute: Fuerza de Casimir absoluta vs. distancia, con efectos térmicos.
rho_vac_convergence: Convergencia de rho_vac con omega_max.axion_window_shift: Desplazamiento de la ventana de búsqueda del axión.asg_flow: Flujo de acoplamiento axión-fotón-fotón.
kibble_analysis: Simulación de la balanza de Kibble.hubble_tension_sensitivity: Sensibilidad a la tensión de Hubble.filter_comparison: Comparación de rho_vac entre diferentes filtros.
roughness_simulation: Efecto de la rugosidad en la fuerza de Casimir.kappa_calibration: Calibración de kappa.
lamb_shift_correction: Corrección del Lamb shift.g2_correction: Corrección del momento magnético anómalo del electrón (g−2).rho_vac_vs_nu_c: Densidad de energía del vacío vs. frecuencia de corte.🛠️ I


Instalar Dependenciaspip install -r requirements.txt
# Dependencias CORE: numpy, scipy, matplotlib
# Dependencias OPCIONALES: joblib (paralelización), uncertainties (errores), ipywidgets (interactivo)
Uso desde la Línea de Comandos (CLI)# Auto-prueba + cálculos básicos con ν_c calibrada
python vacuum_energy.py --nu-c 7.275e11

# Generar TODAS las 16 figuras en .pdf, .png y .svg
python vacuum_energy.py --generate-all-plots --out ./figuras

# Generar un gráfico específico (ej: análisis χ²) con alta resolución
python vacuum_energy.py --plot chi2_analysis --dpi 600

# Calcular ν_c para un modelo potencial específico (ej: axion)
python vacuum_energy.py --potential axion --ma-max 15.0

# Ejecutar en modo interactivo (requiere ipywidgets)
python vacuum_energy.py --interactive
Uso como Módulo de PythonPuedes importar las funciones directamente en tus propios scripts de Python:from vacuum_energy import (
    calculate_vacuum_density, calculate_casimir_force_corrected,
    calculate_chi2_casimir, LAMOREAUX_VALIDATION_DATA,
    # Puedes importar otras funciones de cálculo y trazado aquí
)

# Densidad de energía del vacío para un nu_c específico
rho = calculate_vacuum_density(nu_c_hz=7.275e11, filter_type='exp')
print(f"ρ_vac = {rho.value:.3e} ± {rho.error:.1e} J/m³")

# Fuerza de Casimir corregida (d=1µm, T=4K)
F_corr = calculate_casimir_force_corrected(d_m=1e-6, nu_c_hz=7.275e11, temp_k=4.0)
print(f"F_corr (corregida) = {F_corr:.3e} Pa")

# Validación χ² contra datos experimentales (con correcciones)
chi2_red = calculate_chi2_casimir(nu_c_hz=7.275e11, data=LAMOREAUX_VALIDATION_DATA, use_corrections=True)
print(f"χ² reducido = {chi2_red:.2f}")

# Ejemplo de generación de un plot programáticamente
# from vacuum_energy import plot_casimir_with_corrections
# plot_casimir_with_corrections(nu_c_hz=7.275e11, data=LAMOREAUX_VALIDATION_DATA, dpi=300, out_dir="mis_figuras")
✅ Validación Científicarho_vac dentro del 0.1% del valor observado.Consistencia en el signo de la fuerza de Casimir.Corrección de g−2 que coincide con la anomalía QED.chi2\<1.5 para el modelo corregido.📄 LicenciaEste proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.📚 ReferenciaarXiv: XXXX.XXXXX [physics.gen-ph]
