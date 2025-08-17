vacuum_energy: Scientific Toolkit for Phenomenological QG Modeling (v22.0)High-precision, reproducible Python toolkit for computing vacuum energy density (rho_textvac), Casimir force, and exploring phenomenological quantum gravity models. This package is the computational engine behind the manuscript "h como ciclo y evento elemental: Reinterpretación conceptual de la constante de Planck" (arXiv: [tu-arxiv-id]).Version 22.0 is a major rewrite that addresses a rigorous peer-review process. It evolves beyond a simple calculator into a comprehensive scientific tool, incorporating advanced statistical analysis, physical corrections, and a full suite of publication-ready plotting functions.Novedades en la Versión 22.0Modelo Cuantitativo de Errores Sistemáticos: Implementa un modelo físico para las correcciones de la fuerza de Casimir debido a la rugosidad de la superficie y efectos térmicos.Análisis chi2 Robusto: Compara el modelo teórico (ideal y corregido) contra datos experimentales (Lamoreaux, 1997) usando un ajuste de amplitud para un análisis estadístico riguroso.Validación de la "Tensión de Casimir": Demuestra cuantitativamente cómo la inclusión de correcciones sistemáticas resuelve la tensión entre la nu_c cosmológica y los datos de laboratorio.Suite de Visualización Completa: Genera 13 figuras científicas clave, incluyendo análisis de sensibilidad, convergencia numérica y el desplazamiento predicho en la ventana de búsqueda de axiones.Conexiones con QED: Incluye modelos para calcular la corrección a g−2 y el desplazamiento de Lamb.InstalaciónClona el repositorio e instala las dependencias:git clone https://github.com/[tu-usuario]/vacuum_energy.git
cd vacuum_energy
pip install -r requirements.txt
(Nota: requirements.txt debería contener numpy, scipy, matplotlib, y joblib)Uso (Interfaz de Línea de Comandos)El toolkit se controla a través de una potente CLI.Calcular valores y realizar auto-pruebas:python vacuum_energy.py --nu-c 7.275e11
Generar todas las figuras para el paper:python vacuum_energy.py --generate-all-plots --out ./mis_figuras
Generar un gráfico específico (ej. análisis chi2) con alta resolución:python vacuum_energy.py --plot chi2_analysis --dpi 600
Calcular nu_c a partir de un modelo de potencial teórico:python vacuum_energy.py --potential axion
API PrincipalTambién puedes importar las funciones clave en tus propios scripts.from vacuum_energy import (
    calculate_vacuum_density,
    calculate_casimir_force,
    calculate_casimir_force_corrected,
    calculate_chi2_casimir,
    calculate_g2_correction,
    LAMOREAUX_VALIDATION_DATA
)

# Densidad de energía del vacío (J/m³)
rho_result = calculate_vacuum_density(nu_c_hz=7.275e11)
print(f"ρ_vac = {rho_result.value:.3e} ± {rho_result.error:.1e} J/m³")

# Fuerza de Casimir con correcciones (Pa)
F_corr = calculate_casimir_force_corrected(d_m=1e-6, nu_c_hz=7.275e11)
print(f"Fuerza Corregida (d=1µm) = {F_corr:.3e} Pa")

# Análisis estadístico contra datos experimentales
chi2_red = calculate_chi2_casimir(
    nu_c_hz=7.275e11,
    data=LAMOREAUX_VALIDATION_DATA,
    use_corrections=True
)
print(f"Chi-cuadrado reducido = {chi2_red:.2f}")
FeaturesCálculos: Densidad de energía de vacío de alta precisión, fuerza de Casimir (ideal y corregida), corrección a g−2 y Lamb shift.Análisis Estadístico: Validación de modelos mediante chi2 contra datos experimentales publicados.Visualización: Generación de más de una docena de figuras de calidad de publicación (PDF/PNG a DPI configurable) con estilo LaTeX.Filtros UV: Soporte para múltiples modelos de supresión (exponencial, gaussiano, lorentziano, no local).Paralelismo: Usa joblib para acelerar los cálculos en CPUs multi-núcleo.Exportación: Guarda los datos crudos de todas las figuras en archivos .txt para una reproducibilidad total.Auto-pruebas: Un sistema de _selftest para verificar la integridad de los cálculos principales.ContribucionesLas contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios o envía un pull request. El código debe seguir el estilo PEP 8.LicenciaMIT License © 2025 Juan Galaz & colaboradores. Si utilizas este código en un trabajo académico, por favor cita nuestro manuscrito:Galaz, J. (2025). "h como ciclo y evento elemental: Reinterpretación conceptual de la constante de Planck". arXiv: [tu-arxiv-id]
