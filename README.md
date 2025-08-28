<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vacuum Energy Toolkit (v22.0) - SPA</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        /* Chosen Palette: Serene Scholar */
        :root {
            --color-background: #F8F4E3; /* Warm neutral light background */
            --color-surface: #E1D9C5; /* Slightly darker neutral for cards/sections */
            --color-text: #333333; /* Dark text for readability */
            --color-heading: #2C3E50; /* Darker heading for contrast */
            --color-accent: #4A90E2; /* Subtle blue for highlights/buttons */
            --color-secondary-text: #6B7280; /* Gray for less prominent text */
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--color-background);
            color: var(--color-text);
            scroll-behavior: smooth;
        }
        .container {
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .section-padding {
            padding-top: 4rem;
            padding-bottom: 4rem;
        }
        .sticky-nav {
            position: sticky;
            top: 0;
            z-index: 50;
        }
        .card {
            background-color: var(--color-surface);
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .btn-primary {
            background-color: var(--color-accent);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 0.375rem;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        .btn-primary:hover {
            background-color: #357ABD; /* Darker shade of accent */
        }
        .chart-container {
            position: relative;
            width: 100%;
            max-width: 600px; /* Maximum width for charts */
            margin-left: auto;
            margin-right: auto;
            height: 300px; /* Base height, adjust with media queries or use Tailwind for responsive heights */
            max-height: 400px; /* Max height to prevent excessive vertical stretch */
        }
        @media (min-width: 768px) {
            .chart-container {
                height: 350px;
            }
        }
    </style>
</head>
<body class="text-gray-800">
    <!-- Chosen Palette: Serene Scholar -->
    <!-- Application Structure Plan: La aplicación está diseñada con una arquitectura de página única (SPA) con secciones temáticas principales: Inicio (Hero), Descripción del Proyecto, Características Clave, Figuras de Salida (Visualizaciones interactivas), Instalación y Uso, Validación Científica, y un Pie de Página. Esta estructura se eligió para facilitar una navegación intuitiva y una exploración lineal o no lineal del contenido del informe. Los usuarios pueden saltar a secciones específicas a través de la barra de navegación. La sección de Figuras de Salida es interactiva, permitiendo al usuario seleccionar plots para ver descripciones detalladas, mejorando la comprensión y la interactividad sobre la información generada por el toolkit. -->
    <!-- Visualization & Content Choices:
    - Main Textual Content: Goal -> Inform; Presentation Method -> HTML text (headings, paragraphs, lists); Interaction -> Scroll navigation; Justification -> Clarity and direct presentation of report text.
    - Output Figures Section: Goal -> Allow exploration of each plot's purpose and output; Presentation Method -> Dynamic textual display of plot descriptions, triggered by clickable buttons/list items; Interaction -> Click on plot name to update a central display area with its description; Justification -> Without actual data from the report, this method best provides interactive insight into what each generated plot represents, enhancing understanding of the toolkit's outputs. Library/Method -> Vanilla JS for dynamic content update. NO Chart.js/Plotly.js for actual report data visualizations as no underlying data is provided in the source report.
    - Code Blocks: Goal -> Inform on usage; Presentation Method -> Pre-formatted HTML code blocks; Interaction -> Copy to clipboard (implicit, but not implemented for brevity here); Justification -> Clear presentation of code examples.
    -->
    <!-- CONFIRMATION: NO SVG graphics used. NO Mermaid JS used. -->

    <header class="bg-gradient-to-r from-blue-700 to-blue-500 text-white p-4 sticky-nav shadow-lg">
        <nav class="container mx-auto flex justify-between items-center">
            <h1 class="text-2xl font-bold tracking-tight">
                <a href="#hero">Vacuum Energy Toolkit</a>
            </h1>
            <ul class="flex space-x-6 text-lg font-medium">
                <li><a href="#descripcion" class="hover:text-blue-200 transition duration-300">Descripción</a></li>
                <li><a href="#caracteristicas" class="hover:text-blue-200 transition duration-300">Características</a></li>
                <li><a href="#plots" class="hover:text-blue-200 transition duration-300">Visualizaciones</a></li>
                <li><a href="#uso" class="hover:text-blue-200 transition duration-300">Uso</a></li>
                <li><a href="#validacion" class="hover:text-blue-200 transition duration-300">Validación</a></li>
            </ul>
        </nav>
    </header>

    <main class="container mx-auto mt-8 px-4">
        <!-- Hero Section -->
        <section id="hero" class="section-padding text-center bg-white rounded-lg shadow-xl p-8 mb-12">
            <h2 class="text-5xl font-extrabold text-gray-900 mb-4 tracking-tight">Vacuum Energy Toolkit (v22.0) 🌌</h2>
            <p class="text-xl text-gray-600 mb-6 italic">Cálculo científico de densidad de energía del vacío, fuerza de Casimir y fenómenos QG</p>
            <div class="flex justify-center space-x-4 mb-8">
                <a href="#" class="inline-block px-6 py-2 bg-yellow-500 text-white font-semibold rounded-full shadow-md hover:bg-yellow-600 transition duration-300">License: MIT</a>
                <a href="https://www.python.org/downloads/" target="_blank" class="inline-block px-6 py-2 bg-blue-600 text-white font-semibold rounded-full shadow-md hover:bg-blue-700 transition duration-300">Python 3.9+</a>
                <a href="#" class="inline-block px-6 py-2 bg-red-700 text-white font-semibold rounded-full shadow-md hover:bg-red-800 transition duration-300">arXiv: XXXX.XXXXX</a>
            </div>
            <blockquote class="text-xl leading-relaxed text-gray-700 bg-gray-100 p-6 rounded-lg border-l-4 border-blue-500">
                <p class="font-bold text-gray-800">Versión 22.0:</p>
                Reescritura completa con validación manual. Incluye <strong class="text-blue-700">correcciones físicas</strong>, <strong class="text-blue-700">análisis estadístico robusto</strong> y <strong class="text-blue-700">16 figuras de calidad publicación</strong>, generadas en 3 formatos diferentes (.pdf, .png, .svg).
            </blockquote>
        </section>

        <!-- Descripción del Proyecto -->
        <section id="descripcion" class="section-padding bg-white rounded-lg shadow-xl p-8 mb-12">
            <h2 class="text-4xl font-bold text-gray-900 mb-6 border-b-2 border-blue-500 pb-2">🚀 Descripción del Proyecto</h2>
            <p class="text-lg leading-relaxed text-gray-700 mb-6">
                Este repositorio alberga el motor computacional *end-to-end* para el programa de investigación **"h como Ciclo y Evento Elemental" (Galaz, 2025)**.
                `vacuum_energy.py` es una herramienta robusta diseñada para explorar fenómenos cuánticos del vacío, incluyendo:
            </p>
            <ul class="list-disc list-inside text-lg text-gray-700 space-y-2 pl-4">
                <li><strong class="text-blue-700">Densidad de energía del vacío (rho_vac)</strong> con modelado de error sistemático.</li>
                <li><strong class="text-blue-700">Fuerza de Casimir</strong> con correcciones térmicas y de rugosidad.</li>
                <li><strong class="text-blue-700">Modificaciones del espacio de parámetros del axión.</strong></li>
                <li><strong class="text-blue-700">Anomalías QED</strong> (g-2, Lamb shift).</li>
                <li><strong class="text-blue-700">Análisis de sensibilidad</strong> a la tensión de Hubble.</li>
            </ul>
        </section>

        <!-- Características Clave -->
        <section id="caracteristicas" class="section-padding bg-white rounded-lg shadow-xl p-8 mb-12">
            <h2 class="text-4xl font-bold text-gray-900 mb-6 border-b-2 border-blue-500 pb-2">✨ Características Clave</h2>

            <div class="mb-8">
                <h3 class="text-3xl font-semibold text-gray-800 mb-4">🔍 Fenómenos Cuánticos del Vacío</h3>
                <ul class="list-disc list-inside text-lg text-gray-700 space-y-2 pl-4">
                    <li><strong class="text-blue-700">Densidad de Energía del Vacío (rho_vac):</strong> Calcula rho_vac utilizando 4 filtros de corte UV: `exp`, `gauss`, `lorentz`, `nonloc`.</li>
                    <li><strong class="text-blue-700">Fuerza de Casimir:</strong> Modelo ideal con correcciones avanzadas para rugosidad superficial y efectos térmicos.</li>
                    <li><strong class="text-blue-700">Axiones:</strong> Predicción de la ventana de búsqueda de axiones con escalado nu_c y visualización del flujo de acoplamiento ASG.</li>
                    <li><strong class="text-blue-700">Anomalías QED:</strong> Correcciones a g-2 y al Lamb shift.</li>
                    <li><strong class="text-blue-700">Cosmología:</strong> Análisis de sensibilidad de la tensión de Hubble.</li>
                </ul>
            </div>

            <div>
                <h3 class="text-3xl font-semibold text-gray-800 mb-4">📊 Rigor y Rendimiento</h3>
                <ul class="list-disc list-inside text-lg text-gray-700 space-y-2 pl-4">
                    <li><strong class="text-blue-700">Validación Experimental:</strong> Análisis de chi2 contra datos experimentales de Lamoreaux (1997).</li>
                    <li><strong class="text-blue-700">Propagación de Errores:</strong> Integración completa con la librería `uncertainties.py`.</li>
                    <li><strong class="text-blue-700">Rendimiento:</strong> Aceleración 5-10x mediante paralelización con `Joblib`.</li>
                    <li><strong class="text-blue-700">Estabilidad:</strong> Guardas NaN/Inf en todas las integraciones numéricas.</li>
                </ul>
            </div>
        </section>

        <!-- Figuras de Salida (Visualizaciones Interactivas) -->
        <section id="plots" class="section-padding bg-white rounded-lg shadow-xl p-8 mb-12">
            <h2 class="text-4xl font-bold text-gray-900 mb-6 border-b-2 border-blue-500 pb-2">📈 Figuras de Salida (16 únicas, en 3 formatos: .pdf, .png, .svg)</h2>
            <p class="text-lg leading-relaxed text-gray-700 mb-6">
                El script genera un total de <strong class="text-blue-700">16 figuras profesionales</strong>, cada una disponible en PDF, PNG y SVG. Estas figuras están listas para su publicación y se organizan por tipo de filtro cuando se genera en la carpeta `resultados_completos`. Haz clic en cualquier título para ver su descripción detallada.
            </p>

            <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                <!-- Plot list on the left -->
                <div class="md:col-span-1 lg:col-span-1 card p-4">
                    <h3 class="text-2xl font-semibold text-gray-800 mb-4">Seleccionar Figura:</h3>
                    <ul id="plot-list" class="space-y-2">
                        <!-- Plot buttons will be dynamically inserted here -->
                    </ul>
                </div>

                <!-- Dynamic plot description display on the right -->
                <div class="md:col-span-1 lg:col-span-2 card p-6 flex flex-col items-center justify-center">
                    <h3 id="plot-display-title" class="text-3xl font-semibold text-gray-800 mb-4 text-center">Haz clic en una figura para ver su descripción</h3>
                    <p id="plot-display-description" class="text-lg text-gray-700 text-center"></p>
                    <!-- This div could theoretically hold a chart if data were available -->
                    <div id="plot-chart-container" class="chart-container mt-6 hidden">
                        <canvas id="myChart"></canvas>
                    </div>
                </div>
            </div>
        </section>

        <!-- Instalación y Uso -->
        <section id="uso" class="section-padding bg-white rounded-lg shadow-xl p-8 mb-12">
            <h2 class="text-4xl font-bold text-gray-900 mb-6 border-b-2 border-blue-500 pb-2">🛠️ Instalación y Uso</h2>

            <div class="mb-8">
                <h3 class="text-3xl font-semibold text-gray-800 mb-4">Instalar Dependencias</h3>
                <p class="text-lg leading-relaxed text-gray-700 mb-4">Primero, asegúrate de tener Python 3.9+ instalado y luego clona el repositorio e instala las dependencias:</p>
                <pre class="bg-gray-900 text-white p-4 rounded-md overflow-x-auto text-sm mb-4"><code>git clone https://github.com/tu-usuario/vacuum-energy.git
cd vacuum-energy
pip install -r requirements.txt
# Dependencias CORE: numpy, scipy, matplotlib
# Dependencias OPCIONALES: joblib (paralelización), uncertainties (errores), ipywidgets (interactivo)</code></pre>
            </div>

            <div class="mb-8">
                <h3 class="text-3xl font-semibold text-gray-800 mb-4">Uso desde la Línea de Comandos (CLI)</h3>
                <pre class="bg-gray-900 text-white p-4 rounded-md overflow-x-auto text-sm mb-4"><code># Auto-prueba + cálculos básicos con ν_c calibrada
python vacuum_energy.py --nu-c 7.275e11

# Generar TODAS las 16 figuras en .pdf, .png y .svg
python vacuum_energy.py --generate-all-plots --out ./figuras

# Generar un gráfico específico (ej: análisis χ²) con alta resolución
python vacuum_energy.py --plot chi2_analysis --dpi 600

# Calcular ν_c para un modelo potencial específico (ej: axion)
python vacuum_energy.py --potential axion --ma-max 15.0

# Ejecutar en modo interactivo (requiere ipywidgets)
python vacuum_energy.py --interactive</code></pre>
            </div>

            <div>
                <h3 class="text-3xl font-semibold text-gray-800 mb-4">Uso como Módulo de Python</h3>
                <p class="text-lg leading-relaxed text-gray-700 mb-4">Puedes importar las funciones directamente en tus propios scripts de Python:</p>
                <pre class="bg-gray-900 text-white p-4 rounded-md overflow-x-auto text-sm"><code>from vacuum_energy import (
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
# plot_casimir_with_corrections(nu_c_hz=7.275e11, data=LAMOREAUX_VALIDATION_DATA, dpi=300, out_dir="mis_figuras")</code></pre>
            </div>
        </section>

        <!-- Validación Científica -->
        <section id="validacion" class="section-padding bg-white rounded-lg shadow-xl p-8 mb-12">
            <h2 class="text-4xl font-bold text-gray-900 mb-6 border-b-2 border-blue-500 pb-2">✅ Validación Científica</h2>
            <p class="text-lg leading-relaxed text-gray-700 mb-4">
                El toolkit ha sido rigurosamente validado, demostrando una alta precisión y consistencia:
            </p>
            <ul class="list-disc list-inside text-lg text-gray-700 space-y-2 pl-4">
                <li><strong class="text-blue-700">rho_vac</strong> dentro del 0.1% del valor observado.</li>
                <li>Consistencia en el signo de la <strong class="text-blue-700">fuerza de Casimir</strong>.</li>
                <li>Corrección de <strong class="text-blue-700">g-2</strong> que coincide con la anomalía QED.</li>
                <li><strong class="text-blue-700">chi2</strong> &lt; 1.5 para el modelo corregido.</li>
            </ul>
        </section>
    </main>

    <footer class="bg-gray-800 text-white p-8 text-center">
        <div class="container mx-auto">
            <div class="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
                <p class="text-lg">&copy; 2025 Vacuum Energy Toolkit. Todos los derechos reservados.</p>
                <div class="flex space-x-6 text-xl">
                    <a href="#" class="hover:text-blue-400 transition duration-300">📄 Licencia MIT</a>
                    <a href="#" class="hover:text-blue-400 transition duration-300">📚 arXiv: XXXX.XXXXX [physics.gen-ph]</a>
                </div>
            </div>
        </div>
    </footer>

    <script>
        const plotDetails = {
            "casimir_with_corrections": {
                title: "Comparación de la Fuerza de Casimir con y sin Correcciones",
                description: "Esta figura compara el comportamiento teórico de la fuerza de Casimir ideal con el modelo corregido (considerando rugosidad y temperatura) frente a los datos experimentales. Es crucial para validar la precisión del modelo."
            },
            "chi2_analysis_casimir": {
                title: "Análisis chi2 para la Fuerza de Casimir",
                description: "Este gráfico muestra el valor del chi-cuadrado reducido (chi2_red) para los modelos de fuerza de Casimir (ideal y corregido) en función de la frecuencia de corte (nu_c). Permite identificar el mejor ajuste a los datos experimentales de Lamoreaux."
            },
            "casimir_comparison_relative": {
                title: "Comparación Relativa de la Fuerza de Casimir entre Filtros",
                description: "Aquí se comparan las predicciones de la fuerza de Casimir obtenidas con diferentes tipos de filtros UV (gauss, lorentz, nonloc) con respecto al filtro exponencial, revelando cómo cada filtro influye en los resultados."
            },
            "sensitivity_eta_vs_rho": {
                title: "Sensibilidad de rho_vac vs. eta",
                description: "Esta figura ilustra cómo la densidad de energía del vacío (rho_vac) varía en función de un factor de escala (eta = nu_c / nu_c0), lo que es fundamental para entender la sensibilidad del modelo a los parámetros de corte."
            },
            "casimir_absolute": {
                title: "Fuerza de Casimir Absoluta vs. Distancia con Efectos Térmicos",
                description: "Representa la magnitud de la fuerza de Casimir en función de la distancia entre las placas, mostrando la influencia de las correcciones térmicas y cómo se desvía del caso ideal (temperatura cero)."
            },
            "rho_vac_convergence": {
                title: "Convergencia de rho_vac con omega_max",
                description: "Este gráfico analiza la convergencia del cálculo de la densidad de energía del vacío (rho_vac) a medida que el límite superior de integración (omega_max) aumenta, asegurando la estabilidad numérica de los resultados."
            },
            "axion_window_shift": {
                title: "Desplazamiento de la Ventana de Búsqueda del Axión",
                description: "Visualiza cómo el espacio de parámetros para la búsqueda de axiones se ve modificado por la introducción de la frecuencia de corte (nu_c), un elemento clave para la física de partículas y la cosmología."
            },
            "asg_flow": {
                title: "Flujo de Acoplamiento Axión-Fotón-Fotón",
                description: "Este plot muestra el comportamiento del acoplamiento axión-fotón-fotón (ASG) con y sin la aplicación del filtro de corte, lo que es importante para estudiar las interacciones de los axiones."
            },
            "kibble_analysis": {
                title: "Simulación de la Balanza de Kibble",
                description: "Una simulación de los resultados esperados en un experimento de balanza de Kibble al medir la fuerza de Casimir, ayudando a contextualizar las predicciones teóricas con posibles datos experimentales."
            },
            "hubble_tension_sensitivity": {
                title: "Sensibilidad a la Tensión de Hubble",
                description: "Este gráfico examina cómo la densidad de energía del vacío (rho_vac) responde a variaciones en la constante de Hubble (H_0), proporcionando información crucial para la resolución de la 'tensión de Hubble' en cosmología."
            },
            "filter_comparison": {
                title: "Comparación de rho_vac entre Diferentes Filtros",
                description: "Compara el valor de la densidad de energía del vacío (rho_vac) calculado con cada uno de los cuatro filtros UV disponibles, destacando las diferencias y similitudes entre ellos."
            },
            "roughness_simulation": {
                title: "Efecto de la Rugosidad en la Fuerza de Casimir",
                description: "Simula cómo las irregularidades o rugosidades de la superficie de las placas afectan la magnitud de la fuerza de Casimir, ofreciendo una visión de las correcciones necesarias para experimentos reales."
            },
            "kappa_calibration": {
                title: "Calibración de Kappa",
                description: "Este plot está dedicado a la calibración del parámetro kappa, un elemento fundamental en los cálculos de la energía del vacío y la fuerza de Casimir."
            },
            "lamb_shift_correction": {
                title: "Corrección del Lamb Shift",
                description: "Ilustra la corrección aplicada al Lamb shift, una anomalía QED, en función de los parámetros del modelo, lo que contribuye a la precisión de las predicciones teóricas."
            },
            "g2_correction": {
                title: "Corrección del Momento Magnético Anómalo del Electrón (g-2)",
                description: "Este gráfico muestra la corrección calculada para el momento magnético anómalo del electrón (g-2), un área clave para la validación de la electrodinámica cuántica (QED)."
            },
            "rho_vac_vs_nu_c": {
                title: "Densidad de Energía del Vacío vs. Frecuencia de Corte",
                description: "Presenta la relación entre la densidad de energía del vacío (rho_vac) y la frecuencia de corte (nu_c), un parámetro central en el modelo."
            }
        };

        const plotList = document.getElementById('plot-list');
        const plotDisplayTitle = document.getElementById('plot-display-title');
        const plotDisplayDescription = document.getElementById('plot-display-description');
        const plotChartContainer = document.getElementById('plot-chart-container');
        let myChartInstance = null; // To store Chart.js instance

        // Populate plot list and add event listeners
        for (const key in plotDetails) {
            const listItem = document.createElement('li');
            const button = document.createElement('button');
            button.className = 'text-left w-full p-2 rounded-md hover:bg-blue-200 transition duration-200 text-gray-700 font-medium';
            button.textContent = plotDetails[key].title;
            button.dataset.plotKey = key;
            button.addEventListener('click', (event) => {
                const selectedKey = event.target.dataset.plotKey;
                plotDisplayTitle.textContent = plotDetails[selectedKey].title;
                plotDisplayDescription.textContent = plotDetails[selectedKey].description;

                // Example of how you *could* dynamically update a chart if you had data
                // For this report, we focus on textual description as actual data is not provided.
                if (myChartInstance) {
                    myChartInstance.destroy(); // Destroy previous chart instance
                }
                plotChartContainer.classList.add('hidden'); // Hide container by default
            });
            listItem.appendChild(button);
            plotList.appendChild(listItem);
        }
    </script>
</body>
</html>
