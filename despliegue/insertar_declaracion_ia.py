"""
Script para insertar la celda de Declaración de Uso de IA
como primera celda en los notebooks del proyecto MLP.
Ejecutar una sola vez.
"""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"

DECLARACION_SOURCE = [
    "## Declaración sobre el uso de Inteligencia Artificial\n",
    "\n",
    "Durante el desarrollo e implementación del código de este notebook, **utilizamos** las siguientes herramientas de apoyo tecnológico bajo nuestra supervisión directa:\n",
    "\n",
    "- **GitHub Copilot (GitHub, 2026):** Utilizado como asistente de autocompletado y sugerencia de fragmentos de código repetitivos en Python para el procesamiento de series de tiempo, la implementación del estimador F-KNN y el flujo de modelamiento.\n",
    "\n",
    "- **Antigravity (Google DeepMind, 2026):** Empleado como asistente de ingeniería de software en la depuración del código, la modularización del estimador `FocalizedKNNImputer`, la estructuración del framework de despliegue en Streamlit y la construcción de visualizaciones interactivas.\n",
    "\n",
    "La validación experimental, la verificación del código, la interpretación analítica de los resultados y la autoría íntegra del contenido técnico recaen exclusivamente sobre nosotros como autores del proyecto.\n",
    "\n",
    "> **Autores:** Gabriel Castro Benítez · Karen Daniela Torres Henao  \n",
    "> Pregrado en Ciencia de Datos — Universidad Externado de Colombia, 2026."
]

CELDA_DECLARACION = {
    "cell_type": "markdown",
    "id": str(uuid.uuid4()).replace("-", "")[:8],
    "metadata": {},
    "source": DECLARACION_SOURCE
}

notebooks = list(NOTEBOOKS_DIR.glob("*.ipynb"))
print(f"Notebooks encontrados: {len(notebooks)}")

for nb_path in notebooks:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Verificar si ya tiene la declaración para no duplicar
    primera = "".join(nb["cells"][0]["source"]) if nb["cells"] else ""
    if "Declaración sobre el uso de Inteligencia Artificial" in primera:
        print(f"  [SKIP] {nb_path.name} — ya tiene la declaración.")
        continue

    # Insertar como primera celda
    nb["cells"].insert(0, CELDA_DECLARACION)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"  [OK]   {nb_path.name} — declaración insertada.")

print("\nListo.")
