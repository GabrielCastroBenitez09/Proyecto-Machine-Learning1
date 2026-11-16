# Despliegue del Imputador Meteorológico con F-KNN

Este directorio contiene la solución completa e independiente para el despliegue en producción del modelo ganador **F-KNN (*Focalized K-Nearest Neighbors*)** mediante una aplicación web interactiva desarrollada en **Streamlit**.

---

## 📁 Estructura del Módulo de Despliegue

```text
despliegue/
│
├── src/                          # Módulos Python reutilizables (evita error de __main__)
│   ├── __init__.py
│   ├── fknn.py                   # Clase FocalizedKNNImputer y reglas de dominio
│   └── preprocessing.py          # Limpieza de noches anómalas y validación de entrada
│
├── models/                       # Artefactos del modelo entrenado
│   ├── fknn_pipeline.joblib      # Pipeline serializado ajustado con todo 2018
│   └── metadata_despliegue.json  # Variables, rangos admisibles y métricas de prueba
│
├── data/                         # Datos de prueba para la interfaz
│   └── ejemplo_para_probar.csv   # Muestra temporal de 1 semana con ~20 % de faltantes
│
├── app.py                        # Aplicación web interactiva en Streamlit
├── export_model.py               # Script para reentrenar y reexportar el modelo
├── requirements.txt              # Librerías necesarias para el despliegue
└── README.md                     # Guía de uso y documentación técnica
```

---

## ⚙️ Configuración del Modelo Ganador

El modelo desplegado corresponde a la mejor configuración encontrada en la fase de optimización sistemática:

* **Algoritmo:** Focalized K-Nearest Neighbors (`F-KNN`)
* **Vecinos cercanos ($k$):** $25$
* **Umbral de correlación de Spearman ($\tau$):** $0.4$
* **Máximo de correlatos por variable ($n$):** $5$
* **Rezagos temporales (**max_lags**):** $1$ (candidatos: $2, 48, 96$ pasos de 30 min)
* **Entrenamiento:** Ajustado con el 100 % de los datos históricos de 2018 (`ajuste` + `validacion`).

---

## 🚀 Cómo Ejecutar la Aplicación

### 1. Instalar dependencias
Desde la terminal, sitúate en esta carpeta (`MLP/despliegue`) e instala los paquetes necesarios:

```bash
pip install -r requirements.txt
```

### 2. Iniciar Streamlit
Ejecuta la aplicación web con el siguiente comando:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador web en la dirección local:
`http://localhost:8501`

---

## 🖥️ Funcionalidades de la Interfaz

1. **Carga Flexible:** Permite arrastrar cualquier archivo CSV meteorológico o usar el botón de **"Cargar datos de ejemplo"** (una semana de 2019 con 20 % de huecos sintéticos).
2. **Diagnóstico Automático:** Muestra un panel con total de celdas faltantes, rango temporal y desglose variable por variable comparado con los rangos históricos de 2018.
3. **Imputación en un Clic:** Ejecuta el pipeline en memoria devolviendo la serie completa sin pérdida de continuidad.
4. **Inspección Visual Interactiva con Plotly:** Gráfica interactiva de serie de tiempo con selector de rango temporal (*range slider*), zoom dinámico y tooltips detallados. Diseñada bajo la **paleta cromática Viridis** (línea continua en tono teal `#21918c` y puntos imputados en amarillo brillante `#fde725` con borde púrpura `#440154`), además de un gráfico de barras con el porcentaje de faltantes por sensor.
5. **Descarga de Datos:** Botón para exportar el dataset limpio y completo en formato `.csv`.

---

## 🔄 Reentrenar o Regenerar el Modelo (`.joblib`)

Si en el futuro se actualizan los datos de 2018 o se modifican los hiperparámetros, puedes volver a generar el pipeline ejecutando:

```bash
python export_model.py
```
Este script ajustará el modelo desde cero y actualizará automáticamente `models/fknn_pipeline.joblib` y `models/metadata_despliegue.json`.
