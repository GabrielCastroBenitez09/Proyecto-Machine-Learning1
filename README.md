
# IMPUTACIÓN DE SERIES DE TIEMPO CON K-NN FOCALIZADO
### Gabriel Castro Benitez, Karen Daniela Torres Henao
>>>>>>> 0bb09fa (Agregar nombre)

<br>

> Machine Learning 1
>
> Pregrado en Ciencia de Datos
>
> Universidad Externado de Colombia

## Descripción
Este proyecto estudia la imputación de valores faltantes en series de tiempo meteorológicas de Aranzazu, Colombia, mediante métodos basados en vecinos cercanos. Se implementa y evalúa F-KNN (*Focalized K-Nearest Neighbors*) y se compara con K-NN, regresión lineal iterativa y líneas base de media, mediana y moda.

El flujo comprende la exploración y limpieza de los datos, la preparación reproducible de escenarios de faltantes y la comparación de los imputadores. El ajuste y la validación se realizan con tramos de 2018; el tramo de 2019 se reserva para la evaluación final. Los resultados de ajuste/CV y validación se guardan en archivos CSV dentro de `data/gold/resultados_modelos/`.

## Ejecución
Instala las dependencias desde la carpeta del proyecto:

```bash
pip install -r requirements.txt
```

Ejecuta los notebooks en orden desde `notebooks/`:

1. `01_eda.ipynb`: explora y limpia los datos; genera los archivos procesados y los tramos temporales.
2. `02_generador_faltantes_ajuste_validacion_prueba (1).ipynb`: crea escenarios reproducibles de faltantes y sus índices.
3. `03_flujo_modelamiento_ajuste_validacion_prueba_corregido.ipynb`: entrena y compara los métodos, y guarda los resultados de CV y validación.

Las rutas relativas de los notebooks están planteadas para ejecutarlos desde esa carpeta. Para reproducir el flujo completo, ejecuta sus celdas en orden. El notebook de modelamiento no ejecuta automáticamente las celdas posteriores cuando se vuelve a correr una celda anterior.

## Estructura del Repositorio
```text
MLP/
│
├── archivos_proyecto/
│   ├── guia/
│   │   ├── Focalize K-NN - An Imputation Algorithm for Time Series Datasets.pdf
│   │   └── Machine Learning I - Proyecto Integrador.pdf
│   │
│   ├── informes/
│   │   ├── Informe 1 - Comprensión del Artículo y Planteamiento.pdf
│   │   └── Informe 2 - Datos, Modelamiento y Validación
│   │
│   └── presentaciones/
│       └── Presentación 1 - Comprensión del Artículo y Planteamiento.pdf
│
├── data/
│   ├── bronze/
│   │   └── meteorologicos_aranzazu_raw.csv
│   │
│   ├── silver/
│   │   └── meteorologicos_aranzazu_silver.csv
│   │
│   ├── gold/
│   │   ├── mascaras/
│   │   │   ├── indice_ajuste_v1.csv
│   │   │   ├── indice_validacion_v1.csv
│   │   │   ├── indice_evaluacion_v2.csv
│   │   │   ├── mascaras_ajuste_v1.npz
│   │   │   ├── mascaras_validacion_v1.npz
│   │   │   └── mascaras_evaluacion_v2.npz
│   │   ├── resultados_modelos/
│   │   │   ├── baseline_train_cv.csv
│   │   │   ├── baseline_validation_2018.csv
│   │   │   ├── fknn_train_cv.csv
│   │   │   ├── fknn_train_cv_exploracion.csv
│   │   │   ├── fknn_validation_2018.csv
│   │   │   ├── knn_train_cv.csv
│   │   │   ├── knn_validation_2018.csv
│   │   │   ├── linear_regression_train_cv.csv
│   │   │   └── linear_regression_validation_2018.csv
│   │   ├── tramos/
│   │   │   ├── tramo_ajuste_2018.csv
│   │   │   └── tramo_evaluacion_2019.csv
│   │
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_generador_faltantes_ajuste_validacion_prueba (1).ipynb
│   └── 03_flujo_modelamiento_ajuste_validacion_prueba_corregido.ipynb
│
├── requirements.txt
│
└── README.md
```
