"""
Script para entrenar y exportar el Pipeline del modelo ganador (F-KNN)
utilizando todos los datos históricos de 2018 (ajuste + validación).
"""
import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Permitir importaciones relativas desde src
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.fknn import FocalizedKNNImputer, crear_mapa_predictores
from src.preprocessing import limpiar_noches

def exportar_pipeline():
    print("=" * 60)
    print("ENTRENAMIENTO Y EXPORTACIÓN DEL MODELO GANADOR (F-KNN)")
    print("=" * 60)

    # 1. Cargar datos de 2018 (ajuste + validación)
    # Buscamos la ruta de los datos dentro del proyecto
    ruta_datos_2018 = BASE_DIR.parent / "data" / "gold" / "tramos" / "tramo_ajuste_2018.csv"
    if not ruta_datos_2018.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {ruta_datos_2018}")

    print(f"Cargando datos históricos desde: {ruta_datos_2018.name}")
    raw_2018 = pd.read_csv(ruta_datos_2018, index_col="Time_dt", parse_dates=["Time_dt", "Time_original"])
    raw_2018.index = pd.DatetimeIndex(raw_2018.index, name="Time_dt")

    # Limpieza de noches anómalas
    df_2018 = limpiar_noches(raw_2018)
    print(f"Registros limpios para entrenamiento: {len(df_2018):,} filas.")

    # 2. Definir variables modeladas
    variables = [c for c in df_2018.columns if c not in ("Time_original", "Relative Pressure(hpa)")]
    X_train = df_2018[variables].to_numpy(dtype=float)

    # 3. Configurar e instanciar el Pipeline con hiperparámetros ganadores
    print("\nConfigurando F-KNN con hiperparámetros óptimos:")
    print(" - k (vecinos): 25")
    print(" - tau (umbral correlación Spearman): 0.4")
    print(" - n (máximo de correlatos por objetivo): 5")
    print(" - max_lags (rezagos temporales): 1")
    print(" - lags candidatos: (2, 48, 96)")

    mapa_predictores = crear_mapa_predictores(variables)
    fknn_imputer = FocalizedKNNImputer(
        n_neighbors=25,
        correlation_threshold=0.4,
        n_features=5,
        lags=(2, 48, 96),
        max_lags=1,
        predictor_indices=mapa_predictores
    )

    pipeline = Pipeline([
        ("fknn", fknn_imputer)
    ])

    # 4. Ajustar el pipeline con todo 2018
    print("\nAjustando el modelo...")
    pipeline.fit(X_train)
    print("¡Modelo ajustado con éxito!")

    # 5. Serializar en .joblib
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    joblib_path = models_dir / "fknn_pipeline.joblib"
    joblib.dump(pipeline, joblib_path, compress=3)
    print(f"\n[OK] Pipeline guardado en: {joblib_path}")

    # 6. Guardar metadatos para la aplicación Streamlit
    min_vals = np.nanmin(X_train, axis=0).tolist()
    max_vals = np.nanmax(X_train, axis=0).tolist()
    metadata = {
        "modelo": "Focalized K-Nearest Neighbors (F-KNN)",
        "configuracion": "k=25, tau=0.4, n=5, max_lags=1, lags=[2, 48, 96]",
        "variables": variables,
        "n_variables": len(variables),
        "frecuencia": "30min",
        "rangos_historicos_2018": {
            var: {"min": float(min_vals[i]), "max": float(max_vals[i])}
            for i, var in enumerate(variables)
        },
        "metricas_prueba_2019": {
            "disjoint": {"R2_medio": 0.2212, "MSE_medio": 0.0180},
            "overlap": {"R2_medio": -0.0662, "MSE_medio": 0.0273}
        }
    }
    meta_path = models_dir / "metadata_despliegue.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metadatos guardados en: {meta_path}")

    # 7. Crear un dataset de ejemplo con huecos para probar Streamlit
    ruta_datos_2019 = BASE_DIR.parent / "data" / "gold" / "tramos" / "tramo_evaluacion_2019.csv"
    if ruta_datos_2019.exists():
        raw_2019 = pd.read_csv(ruta_datos_2019, index_col="Time_dt", parse_dates=["Time_dt"])
        df_sample = limpiar_noches(raw_2019).head(336)[variables].copy()  # 1 semana (336 observaciones)
        
        # Inyectar huecos realistas para simular caída de sensores
        np.random.seed(42)
        mascara = np.random.rand(*df_sample.shape) < 0.20  # 20 % de faltantes aleatorios
        df_sample_con_faltantes = df_sample.mask(mascara)
        
        ejemplo_path = BASE_DIR / "data" / "ejemplo_para_probar.csv"
        df_sample_con_faltantes.to_csv(ejemplo_path)
        print(f"[OK] Archivo de prueba para Streamlit generado en: {ejemplo_path}")

    print("\nProceso de exportación completado con éxito.")

if __name__ == "__main__":
    exportar_pipeline()
