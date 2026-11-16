"""Funciones de preprocesamiento, limpieza y validación temporal."""
from pathlib import Path
import numpy as np
import pandas as pd

LUZ = "Light(lux)"
NOCHE_INI, NOCHE_FIN = 19, 5
UMBRAL_LUZ_NOCHE = 5.0


def es_noche(idx):
    """Retorna máscara booleana indicando si la hora corresponde a horario nocturno (19:00 a 05:00)."""
    return (idx.hour >= NOCHE_INI) | (idx.hour < NOCHE_FIN)


def filas_noche_anomala(df):
    """Detecta noches completas donde se registraron valores anómalos de luz (> 5.0 lux)."""
    if LUZ not in df.columns:
        return pd.Series(False, index=df.index)
    noche = es_noche(df.index)
    id_noche = df.index.normalize() - pd.to_timedelta((df.index.hour < NOCHE_FIN).astype(int), unit="D")
    malas = pd.Index(id_noche[noche & df[LUZ].gt(UMBRAL_LUZ_NOCHE).fillna(False).to_numpy()]).unique()
    return noche & id_noche.isin(malas)


def limpiar_noches(df):
    """Excluye observaciones correspondientes a noches anómalas."""
    return df.loc[~filas_noche_anomala(df)].copy()


def preparar_dataframe_entrada(df_subido, variables_esperadas):
    """
    Valida y estandariza un DataFrame subido por el usuario en Streamlit:
    1. Asegura que la columna de tiempo se use como índice DatetimeIndex.
    2. Comprueba que estén todas las variables esperadas por el modelo.
    3. Excluye columnas innecesarias o colineales (ej. Relative Pressure).
    4. Aplica el filtro de noches anómalas si corresponde.
    """
    df = df_subido.copy()

    # Identificar columna de tiempo
    col_tiempo = None
    for candidata in ["Time_dt", "Time", "Fecha", "datetime", "Date"]:
        if candidata in df.columns:
            col_tiempo = candidata
            break

    if col_tiempo is not None:
        df[col_tiempo] = pd.to_datetime(df[col_tiempo])
        df = df.set_index(col_tiempo)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("No se encontró una columna de fecha/hora válida ni un índice temporal convertible.")

    # Validar presencia de variables
    faltantes = [v for v in variables_esperadas if v not in df.columns]
    if faltantes:
        raise ValueError(f"El archivo no contiene las siguientes columnas requeridas: {faltantes}")

    # Filtrar noches anómalas
    df = limpiar_noches(df)

    # Reordenar columnas según el orden esperado
    df_modelo = df[variables_esperadas].copy()
    return df_modelo
