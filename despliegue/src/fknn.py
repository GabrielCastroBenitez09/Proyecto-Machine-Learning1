"""Implementación del algoritmo Focalized K-Nearest Neighbors (F-KNN) y reglas de dominio."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


def crear_mapa_predictores(columnas):
    """Reglas de dominio meteorológico: descartar derivados, duplicados y acumuladores colineales."""
    pos = {nombre: i for i, nombre in enumerate(columnas)}
    temp, humedad = "Outdoor Temperature(°C)", "Outdoor Humidity(%)"
    viento, rafaga = "Wind Speed(km/h)", "Gust(km/h)"
    rocio, sensacion = "DewPoint(°C)", "WindChill(°C)"
    lluvia_h, lluvia_24 = "Hour Rainfall(mm)", "24 Hour Rainfall(mm)"
    acumuladas = [c for c in (lluvia_24, "Week Rainfall(mm)", "Month Rainfall(mm)",
                              "Total Rainfall(mm)") if c in pos]
    derivados = {c for c in (rocio, sensacion) if c in pos}

    mapa = {}
    for objetivo, j in pos.items():
        if objetivo == rocio:
            permitidos = [c for c in (temp, humedad) if c in pos]
        elif objetivo == sensacion:
            permitidos = [c for c in (temp, viento) if c in pos]
        elif objetivo in acumuladas:
            permitidos = [c for c in (lluvia_h, lluvia_24, humedad) if c in pos and c != objetivo]
        else:
            permitidos = [c for c in columnas
                          if c != objetivo and c not in derivados and c not in acumuladas]
            if objetivo not in (viento, rafaga) and rafaga in permitidos:
                permitidos.remove(rafaga)
        mapa[j] = tuple(pos[c] for c in permitidos if c != objetivo)
    return mapa


class FocalizedKNNImputer(BaseEstimator, TransformerMixin):
    """
    Imputador F-KNN (Focalized K-Nearest Neighbors) para series de tiempo multivariadas.
    
    Por cada variable objetivo:
    1. Selecciona los predictores de dominio con mayor correlación de Spearman absoluta (|rho| >= tau).
    2. Identifica los mejores rezagos temporales (lags) de la propia variable.
    3. Ajusta un estimador KNN en el espacio reducido (objetivo + correlatos + rezagos).
    4. Imputa iterativamente priorizando las variables con menor cantidad de valores faltantes.
    """

    def __init__(self, n_neighbors=25, correlation_threshold=0.4, n_features=5,
                 lags=(2, 48, 96), max_lags=1, predictor_indices=None):
        self.n_neighbors = n_neighbors
        self.correlation_threshold = correlation_threshold
        self.n_features = n_features
        self.lags = lags
        self.max_lags = max_lags
        self.predictor_indices = predictor_indices

    @staticmethod
    def _lagged(values, lags):
        """Genera matriz de rezagos temporales."""
        n = len(values)
        columns = []
        for lag in lags:
            col = np.full(n, np.nan, dtype=float)
            if lag < n:
                col[lag:] = values[:-lag]
            columns.append(col)
        return np.column_stack(columns) if columns else np.empty((n, 0))

    def fit(self, X, y=None, segment_lengths=None):
        """Ajusta el imputador aprendiendo correlaciones, rezagos y vecindades sobre X."""
        X = np.asarray(X, dtype=float)
        self._fit_reference_ = X.copy()
        self.n_features_in_ = X.shape[1]
        self.scaler_ = MinMaxScaler().fit(X)
        scaled = self.scaler_.transform(X)
        corr = pd.DataFrame(scaled).corr(method="spearman").to_numpy()
        self.selected_ = {}
        self.selected_lags_ = {}
        self.imputers_ = {}

        if segment_lengths is None:
            segment_lengths = [len(scaled)]
        if sum(segment_lengths) != len(scaled):
            raise ValueError("Los límites de segmento no coinciden con las filas de entrenamiento.")

        segmentos = []
        inicio = 0
        for longitud in segment_lengths:
            fin = inicio + longitud
            segmentos.append(scaled[inicio:fin])
            inicio = fin

        for j in range(self.n_features_in_):
            allowed = (self.predictor_indices or {}).get(
                j, tuple(i for i in range(self.n_features_in_) if i != j)
            )
            candidates = [
                i for i in allowed
                if i != j and np.isfinite(corr[j, i])
                and abs(corr[j, i]) >= self.correlation_threshold
            ]
            candidates.sort(key=lambda i: abs(corr[j, i]), reverse=True)
            selected = candidates[:self.n_features]
            self.selected_[j] = selected

            target_train = np.concatenate([segmento[:, j] for segmento in segmentos])
            lag_train = np.vstack([
                self._lagged(segmento[:, j], self.lags) for segmento in segmentos
            ])
            ranking_lags = []
            for posicion, lag in enumerate(self.lags):
                pares = np.isfinite(target_train) & np.isfinite(lag_train[:, posicion])
                if pares.sum() < 3:
                    continue
                rho = pd.Series(target_train[pares]).corr(
                    pd.Series(lag_train[pares, posicion]), method="spearman"
                )
                if np.isfinite(rho):
                    ranking_lags.append((lag, float(rho)))
            ranking_lags.sort(key=lambda par: (-abs(par[1]), par[0]))
            lags_seleccionados = [
                lag for lag, _ in ranking_lags[:max(0, int(self.max_lags))]
            ]
            self.selected_lags_[j] = lags_seleccionados

            bloques = []
            for segmento in segmentos:
                rezagos = self._lagged(segmento[:, j], lags_seleccionados)
                bloques.append(np.column_stack([
                    segmento[:, j],
                    segmento[:, selected] if selected else np.empty((len(segmento), 0)),
                    rezagos,
                ]))
            context = np.vstack(bloques)
            self.imputers_[j] = KNNImputer(
                n_neighbors=self.n_neighbors, weights="uniform", keep_empty_features=True
            ).fit(context)
        return self

    def transform(self, X):
        """Imputa las celdas faltantes de X devolviendo la matriz en unidades originales."""
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Se esperaban {self.n_features_in_} variables, pero se recibieron {X.shape[1]}.")
        scaled = self.scaler_.transform(X)
        completed = scaled.copy()
        fila_vacia = np.isnan(scaled).all(axis=1)

        order = np.argsort(np.isnan(completed).sum(axis=0), kind="stable")
        for j in order:
            idx = np.flatnonzero(np.isnan(completed[:, j]) & ~fila_vacia)
            if not idx.size:
                continue
            selected = self.selected_[j]
            rezagos = self._lagged(completed[:, j], self.selected_lags_[j])
            context = np.column_stack([
                completed[:, j],
                completed[:, selected] if selected else np.empty((len(completed), 0)),
                rezagos,
            ])
            estimated = self.imputers_[j].transform(context[idx])[:, 0]
            fillable = np.isfinite(estimated)
            completed[idx[fillable], j] = estimated[fillable]

        fallback = 0.0
        train_scaled = self.scaler_.transform(self._fit_reference_)
        for j in range(self.n_features_in_):
            still_missing = np.isnan(completed[:, j])
            if still_missing.any():
                value = np.nanmean(train_scaled[:, j])
                completed[still_missing, j] = value if np.isfinite(value) else fallback
        return self.scaler_.inverse_transform(completed)
