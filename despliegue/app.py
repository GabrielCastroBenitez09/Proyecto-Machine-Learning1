"""
Aplicación Web Interactiva en Streamlit para el Despliegue de F-KNN
Imputación de Series de Tiempo Meteorológicas — Aranzazu, Caldas
Visualización interactiva con Plotly y paleta Viridis
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False

# Importante: Importar la clase personalizada antes de cargar el .joblib
from src.fknn import FocalizedKNNImputer
from src.preprocessing import preparar_dataframe_entrada, limpiar_noches

# Configuración de la página
st.set_page_config(
    page_title="Imputador Meteorológico F-KNN",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

# Paleta Viridis estándar
VIRIDIS_DARK_PURPLE = "#440154"
VIRIDIS_BLUE = "#3b528b"
VIRIDIS_TEAL = "#21918c"
VIRIDIS_GREEN = "#5ec962"
VIRIDIS_YELLOW = "#fde725"

# ==========================================
# GESTIÓN DE ESTADO (Session State)
# ==========================================
if "df_crudo" not in st.session_state:
    st.session_state["df_crudo"] = None
if "origen_datos" not in st.session_state:
    st.session_state["origen_datos"] = None
if "df_imputado" not in st.session_state:
    st.session_state["df_imputado"] = None


@st.cache_resource
def cargar_artefactos():
    """Carga el pipeline entrenado y sus metadatos desde disco."""
    pipeline_path = MODELS_DIR / "fknn_pipeline.joblib"
    meta_path = MODELS_DIR / "metadata_despliegue.json"

    if not pipeline_path.exists() or not meta_path.exists():
        st.error("No se encontraron los artefactos del modelo en 'models/'. Ejecuta primero 'python export_model.py'.")
        st.stop()

    pipeline = joblib.load(pipeline_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, metadata

pipeline, metadata = cargar_artefactos()
variables_modelo = metadata["variables"]

# ==========================================
# BARRA LATERAL (Sidebar)
# ==========================================
with st.sidebar:
    # Título distintivo para la barra lateral
    st.markdown("## 🧭 Panel de Control F-KNN")
    st.caption("Estación Meteorológica Aranzazu (Caldas)")

    # 1. Información de autores
    st.markdown("### 👥 Autores")
    st.markdown("""
    * **Gabriel Castro Benítez**
    * **Karen Daniela Torres Henao**
    
    *Machine Learning 1*  
    *Pregrado en Ciencia de Datos*  
    *Universidad Externado de Colombia*
    """)

    st.markdown("---")
    st.subheader("⚙️ Configuración del Modelo")
    st.markdown(f"**Algoritmo:** `{metadata['modelo']}`")
    st.markdown(f"**Hiperparámetros:** `{metadata['configuracion']}`")
    st.markdown(f"**Frecuencia esperada:** `{metadata['frecuencia']}`")

    st.markdown("---")
    st.subheader("📊 Métricas en Prueba 2019")
    m_dis = metadata["metricas_prueba_2019"]["disjoint"]
    m_ov = metadata["metricas_prueba_2019"]["overlap"]
    st.write(f"• **Disjoint:** $R^2$: `{m_dis['R2_medio']}`, $MSE$: `{m_dis['MSE_medio']}`")
    st.write(f"• **Overlap:** $R^2$: `{m_ov['R2_medio']}`, $MSE$: `{m_ov['MSE_medio']}`")

    st.markdown("---")
    st.subheader("💡 Atajos de datos")
    if st.button("📂 Cargar datos de ejemplo (Semana 2019)", use_container_width=True):
        ejemplo_path = DATA_DIR / "ejemplo_para_probar.csv"
        if ejemplo_path.exists():
            st.session_state["df_crudo"] = pd.read_csv(ejemplo_path, index_col=0, parse_dates=True)
            st.session_state["origen_datos"] = "Dataset de ejemplo (Semana 2019 con 20 % de faltantes)"
            st.session_state["df_imputado"] = None
            st.rerun()
        else:
            st.error("No se encontró el archivo 'data/ejemplo_para_probar.csv'.")

    if st.session_state["df_crudo"] is not None:
        if st.button("🔄 Limpiar / Cargar otro archivo", use_container_width=True):
            st.session_state["df_crudo"] = None
            st.session_state["origen_datos"] = None
            st.session_state["df_imputado"] = None
            st.rerun()

    st.markdown("---")

    # 2. Declaración sobre el uso de Inteligencia Artificial
    st.markdown("### 🤖 Uso Declarado de IA")
    st.markdown("""
    Durante el diseño, estructuración e implementación de este módulo de despliegue, **utilizamos** la siguiente herramienta de apoyo tecnológico bajo nuestra supervisión directa:

    * **Antigravity (Google DeepMind, 2026):** Empleado como asistente de ingeniería de software para la modularización del estimador `FocalizedKNNImputer`, la estructuración del framework de despliegue en Streamlit, el control de persistencia de estado (`session_state`) y la construcción de las visualizaciones interactivas en Plotly con paleta Viridis.

    La validación experimental, la verificación del código, la interpretación analítica de los resultados y la autoría íntegra del contenido técnico recaen exclusivamente sobre nosotros como autores del proyecto.
    """)

    st.markdown("---")

    # 3. Logos institucionales en la parte inferior
    logo_ext = ASSETS_DIR / "logo_externado.png"
    logo_cd = ASSETS_DIR / "logo_ciencia_de_datos.png"

    if logo_ext.exists() and logo_cd.exists():
        col_l1, col_l2 = st.columns([1.3, 1])
        with col_l1:
            st.image(str(logo_ext), use_container_width=True)
        with col_l2:
            st.image(str(logo_cd), use_container_width=True)
    elif logo_ext.exists():
        st.image(str(logo_ext), use_container_width=True)
    elif logo_cd.exists():
        st.image(str(logo_cd), width=160)

# ==========================================
# CUERPO PRINCIPAL
# ==========================================
st.title("🌤️ Sistema de Imputación de Sensores Meteorológicos")
st.markdown("""
Esta plataforma utiliza el modelo ganador **F-KNN (*Focalized K-Nearest Neighbors*)** para reconstruir 
valores faltantes en series temporales meteorológicas multivariadas. Combina selección de correlatos 
por correlación de Spearman y rezagos temporales autoregresivos.
""")

# Advertencia si Plotly no está instalado
if not PLOTLY_INSTALLED:
    st.warning("⚠️ La librería `plotly` no está instalada. Para visualizar las gráficas interactivas, ejecuta en tu terminal: `pip install plotly`.")

# ==========================================
# SECCIÓN 1: CARGA DE ARCHIVO
# ==========================================
st.markdown("### 1. Cargar archivo de datos")

archivo_subido = st.file_uploader(
    "Sube un archivo CSV con las series de tiempo (frecuencia recomendada de 30 minutos):",
    type=["csv"]
)

if archivo_subido is not None:
    try:
        nuevo_df = pd.read_csv(archivo_subido)
        if st.session_state["origen_datos"] != archivo_subido.name:
            st.session_state["df_crudo"] = nuevo_df
            st.session_state["origen_datos"] = archivo_subido.name
            st.session_state["df_imputado"] = None
            st.rerun()
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")

# ==========================================
# SECCIÓN 2: DIAGNÓSTICO
# ==========================================
df_crudo = st.session_state["df_crudo"]

if df_crudo is not None:
    st.info(f"📁 **Datos activos:** {st.session_state['origen_datos']}")

    try:
        df_preparado = preparar_dataframe_entrada(df_crudo, variables_modelo)
    except Exception as err:
        st.error(f"Error de validación en los datos: {err}")
        st.stop()

    n_filas = len(df_preparado)
    total_celdas = df_preparado.size
    total_faltantes = int(df_preparado.isna().sum().sum())
    pct_faltantes_global = (total_faltantes / total_celdas) * 100

    st.markdown("### 2. Diagnóstico de valores faltantes")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Observaciones", f"{n_filas:,}")
    kpi2.metric("Rango Temporal", f"{df_preparado.index.min().strftime('%d/%m/%Y')} a {df_preparado.index.max().strftime('%d/%m/%Y')}")
    kpi3.metric("Celdas Faltantes", f"{total_faltantes:,}")
    kpi4.metric("% Faltantes Global", f"{pct_faltantes_global:.2f} %")

    # Resumen por sensor
    faltantes_col = df_preparado.isna().sum()
    pct_col = (faltantes_col / n_filas) * 100
    df_diag = pd.DataFrame({
        "Variable": variables_modelo,
        "Faltantes": [int(faltantes_col[v]) for v in variables_modelo],
        "% Faltante": [float(pct_col[v].round(2)) for v in variables_modelo],
        "Mín Histórico 2018": [metadata["rangos_historicos_2018"][v]["min"] for v in variables_modelo],
        "Máx Histórico 2018": [metadata["rangos_historicos_2018"][v]["max"] for v in variables_modelo]
    })

    # Gráfica de diagnóstico interactiva en Plotly con paleta Viridis
    if PLOTLY_INSTALLED and total_faltantes > 0:
        fig_diag = px.bar(
            df_diag.sort_values("% Faltante", ascending=True),
            x="% Faltante",
            y="Variable",
            orientation="h",
            color="% Faltante",
            color_continuous_scale="Viridis",
            title="Porcentaje de Valores Faltantes por Sensor Meteorológico",
            labels={"% Faltante": "% de Faltantes", "Variable": "Sensor"},
            text="% Faltante"
        )
        fig_diag.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_diag.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_colorbar=dict(title="%"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_diag, use_container_width=True)

    with st.expander("🔍 Ver tabla detallada de diagnóstico por sensor"):
        st.dataframe(df_diag.set_index("Variable").T, use_container_width=True)

    # ==========================================
    # SECCIÓN 3: RECONSTRUCCIÓN CON F-KNN
    # ==========================================
    st.markdown("### 3. Reconstrucción con F-KNN")

    if total_faltantes == 0:
        st.success("🎉 El dataset no presenta valores faltantes. No se requiere imputación.")
        st.session_state["df_imputado"] = df_preparado.copy()
    else:
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            ejecutar = st.button("🚀 Ejecutar Imputación con Pipeline F-KNN", type="primary", use_container_width=True)

        if ejecutar:
            with st.spinner("Reconstruyendo series de tiempo con F-KNN..."):
                try:
                    X_array = df_preparado.to_numpy(dtype=float)
                    X_imputado = pipeline.transform(X_array)
                    
                    df_imputado = pd.DataFrame(
                        X_imputado,
                        index=df_preparado.index,
                        columns=variables_modelo
                    )
                    st.session_state["df_imputado"] = df_imputado
                    st.rerun()
                except Exception as ex:
                    st.error(f"Error durante la imputación: {ex}")

    # ==========================================
    # SECCIÓN 4 Y 5: VISUALIZACIÓN Y DESCARGA
    # ==========================================
    if st.session_state["df_imputado"] is not None:
        df_res = st.session_state["df_imputado"]
        st.success(f"✅ ¡Imputación completada con éxito! Se reconstruyeron {total_faltantes:,} celdas.")

        st.markdown("### 4. Inspección visual interactiva (Plotly — Paleta Viridis)")
        variable_seleccionada = st.selectbox(
            "Selecciona la variable que deseas inspeccionar:",
            variables_modelo
        )

        serie_original = df_preparado[variable_seleccionada]
        serie_imputada = df_res[variable_seleccionada]
        mascara_faltantes = serie_original.isna()

        if PLOTLY_INSTALLED:
            fig = go.Figure()

            # 1. Serie Reconstruida continua (F-KNN) usando tono Teal/Verde Viridis
            fig.add_trace(go.Scatter(
                x=serie_imputada.index,
                y=serie_imputada.values,
                mode="lines",
                name="Serie Completa (F-KNN)",
                line=dict(color=VIRIDIS_TEAL, width=2),
                hovertemplate="<b>Fecha:</b> %{x|%d/%m/%Y %H:%M}<br><b>Valor:</b> %{y:.2f}<extra></extra>"
            ))

            # 2. Celdas Imputadas usando Amarillo Viridis brillante con borde Púrpura
            if mascara_faltantes.any():
                fig.add_trace(go.Scatter(
                    x=serie_imputada.index[mascara_faltantes],
                    y=serie_imputada.values[mascara_faltantes],
                    mode="markers",
                    name=f"Valores Imputados ({mascara_faltantes.sum()} celdas)",
                    marker=dict(
                        color=VIRIDIS_YELLOW,
                        size=8,
                        symbol="circle",
                        line=dict(color=VIRIDIS_DARK_PURPLE, width=1.5)
                    ),
                    hovertemplate="<b>⚠️ Imputado F-KNN</b><br><b>Fecha:</b> %{x|%d/%m/%Y %H:%M}<br><b>Valor Estimado:</b> %{y:.2f}<extra></extra>"
                ))

            fig.update_layout(
                title=dict(
                    text=f"Reconstrucción Temporal de <b>{variable_seleccionada}</b>",
                    font=dict(size=15)
                ),
                xaxis=dict(
                    title="Fecha y Hora",
                    rangeslider=dict(visible=True, thickness=0.06),
                    type="date"
                ),
                yaxis=dict(
                    title=f"Valor ({variable_seleccionada})",
                    gridcolor="rgba(128,128,128,0.15)"
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified",
                template="plotly_white",
                height=480,
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No se pudo renderizar la gráfica interactiva porque `plotly` no está instalado en el entorno.")

        # Descarga
        st.markdown("### 5. Descargar datos limpios e imputados")
        csv_bytes = df_res.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Descargar CSV Imputado",
            data=csv_bytes,
            file_name="dataset_meteorologico_imputado_fknn.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    st.info("👆 Por favor sube un archivo CSV o haz clic en **'Cargar datos de ejemplo'** en la barra lateral para comenzar.")
