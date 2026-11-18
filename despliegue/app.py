"""
Aplicación Web Interactiva en Streamlit para el Despliegue de F-KNN
Imputación de Series de Tiempo Meteorológicas — Aranzazu, Caldas
Gráficas Plotly con paleta Viridis.
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

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y RUTAS
# ==========================================
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

# Paleta Viridis
VIRIDIS_DARK_PURPLE = "#440154"
VIRIDIS_BLUE = "#3b528b"
VIRIDIS_TEAL = "#21918c"
VIRIDIS_GREEN = "#5ec962"
VIRIDIS_YELLOW = "#fde725"

# ==========================================
# ESTILOS (CSS)
# ==========================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');

:root{
    --texto:#e9eefb;
    --suave:#a9b6d6;
    --sol:#f4c95d;
    --panel:rgba(12,18,45,.58);
    --borde:rgba(255,255,255,.13);
}

/* ---------- Fondo: cielo de noche a atardecer ---------- */
.stApp{
    background:linear-gradient(180deg,#0c1230 0%,#19244f 45%,#2a3f6b 80%,#3b5a73 100%);
    background-attachment:fixed;
    font-family:'DM Sans',sans-serif;
}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}

/* ---------- Estrellas ---------- */
.stApp::before, .stApp::after{
    content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
    -webkit-mask-image:linear-gradient(#000 20%,transparent 75%);
    mask-image:linear-gradient(#000 20%,transparent 75%);
}
.stApp::before{
    background-image:
        radial-gradient(1.5px 1.5px at 40px 60px,#fff,transparent),
        radial-gradient(1px 1px at 130px 120px,#f4e3a8,transparent),
        radial-gradient(1.5px 1.5px at 210px 40px,#fff,transparent),
        radial-gradient(1px 1px at 90px 200px,#fff,transparent),
        radial-gradient(1px 1px at 260px 170px,#cfe0ff,transparent);
    background-size:320px 320px;
    animation:titilar 8s ease-in-out infinite alternate;
}
.stApp::after{
    background-image:
        radial-gradient(1px 1px at 70px 90px,#fff,transparent),
        radial-gradient(1.5px 1.5px at 300px 150px,#f4e3a8,transparent),
        radial-gradient(1px 1px at 150px 330px,#fff,transparent),
        radial-gradient(1px 1px at 380px 40px,#cfe0ff,transparent);
    background-size:450px 450px;
    animation:titilar 11s ease-in-out -3s infinite alternate;
}
@keyframes titilar{from{opacity:.2}to{opacity:.85}}

/* ---------- Nubes ---------- */
.nube{
    position:fixed;width:200px;height:56px;border-radius:100px;
    background:rgba(255,255,255,.07);pointer-events:none;z-index:1;scale:var(--s,1);
    animation:derivar linear infinite;
}
.nube::before,.nube::after{content:"";position:absolute;background:inherit;border-radius:50%;}
.nube::before{width:90px;height:90px;top:-44px;left:28px;}
.nube::after{width:120px;height:120px;top:-62px;right:26px;}
@keyframes derivar{from{transform:translateX(-380px)}to{transform:translateX(112vw)}}

/* ---------- Contenedor ---------- */
.block-container{position:relative;z-index:2;max-width:1150px;padding-top:2.2rem;padding-bottom:2rem;}

/* ---------- Color de texto de los elementos nativos de Streamlit ---------- */
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong,
.stApp [data-testid="stMarkdownContainer"] em,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3,
.stApp [data-testid="stMarkdownContainer"] h4,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label,
.stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stFileUploader"] *,
.stApp [data-testid="stExpander"] summary *,
.stApp [data-testid="stSpinner"] *,
.stApp [data-testid="stAlert"] *,
.stApp [data-testid="stTable"] *{
    color:var(--texto);
}

/* ---------- Encabezado principal ---------- */
.hero{
    position:relative;padding:2.2rem 2.4rem 2rem 2.4rem;margin-bottom:1.5rem;
    background:var(--panel);border:1px solid var(--borde);border-radius:12px;
}
.hero-sol{position:absolute;top:34px;right:56px;width:84px;height:84px;}
.rayos{
    position:absolute;inset:-30px;border-radius:50%;
    background:repeating-conic-gradient(from 0deg,rgba(244,201,93,.45) 0deg 3deg,transparent 3deg 18deg);
    -webkit-mask-image:radial-gradient(circle,#000 35%,transparent 70%);
    mask-image:radial-gradient(circle,#000 35%,transparent 70%);
    animation:girar 90s linear infinite;
}
.sol{
    position:absolute;inset:0;border-radius:50%;
    background:radial-gradient(circle at 38% 36%,#fbe39a,#f4c95d 60%,#e2a93b);
    box-shadow:0 0 50px 8px rgba(244,201,93,.28);
}
@keyframes girar{to{transform:rotate(360deg)}}
.kicker{font-size:.78rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--sol);}
.stApp .hero h1{
    font-family:'Fraunces',serif;font-weight:600;font-size:2.35rem;line-height:1.15;
    color:#fff;margin:.6rem 0 .8rem 0;padding:0;max-width:640px;
}
.stApp .hero p{color:#cdd7ee;font-size:1rem;line-height:1.6;max-width:640px;margin:0;}
@media (max-width:760px){.hero-sol{display:none;}}

/* ---------- Secciones ---------- */
.sec{margin:2.4rem 0 1rem 0;padding-bottom:.6rem;border-bottom:1px solid var(--borde);}
.sec-top{display:flex;align-items:baseline;gap:.9rem;}
.sec-n{font-family:'Fraunces',serif;font-size:1.05rem;color:var(--sol);}
.sec-t{font-family:'Fraunces',serif;font-weight:600;font-size:1.45rem;color:#fff;}
.sec-s{font-size:.9rem;color:var(--suave);margin-top:.25rem;}

/* ---------- Indicadores ---------- */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.9rem;margin-bottom:1.2rem;}
.kpi{
    padding:1rem 1.2rem;background:var(--panel);border:1px solid var(--borde);
    border-left:3px solid var(--sol);border-radius:8px;
}
.kpi-l{font-size:.74rem;font-weight:700;color:var(--suave);text-transform:uppercase;letter-spacing:.09em;}
.kpi-v{font-family:'Fraunces',serif;font-weight:500;font-size:1.9rem;color:#fff;margin-top:.25rem;}
.kpi-v.sm{font-size:1.1rem;line-height:2.4rem;}

.archivo{
    margin:.3rem 0 .8rem 0;padding:.5rem 1rem;border-left:3px solid var(--sol);
    background:var(--panel);border-radius:0 8px 8px 0;color:var(--texto);font-size:.92rem;
}

/* ---------- Widgets ---------- */
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"]{
    background:var(--panel);border:1.5px dashed rgba(244,201,93,.55);border-radius:10px;
}
[data-testid="stFileUploaderFile"]{background:rgba(255,255,255,.08);border-radius:8px;}
[data-testid="stAlert"]{
    background:rgba(10,16,40,.75) !important;border:1px solid var(--borde);border-radius:10px;
}
[data-testid="stExpander"]{background:var(--panel);border:1px solid var(--borde);border-radius:10px;}
[data-testid="stExpander"] details{border:none;background:transparent;}
[data-testid="stPlotlyChart"]{
    background:var(--panel);border:1px solid var(--borde);border-radius:12px;padding:.5rem;
}

/* Selector desplegable */
[data-baseweb="select"] > div{
    background:rgba(12,18,45,.75);border:1px solid rgba(255,255,255,.22);border-radius:8px;
}
[data-baseweb="select"] *{color:var(--texto) !important;}
[data-baseweb="popover"] ul, [data-baseweb="popover"] [data-baseweb="menu"]{background:#121a3d !important;}
[data-baseweb="popover"] li, [data-baseweb="popover"] li *{color:var(--texto) !important;}
[data-baseweb="popover"] li:hover{background:rgba(255,255,255,.1) !important;}

/* Tabla */
.stApp [data-testid="stTable"] table{background:transparent;border-collapse:collapse;width:100%;}
.stApp [data-testid="stTable"] th{background:rgba(255,255,255,.08);}
.stApp [data-testid="stTable"] th, .stApp [data-testid="stTable"] td{border-bottom:1px solid var(--borde);}

/* Botones */
.stButton button, .stDownloadButton button{
    border-radius:8px;font-weight:700;padding:.55rem 1rem;
    background:transparent;color:var(--texto);border:1px solid rgba(255,255,255,.35);
}
.stButton button:hover{border-color:var(--sol);color:var(--sol);}
button[kind="primary"], [data-testid="stBaseButton-primary"], .stDownloadButton button{
    background:var(--sol);color:#10162f;border:1px solid var(--sol);
}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover, .stDownloadButton button:hover{
    background:#f8d98a;color:#10162f;border-color:#f8d98a;
}
.stButton button p, .stDownloadButton button p{color:inherit !important;}

/* ---------- Barra lateral ---------- */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0b1129 0%,#142046 70%,#18334a 100%);
    border-right:1px solid var(--borde);
}
.lado-t{font-family:'Fraunces',serif;font-size:1.25rem;font-weight:600;color:#fff;}
.lado-s{font-size:.82rem;color:var(--suave);margin-bottom:1.1rem;}
.lado-h{
    font-size:.72rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
    color:var(--sol);margin:1.2rem 0 .4rem 0;
}
.lado-p{font-size:.88rem;color:var(--texto);line-height:1.5;}
.fila{display:flex;justify-content:space-between;gap:.8rem;font-size:.84rem;padding:.28rem 0;border-bottom:1px solid var(--borde);}
.fila .k{color:var(--suave);}
.fila .v{color:var(--texto);text-align:right;}
[data-testid="stSidebar"] [data-testid="stImage"] img{background:#fff;border-radius:8px;padding:8px;}

/* ---------- Plantas al pie ---------- */
.paisaje{width:100%;height:120px;display:block;margin-top:3rem;}
.sway{transform-box:fill-box;transform-origin:50% 100%;animation:mecer 6s ease-in-out infinite alternate;}
@keyframes mecer{from{transform:rotate(-2.5deg)}to{transform:rotate(2.5deg)}}
.pie{text-align:center;color:#c9d8e6;font-size:.82rem;padding:.6rem 0 .2rem 0;}

/* ==========================================
   CORRECCIONES DE FONDOS Y CONTRASTE
   ========================================== */

/* Mantener transparentes los contenedores generales de Streamlit */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"] {
    background-color: transparent !important;
}

/* Área de carga de archivos */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"] {
    background: rgba(12,18,45,.72) !important;
    color: #e9eefb !important;
    border: 1.5px dashed rgba(244,201,93,.65) !important;
    border-radius: 10px !important;
}

/* Botón Upload: amarillo Viridis con texto oscuro */
[data-testid="stFileUploader"] button {
    background: #fde725 !important;
    color: #10162f !important;
    border: 1px solid #fde725 !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploader"] button * {
    color: #10162f !important;
    fill: #10162f !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #d5e51e !important;
    border-color: #d5e51e !important;
}

/* Texto secundario del cargador */
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: #cdd7ee !important;
}

/* Inputs de texto y numéricos, si se usan */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stTextArea textarea {
    background-color: rgba(12,18,45,.85) !important;
    color: #e9eefb !important;
    border: 1px solid rgba(255,255,255,.2) !important;
}

/* Tablas */
[data-testid="stTable"],
[data-testid="stDataFrame"] {
    background-color: rgba(12,18,45,.65) !important;
    color: #e9eefb !important;
}

[data-testid="stTable"] table,
[data-testid="stTable"] th,
[data-testid="stTable"] td {
    background-color: transparent !important;
    color: #e9eefb !important;
    border-color: rgba(255,255,255,.15) !important;
}

/* Métricas, en caso de incorporarlas */
[data-testid="stMetric"] {
    background-color: rgba(12,18,45,.58) !important;
    border: 1px solid rgba(255,255,255,.13) !important;
    border-radius: 10px;
    padding: 12px;
}

[data-testid="stMetric"] * {
    color: #e9eefb !important;
}

/* Expanders */
[data-testid="stExpander"],
[data-testid="stExpander"] details {
    background-color: rgba(12,18,45,.65) !important;
    border-color: rgba(255,255,255,.15) !important;
}

/* Mensajes de alerta nativos */
[data-testid="stAlert"] {
    background-color: rgba(10,16,40,.85) !important;
    color: #e9eefb !important;
    border: 1px solid rgba(255,255,255,.15) !important;
}

/* ==========================================
   FORZAR TEMA OSCURO EN COMPONENTES STREAMLIT
   ========================================== */

/* Alertas: info, success, warning y error */
[data-testid="stAlert"],
[data-testid="stAlert"] > div,
[data-testid="stAlert"] div[role="alert"] {
    background: rgba(12,18,45,.92) !important;
    color: #e9eefb !important;
    border: 1px solid rgba(255,255,255,.18) !important;
}

/* Cargador CSV: todos sus contenedores */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div,
[data-testid="stFileUploaderDropzone"] > section {
    background: rgba(12,18,45,.85) !important;
    color: #e9eefb !important;
}

/* Botón interno del cargador */
[data-testid="stFileUploader"] button {
    background: #fde725 !important;
    color: #10162f !important;
}

/* Texto e iconos del cargador */
[data-testid="stFileUploader"] svg {
    color: #cdd7ee !important;
}

/* Mensajes y contenedores */
.stApp [data-testid="stMarkdownContainer"] {
    color: #e9eefb;
}

/* Eliminar fondos blancos de contenedores BaseWeb */
[data-baseweb="notification"],
[data-baseweb="card"] {
    background: rgba(12,18,45,.90) !important;
    color: #e9eefb !important;
}

/* Fondo general */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: transparent !important;
}


/* ==========================================
   MENSAJE PERSONALIZADO CUANDO NO HAY ARCHIVO
   ========================================== */
.aviso-vacio {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.2rem 1.5rem;
    margin-top: .8rem;
    background: rgba(12,18,45,.72);
    border: 1px solid rgba(94,201,98,.45);
    border-left: 4px solid #5ec962;
    border-radius: 10px;
    color: #e9eefb;
    font-size: .95rem;
    line-height: 1.5;
}

.aviso-vacio b {
    color: #fde725;
}

.aviso-icono {
    font-size: 1.6rem;
    color: #5ec962;
}
"""


def limpiar_html(texto):
    """Quita sangrías y líneas vacías para que Markdown no lo interprete como bloque de código."""
    return "\n".join(l.strip() for l in texto.splitlines() if l.strip())


def inyectar_estilos():
    st.markdown(f"<style>{limpiar_html(CSS)}</style>", unsafe_allow_html=True)


def nubes_de_fondo():
    """Tres nubes tenues que cruzan la pantalla lentamente."""
    nubes = [
        "top:12%;--s:1.2;animation-duration:110s;animation-delay:-20s;",
        "top:30%;--s:0.8;animation-duration:140s;animation-delay:-80s;",
        "top:52%;--s:1.5;animation-duration:170s;animation-delay:-50s;",
    ]
    html = "".join(f'<div class="nube" style="{estilo}"></div>' for estilo in nubes)
    st.markdown(html, unsafe_allow_html=True)


def paisaje_plantas():
    """SVG con colinas y plantas que se mecen suavemente."""
    plantas = []
    for k, x in enumerate(range(40, 1200, 85)):
        h = 36 + (k * 17) % 34
        y0, yc, ym = 120, 120 - h, 120 - h * 0.55
        tallo, hoja, punta = ("#2f6b4f", "#3f8a5f", "#5aa578") if k % 2 else ("#285c45", "#357852", "#4d9468")
        retraso = -(k % 5) * 1.1
        plantas.append(
            f'<g class="sway" style="animation-delay:{retraso:.1f}s">'
            f'<path d="M{x} {y0} Q{x + 7} {y0 - h / 2:.0f} {x} {yc:.0f}" stroke="{tallo}" '
            f'stroke-width="3" fill="none" stroke-linecap="round"/>'
            f'<ellipse cx="{x - 10}" cy="{ym:.0f}" rx="10" ry="4.5" fill="{hoja}" '
            f'transform="rotate(-35 {x - 10} {ym:.0f})"/>'
            f'<ellipse cx="{x + 10}" cy="{ym - 8:.0f}" rx="10" ry="4.5" fill="{hoja}" '
            f'transform="rotate(35 {x + 10} {ym - 8:.0f})"/>'
            f'<ellipse cx="{x}" cy="{yc - 6:.0f}" rx="4.5" ry="10" fill="{punta}"/>'
            f'</g>'
        )
    svg = (
        '<svg class="paisaje" viewBox="0 0 1200 120" preserveAspectRatio="xMidYMax slice" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<path d="M0 120 L0 80 Q150 50 320 76 T640 70 T960 64 T1200 74 L1200 120Z" fill="#14333a"/>'
        + "".join(plantas) +
        '<path d="M0 120 L0 106 Q200 96 420 108 T820 104 T1200 108 L1200 120Z" fill="#0d262b"/>'
        '</svg>'
    )
    st.markdown(svg, unsafe_allow_html=True)


def seccion(numero, titulo, subtitulo=""):
    st.markdown(
        f'<div class="sec"><div class="sec-top"><span class="sec-n">{numero}</span>'
        f'<span class="sec-t">{titulo}</span></div>'
        f'<div class="sec-s">{subtitulo}</div></div>',
        unsafe_allow_html=True
    )


def tarjetas_kpi(items):
    """items: lista de (etiqueta, valor)."""
    cuerpo = "".join(
        f'<div class="kpi"><div class="kpi-l">{lab}</div>'
        f'<div class="kpi-v{" sm" if len(str(val)) > 14 else ""}">{val}</div></div>'
        for lab, val in items
    )
    st.markdown(f'<div class="kpis">{cuerpo}</div>', unsafe_allow_html=True)


def fila(clave, valor):
    return f'<div class="fila"><span class="k">{clave}</span><span class="v">{valor}</span></div>'


def estilo_plotly(fig, altura):
    """Aplica fondo transparente y texto claro a cualquier figura de Plotly."""
    fig.update_layout(
        template="plotly_dark",
        height=altura,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#e9eefb"),
        margin=dict(l=30, r=30, t=60, b=30),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig


inyectar_estilos()
nubes_de_fondo()

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
    st.markdown(
        '<div class="lado-t">Panel de control</div>'
        '<div class="lado-s">Estación meteorológica de Aranzazu (Caldas)</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="lado-h">Autores</div>'
        '<div class="lado-p">Gabriel Castro Benítez<br>Karen Daniela Torres Henao<br>'
        '<span style="color:#a9b6d6">Machine Learning 1 · Pregrado en Ciencia de Datos<br>'
        'Universidad Externado de Colombia</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="lado-h">Modelo</div>'
        + fila("Algoritmo", "F-KNN")
        + fila("Hiperparámetros", metadata["configuracion"])
        + fila("Frecuencia", metadata["frecuencia"]),
        unsafe_allow_html=True
    )

    m_dis = metadata["metricas_prueba_2019"]["disjoint"]
    m_ov = metadata["metricas_prueba_2019"]["overlap"]
    st.markdown(
        '<div class="lado-h">Métricas en prueba 2019</div>'
        + fila("Disjoint · R²", m_dis["R2_medio"])
        + fila("Disjoint · MSE", m_dis["MSE_medio"])
        + fila("Overlap · R²", m_ov["R2_medio"])
        + fila("Overlap · MSE", m_ov["MSE_medio"]),
        unsafe_allow_html=True
    )

    st.markdown('<div class="lado-h">Datos</div>', unsafe_allow_html=True)
    if st.button("Cargar datos de ejemplo (semana 2019)", use_container_width=True):
        ejemplo_path = DATA_DIR / "ejemplo_para_probar.csv"
        if ejemplo_path.exists():
            st.session_state["df_crudo"] = pd.read_csv(ejemplo_path, index_col=0, parse_dates=True)
            st.session_state["origen_datos"] = "Dataset de ejemplo (Semana 2019 con 20 % de faltantes)"
            st.session_state["df_imputado"] = None
            st.rerun()
        else:
            st.error("No se encontró el archivo 'data/ejemplo_para_probar.csv'.")

    if st.session_state["df_crudo"] is not None:
        if st.button("Limpiar / cargar otro archivo", use_container_width=True):
            st.session_state["df_crudo"] = None
            st.session_state["origen_datos"] = None
            st.session_state["df_imputado"] = None
            st.rerun()

    st.markdown('<div class="lado-h">Transparencia</div>', unsafe_allow_html=True)
    with st.expander("Uso declarado de IA"):
        st.markdown("""
Durante el diseño, estructuración e implementación de este módulo de despliegue, **utilizamos** la siguiente herramienta de apoyo tecnológico bajo nuestra supervisión directa:

* **Antigravity (Google DeepMind, 2026):** Empleado como asistente de ingeniería de software para la modularización del estimador `FocalizedKNNImputer`, la estructuración del framework de despliegue en Streamlit, el control de persistencia de estado (`session_state`) y la construcción de las visualizaciones interactivas en Plotly con paleta Viridis.

La validación experimental, la verificación del código, la interpretación analítica de los resultados y la autoría íntegra del contenido técnico recaen exclusivamente sobre nosotros como autores del proyecto.
        """)

    st.markdown("---")

    # Logos institucionales
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
# ENCABEZADO
# ==========================================
st.markdown(limpiar_html("""
<div class="hero">
<div class="hero-sol"><div class="rayos"></div><div class="sol"></div></div>
<div class="kicker">Estación meteorológica · Aranzazu, Caldas</div>
<h1>Imputación de datos faltantes en sensores meteorológicos</h1>
<p>Esta herramienta reconstruye los valores que los sensores dejaron de registrar. Usa el modelo
F-KNN (Focalized K-Nearest Neighbors), que busca registros parecidos a partir de las variables
más correlacionadas y de los rezagos de la propia serie.</p>
</div>
"""), unsafe_allow_html=True)

if not PLOTLY_INSTALLED:
    st.warning("La librería `plotly` no está instalada. Para ver las gráficas interactivas ejecuta: `pip install plotly`.")

# ==========================================
# SECCIÓN 1: CARGA DE ARCHIVO
# ==========================================
seccion("01", "Cargar archivo", "Sube el CSV con las series de tiempo o usa el ejemplo del panel lateral")

archivo_subido = st.file_uploader(
    "Archivo CSV con las series de tiempo (frecuencia recomendada de 30 minutos):",
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
    st.markdown(
        f'<div class="archivo">Archivo en uso: <b>{st.session_state["origen_datos"]}</b></div>',
        unsafe_allow_html=True
    )

    try:
        df_preparado = preparar_dataframe_entrada(df_crudo, variables_modelo)
    except Exception as err:
        st.error(f"Error de validación en los datos: {err}")
        st.stop()

    n_filas = len(df_preparado)
    total_celdas = df_preparado.size
    total_faltantes = int(df_preparado.isna().sum().sum())
    pct_faltantes_global = (total_faltantes / total_celdas) * 100

    seccion("02", "Diagnóstico de valores faltantes", "Qué tan completos llegaron los datos de cada sensor")

    rango_txt = (
        f"{df_preparado.index.min().strftime('%d/%m/%Y')} → "
        f"{df_preparado.index.max().strftime('%d/%m/%Y')}"
    )
    tarjetas_kpi([
        ("Observaciones", f"{n_filas:,}"),
        ("Rango temporal", rango_txt),
        ("Celdas faltantes", f"{total_faltantes:,}"),
        ("% faltantes global", f"{pct_faltantes_global:.2f} %"),
    ])

    faltantes_col = df_preparado.isna().sum()
    pct_col = (faltantes_col / n_filas) * 100
    df_diag = pd.DataFrame({
        "Variable": variables_modelo,
        "Faltantes": [int(faltantes_col[v]) for v in variables_modelo],
        "% Faltante": [float(pct_col[v].round(2)) for v in variables_modelo],
        "Mín Histórico 2018": [metadata["rangos_historicos_2018"][v]["min"] for v in variables_modelo],
        "Máx Histórico 2018": [metadata["rangos_historicos_2018"][v]["max"] for v in variables_modelo]
    })

    if PLOTLY_INSTALLED and total_faltantes > 0:
        fig_diag = px.bar(
            df_diag.sort_values("% Faltante", ascending=True),
            x="% Faltante",
            y="Variable",
            orientation="h",
            color="% Faltante",
            color_continuous_scale="Viridis",
            title="Porcentaje de valores faltantes por sensor",
            labels={"% Faltante": "% de faltantes", "Variable": "Sensor"},
            text="% Faltante"
        )
        fig_diag.update_traces(
            texttemplate='%{text:.1f}%', textposition='outside',
            marker_line_width=0, cliponaxis=False
        )
        estilo_plotly(fig_diag, 420)
        fig_diag.update_layout(coloraxis_showscale=False)
        fig_diag.update_xaxes(range=[0, max(df_diag["% Faltante"].max() * 1.25, 1)])
        st.plotly_chart(fig_diag, use_container_width=True)

    with st.expander("Ver tabla detallada por sensor"):
        st.table(df_diag.set_index("Variable"))

    # ==========================================
    # SECCIÓN 3: RECONSTRUCCIÓN CON F-KNN
    # ==========================================
    seccion("03", "Reconstrucción con F-KNN", "Rellena los huecos con el pipeline entrenado con los datos de 2018")

    if total_faltantes == 0:
        st.success("El dataset no presenta valores faltantes. No se requiere imputación.")
        st.session_state["df_imputado"] = df_preparado.copy()
    else:
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            ejecutar = st.button("Ejecutar imputación con F-KNN", type="primary", use_container_width=True)

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
        st.success(f"Imputación completada. Se reconstruyeron {total_faltantes:,} celdas.")

        seccion("04", "Inspección visual", "Los puntos amarillos son los valores estimados por F-KNN")

        variable_seleccionada = st.selectbox(
            "Variable a inspeccionar:",
            variables_modelo
        )

        serie_original = df_preparado[variable_seleccionada]
        serie_imputada = df_res[variable_seleccionada]
        mascara_faltantes = serie_original.isna()

        if PLOTLY_INSTALLED:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=serie_imputada.index,
                y=serie_imputada.values,
                mode="lines",
                name="Serie completa (F-KNN)",
                line=dict(color=VIRIDIS_GREEN, width=2),
                hovertemplate="<b>Fecha:</b> %{x|%d/%m/%Y %H:%M}<br><b>Valor:</b> %{y:.2f}<extra></extra>"
            ))

            if mascara_faltantes.any():
                fig.add_trace(go.Scatter(
                    x=serie_imputada.index[mascara_faltantes],
                    y=serie_imputada.values[mascara_faltantes],
                    mode="markers",
                    name=f"Valores imputados ({int(mascara_faltantes.sum())} celdas)",
                    marker=dict(
                        color=VIRIDIS_YELLOW,
                        size=8,
                        symbol="circle",
                        line=dict(color=VIRIDIS_DARK_PURPLE, width=1.5)
                    ),
                    hovertemplate="<b>Imputado F-KNN</b><br><b>Fecha:</b> %{x|%d/%m/%Y %H:%M}<br><b>Valor estimado:</b> %{y:.2f}<extra></extra>"
                ))

            estilo_plotly(fig, 520)
            fig.update_layout(
                title=dict(
                    text=f"Reconstrucción temporal de <b>{variable_seleccionada}</b>",
                    font=dict(size=16)
                ),
                xaxis=dict(
                    title="Fecha y hora",
                    rangeslider=dict(visible=True, thickness=0.06, bgcolor="rgba(255,255,255,0.05)"),
                    type="date"
                ),
                yaxis=dict(title=f"Valor ({variable_seleccionada})"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No se pudo renderizar la gráfica interactiva porque `plotly` no está instalado en el entorno.")

        # Descarga
        seccion("05", "Descargar resultados", "Archivo CSV con la serie completa reconstruida")
        csv_bytes = df_res.to_csv().encode("utf-8")
        st.download_button(
            label="Descargar CSV imputado",
            data=csv_bytes,
            file_name="dataset_meteorologico_imputado_fknn.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    st.markdown("""
    <div class="aviso-vacio">
        <span class="aviso-icono">☁</span>
        <span>
            Sube un archivo CSV o usa <b>Cargar datos de ejemplo</b>
            en el panel lateral para comenzar.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PIE DE PÁGINA
# ==========================================
paisaje_plantas()
st.markdown(
    '<div class="pie">Universidad Externado de Colombia · Ciencia de Datos · Machine Learning 1 · 2026</div>',
    unsafe_allow_html=True
)
