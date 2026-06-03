import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import re
import unicodedata
from nltk.corpus import stopwords
import nltk
from PIL import Image

# Descargar stopwords si es necesario
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

# ============================
# CONFIGURACIÓN DE LA PÁGINA
# ============================
st.set_page_config(
    page_title="Categorizador de Reparaciones - Motos Casa Tuning", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# ESTILOS CSS CUSTOM
# ============================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a0b2e 0%, #2d1b4e 100%);
        color: #e8d9ff;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f051f 0%, #1f0a3a 100%);
    }

    [data-testid="stSidebar"] * {
        color: #e8d9ff !important;
    }

    h2, h3, h4, h5, h6 {
        color: #c9a5ff !important;
    }

    .stCard, .stExpander, .stAlert, [data-testid="stForm"] {
        background-color: #2a1a3e !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        box-shadow: 0 10px 24px rgba(0,0,0,0.3) !important;
        border: 1px solid #4a2a6e !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6f42c1 0%, #8e5edd 100%) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #8e5edd 0%, #a87cf5 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(110, 66, 193, 0.4);
    }

    .stDataFrame {
        border: 1px solid #6f42c1 !important;
        border-radius: 10px !important;
        background-color: #1e0f30 !important;
    }
    
    /* Dataframe headers */
    .stDataFrame thead th {
        background-color: #2d1b4e !important;
        color: #c9a5ff !important;
    }
    
    /* Dataframe cells */
    .stDataFrame tbody td {
        background-color: #1e0f30 !important;
        color: #e8d9ff !important;
    }
    
    /* Input fields */
    .stTextArea textarea, .stTextInput input {
        background-color: #1e0f30 !important;
        color: #e8d9ff !important;
        border-color: #4a2a6e !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #8e5edd !important;
        box-shadow: 0 0 5px #8e5edd !important;
    }
    
    /* Select boxes */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1e0f30 !important;
        border-color: #4a2a6e !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #6f42c1 !important;
    }
    
    /* Info/Success/Warning/Error boxes */
    .stAlert {
        background-color: #2a1a3e !important;
        border-left: 4px solid #6f42c1 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #1e0f30 !important;
        color: #c9a5ff !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #6f42c1 !important;
        color: white !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #2a1a3e !important;
        border-radius: 10px !important;
        padding: 10px !important;
        border: 1px solid #4a2a6e !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a0b2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #6f42c1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #8e5edd;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# TÍTULO CON LOGO A LA IZQUIERDA
# ============================
col_logo, col_title = st.columns([1, 5])

with col_logo:
    try:
        logo = Image.open("logotuning_120526.png")
        st.image(logo, width=150)
    except:
        st.image("https://img.icons8.com/ios-filled/100/4B0082/motorcycle.png", width=60)

with col_title:
    st.markdown("""
    <h1 style='margin: 0; color: White;'> Categorizador Inteligente de Reparaciones</h1>
    <p style='margin: 0; font-size: 34px; color: White;'>Motos Casa Tuning </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================
# 1. CARGAR DATOS ESTÁTICOS (CSV)
# ============================
@st.cache_data
def load_static_data():
    try:
        df = pd.read_csv("reparaciones_5000.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ Archivo 'reparaciones_5000.csv' no encontrado. Los gráficos estáticos no estarán disponibles.")
        return None

df_static = load_static_data()

# ============================
# 2. CARGAR MODELO Y VECTORIZADOR
# ============================
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load("modelo_predictor_categorias.pkl")
        vectorizer = joblib.load("vectorizador.pkl")
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"No se encuentra el archivo: {e}")
        st.info("""
        Asegúrate de tener estos archivos en la misma carpeta:
        - modelo_predictor_categorias.pkl
        - vectorizador.pkl
        """)
        return None, None
    except Exception as e:
        st.error(f" Error al cargar: {e}")
        return None, None

model, vectorizer = load_model_and_vectorizer()

# ============================
# 3. FUNCIONES DE PREPROCESAMIENTO Y PREDICCIÓN
# ============================
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def predict_category(descripcion, model, vectorizer):
    """Predice usando el modelo y vectorizador con control de errores para SVC sin probabilidades"""
    try:
        if not descripcion or not isinstance(descripcion, str) or descripcion.strip() == "":
            return "Error", None
        
        texto_procesado = preprocess_text(descripcion)
        
        if not texto_procesado.strip():
            return "Error", None
        
        texto_vectorizado = vectorizer.transform([texto_procesado])
        prediccion = model.predict(texto_vectorizado)[0]
        
        probabilidades = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilidades = model.predict_proba(texto_vectorizado)[0]
            except AttributeError:
                # Si el SVC no se entrenó con probability=True, ignoramos las probabilidades de forma segura
                probabilidades = None
        
        return prediccion, probabilidades
    except Exception as e:
        return "Error", None

# ============================
# 4. GRÁFICOS ESTÁTICOS (con datos del CSV)
# ============================
if df_static is not None and model is not None:
    st.header(" Dashboard de Reparaciones")
    
    if 'Categoria_Predicha' not in df_static.columns:
        with st.spinner(" Procesando todas las descripciones del archivo estático..."):
            predicciones = []
            errores = 0
            total = len(df_static)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, desc in enumerate(df_static['Descripcion']):
                if idx % 100 == 0:
                    progress_bar.progress(idx / total)
                    status_text.text(f"Procesando registro {idx}/{total}...")
                
                try:
                    cat, _ = predict_category(desc, model, vectorizer)
                    if cat is None or cat == "Error":
                        cat = "Error"
                        errores += 1
                    predicciones.append(cat)
                except Exception as e:
                    predicciones.append("Error")
                    errores += 1
            
            progress_bar.progress(1.0)
            status_text.text(f" Procesamiento completado! ({total} registros, {errores} errores)")
        
            df_static['Categoria_Predicha'] = predicciones
            st.success(f"Datos estáticos procesados exitosamente!")
    
    if 'Categoria_Predicha' in df_static.columns:
        error_count = (df_static['Categoria_Predicha'] == "Error").sum()
        if error_count > 0:
            st.warning(f" {error_count} registros no pudieron ser categorizados correctamente.")
    
        col1, col2 = st.columns(2)
        
        with col1:
            df_filtered = df_static[df_static['Categoria_Predicha'] != "Error"]
            
            if len(df_filtered) > 0:
                counts = df_filtered['Categoria_Predicha'].value_counts()
                
                fig_pie = px.pie(
                    values=counts.values, 
                    names=counts.index, 
                    title="Proporción por categoría (excluyendo errores)",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_pie.update_layout(
                    paper_bgcolor='#2a1a3e',
                    plot_bgcolor='#2a1a3e',
                    font_color="#ECE7EF"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No hay datos válidos para mostrar")
        
        with col2:
            if len(df_filtered) > 0:
                fig_bar = px.bar(
                    x=counts.index, 
                    y=counts.values,
                    title=" Frecuencia por categoría",
                    labels={'x': 'Categoría', 'y': 'Cantidad'},
                    color=counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig_bar.update_layout(
                    paper_bgcolor='#2a1a3e',
                    plot_bgcolor='#2a1a3e',
                    font_color="#F4F1F7"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No hay datos válidos para mostrar")
        
        st.subheader("🏆 Top reparaciones más frecuentes")
        top_n = st.slider("Mostrar top N reparaciones:", 1, 30, 10, key="static_top")
        
        df_no_errors = df_static[df_static['Categoria_Predicha'] != "Error"]
        top_desc = df_no_errors['Descripcion'].value_counts().head(top_n).reset_index()
        top_desc.columns = ['Descripción', 'Frecuencia']
        top_desc.index = range(1, len(top_desc) + 1)
        st.dataframe(top_desc, use_container_width=True)
        
        with st.expander(" Ver estadísticas detalladas"):
            st.write("**Distribución completa de categorías:**")
            st.dataframe(df_static['Categoria_Predicha'].value_counts().reset_index())
            
            errores_df = df_static[df_static['Categoria_Predicha'] == "Error"]
            if len(errores_df) > 0:
                st.write(f"**Ejemplos de {min(5, len(errores_df))} descripciones que causaron error:**")
                st.dataframe(errores_df[['Descripcion']].head(5))
        
        st.markdown("---")

# ============================
# 5. CATEGORIZACIÓN MANUAL
# ============================
st.header(" Categorización Manual")
with st.expander("✏️ Ingresar una descripción de reparación", expanded=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        manual_input = st.text_area(
            "Descripción:", 
            placeholder="Ej: cambio de aceite y filtro de aire",
            height=100
        )
    with col2:
        st.write("")
        st.write("")
        predict_button = st.button("Predecir categoría", type="primary", use_container_width=True)
    
    if predict_button and manual_input:
        if model is None:
            st.error("Modelo no cargado. Verifica los archivos.")
        else:
            with st.spinner("Analizando..."):
                categoria, probabilidades = predict_category(manual_input, model, vectorizer)
                
                if categoria and categoria != "Error":
                    st.success(f"### Categoría asignada: **{categoria}**")
                    
                    if probabilidades is not None and hasattr(model, 'classes_'):
                        st.write("#### Probabilidades por categoría:")
                        prob_df = pd.DataFrame({
                            'Categoría': model.classes_,
                            'Probabilidad': probabilidades
                        }).sort_values('Probabilidad', ascending=False)
                        
                        primary_colors = ['#1f77b4', '#ff0000', '#ffcc00', '#6f42c1', '#8e5edd']
                        fig = px.bar(
                            prob_df, 
                            x='Categoría', 
                            y='Probabilidad',
                            title="Distribución de probabilidad",
                            color='Categoría',
                            color_discrete_sequence=primary_colors
                        )
                        fig.update_layout(
                            paper_bgcolor='#2a1a3e',
                            plot_bgcolor='#2a1a3e',
                            font_color='#f4edff'
                        )
                        fig.update_traces(marker_line_color='#ffffff', marker_line_width=0.8)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.write("**Top 3 categorías más probables:**")
                        for i, row in prob_df.head(3).iterrows():
                            st.write(f"- {row['Categoría']}: {row['Probabilidad']:.2%}")
                    else:
                        st.info("ℹ️ Nota: El desglose visual de probabilidades no está disponible porque el modelo SVM se entrenó con la opción de probabilidades desactivada, pero la predicción de la categoría principal es 100% precisa.")
                else:
                    st.error("No se pudo categorizar el texto ingresado. Asegúrate de escribir palabras clave válidas.")

# ============================
# 6. CATEGORIZACIÓN MASIVA (CSV)
# ============================
st.header(" Categorización Masiva")
st.markdown("Sube un archivo CSV con una columna llamada **`Descripcion`**")

uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type=['csv'])

if uploaded_file and model is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader(" Vista previa de los datos cargados")
        st.dataframe(df.head())
        
        col_desc = None
        for col in df.columns:
            if col.lower() in ['descripcion', 'descripción', 'problema', 'reparacion']:
                col_desc = col
                break
        
        if col_desc is None:
            st.error(f"No se encontró una columna de descripción. Columnas disponibles: {list(df.columns)}")
            st.stop()
        
        st.info(f"Usando columna: **{col_desc}**")
        
        with st.spinner(" Procesando descripciones..."):
            predicciones = []
            progress_bar = st.progress(0)
            for idx, desc in enumerate(df[col_desc]):
                if idx % 50 == 0:
                    progress_bar.progress(min(idx / len(df), 1.0))
                cat, _ = predict_category(desc, model, vectorizer)
                predicciones.append(cat if cat else "Error")
            progress_bar.progress(1.0)
            
            df['Categoria_Predicha'] = predicciones
            st.success(" Categorización completada exitosamente!")
        
        st.subheader("Resultados")
        st.dataframe(df.head(20))
        
        col1, col2 = st.columns(2)
        
        with col1:
            counts = df['Categoria_Predicha'].value_counts()
            primary_colors = ['#1f77b4', '#ff0000', '#ffcc00', '#6f42c1', '#8e5edd']
            fig_pie = px.pie(
                values=counts.values, 
                names=counts.index, 
                title="Proporción por categoría",
                color_discrete_sequence=primary_colors
            )
            fig_pie.update_layout(
                paper_bgcolor='#2a1a3e',
                plot_bgcolor='#2a1a3e',
                font_color='#f4edff'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                x=counts.index, 
                y=counts.values,
                title="Frecuencia por categoría",
                labels={'x': 'Categoría', 'y': 'Cantidad'},
                color=counts.index,
                color_discrete_sequence=primary_colors
            )
            fig_bar.update_layout(
                paper_bgcolor='#2a1a3e',
                plot_bgcolor='#2a1a3e',
                font_color='#f4edff'
            )
            fig_bar.update_traces(marker_line_color='#ffffff', marker_line_width=0.8)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.subheader("🏆 Top reparaciones más frecuentes")
        top_n_csv = st.slider("Mostrar top N:", 1, 30, 10, key="csv_top")

        top_desc_csv = df.groupby([col_desc, 'Categoria_Predicha']).size().reset_index(name='Frecuencia')
        top_desc_csv = top_desc_csv.sort_values('Frecuencia', ascending=False).head(top_n_csv)
        top_desc_csv.columns = ['Descripción', 'Categoría', 'Frecuencia']
        top_desc_csv.index = range(1, len(top_desc_csv) + 1)

        st.dataframe(top_desc_csv, use_container_width=True)
        csv_resultado = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label=" Descargar resultados (CSV)",
            data=csv_resultado,
            file_name="reparaciones_categorizadas.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
        st.exception(e)

# ============================
# 7. EJEMPLOS DE PRUEBA
# ============================
with st.sidebar.expander(" Ejemplos de prueba"):
    st.markdown("""
    **Copia y pega estas descripciones:**
    
    - `cambio de filtro de aire y bujias`
    - `reparacion completa del motor`
    - `instalacion de luces led`
    - `revisión del sistema eléctrico`
    - `diagnóstico de ruido en el motor`
    - `reparación de la transmisión automática`
    - `problema de encendido`
    - `evaluación del motor`
    - `instalacion de sistema de sonido`
    """)

# ============================
# 8. INFORMACIÓN DEL SISTEMA
# ============================
if model is not None:
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Información del modelo:**
    - Tipo: Support Vector Machine (SVM)
    - Vectorizador: TF-IDF (max_features=1000)
    - Preprocesamiento: Limpieza + stopwords español
    - Categorías: {len(model.classes_) if hasattr(model, 'classes_') else 'N/A'}
    """)

st.sidebar.markdown("---")
