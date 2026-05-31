import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import re
import unicodedata
from nltk.corpus import stopwords
import nltk

# Descargar stopwords si es necesario
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

# Configuración de la página
st.set_page_config(page_title="Categorizador de Reparaciones", page_icon="🔧", layout="wide")
st.markdown("""
<style>
    .stApp {
        background-color: #f1ebff;
        color: #2b0b4a;
    }

    [data-testid="stSidebar"] {
        background-color: #380063;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #4B0082 !important;
    }

    .stCard, .stExpander, .stAlert, [data-testid="stForm"] {
        background-color: #ffffff !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        box-shadow: 0 10px 24px rgba(0,0,0,0.08) !important;
    }

    .stButton > button {
        background-color: #6f42c1 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        background-color: #8e5edd !important;
    }

    .stDataFrame {
        border: 1px solid #9b67d4 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔧 Categorizador Inteligente de Reparaciones")

# ============================
# 1. CARGAR MODELO Y VECTORIZADOR
# ============================
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load("modelo_svm.pkl")
        vectorizer = joblib.load("vectorizador.pkl")
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"❌ No se encuentra el archivo: {e}")
        st.info("""
        Asegúrate de tener estos archivos en la misma carpeta:
        - modelo_svm.pkl
        - vectorizador.pkl
        
        Si no los tienes, ejecuta el script de entrenamiento modificado para guardarlos.
        """)
        return None, None
    except Exception as e:
        st.error(f"❌ Error al cargar: {e}")
        return None, None

model, vectorizer = load_model_and_vectorizer()

if model is None or vectorizer is None:
    st.stop()

st.sidebar.success("✅ Modelo SVM y Vectorizador cargados correctamente")

# Mostrar categorías
if hasattr(model, 'classes_'):
    st.sidebar.write("📋 **Categorías disponibles:**")
    for i, cat in enumerate(model.classes_):
        st.sidebar.write(f"{i+1}. {cat}")

# ============================
# 2. FUNCIONES DE PREPROCESAMIENTO (IDÉNTICAS AL ENTRENAMIENTO)
# ============================
def preprocess_text(text):
    """MISMO preprocesamiento que usaste en el entrenamiento"""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()  # Minúsculas
    # Normalizar acentos (igual que en entrenamiento)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-z\s]', '', text)  # Solo letras y espacios
    words = text.split()  # Tokenizar
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in stop_words]  # Eliminar stopwords
    return ' '.join(words)

# ============================
# 3. FUNCIÓN DE PREDICCIÓN
# ============================
def predict_category(descripcion, model, vectorizer):
    """Predice usando el mismo pipeline del entrenamiento"""
    try:
        # Preprocesar (igual que en entrenamiento)
        texto_procesado = preprocess_text(descripcion)
        
        # Transformar con TF-IDF (usando el vectorizador guardado)
        texto_vectorizado = vectorizer.transform([texto_procesado])
        
        # Predecir
        prediccion = model.predict(texto_vectorizado)[0]
        
        # Obtener probabilidades (si el modelo lo soporta)
        probabilidades = None
        if hasattr(model, 'predict_proba'):
            probabilidades = model.predict_proba(texto_vectorizado)[0]
        
        return prediccion, probabilidades
    except Exception as e:
        st.error(f"Error en predicción: {e}")
        return None, None

# ============================
# 4. CATEGORIZACIÓN MANUAL
# ============================
st.header("📝 Categorización Manual")
with st.expander("Ingresar una descripción de reparación", expanded=True):
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
        predict_button = st.button("🔍 Predecir categoría", type="primary", use_container_width=True)
    
    if predict_button and manual_input:
        with st.spinner("Analizando..."):
            categoria, probabilidades = predict_category(manual_input, model, vectorizer)
            
            if categoria:
                st.success(f"### 📌 Categoría asignada: **{categoria}**")
                
                if probabilidades is not None and hasattr(model, 'classes_'):
                    st.write("#### 📊 Probabilidades por categoría:")
                    prob_df = pd.DataFrame({
                        'Categoría': model.classes_,
                        'Probabilidad': probabilidades
                    }).sort_values('Probabilidad', ascending=False)
                    
                    # Gráfico de barras
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
                        paper_bgcolor='#2e2e3c',
                        plot_bgcolor='#2e2e3c',
                        font_color='#f4edff'
                    )
                    fig.update_traces(marker_line_color='#ffffff', marker_line_width=0.8)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar top 3 categorías
                    st.write("**Top 3 categorías más probables:**")
                    for i, row in prob_df.head(3).iterrows():
                        st.write(f"- {row['Categoría']}: {row['Probabilidad']:.2%}")

# ============================
# 5. CATEGORIZACIÓN MASIVA (CSV)
# ============================
st.header("📂 Categorización Masiva (CSV)")
st.markdown("Sube un archivo CSV con una columna llamada **`Descripcion`** (exactamente como en el entrenamiento)")

uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Vista previa de los datos cargados")
        st.dataframe(df.head())
        
        # Buscar columna de descripción (case insensitive)
        col_desc = None
        for col in df.columns:
            if col.lower() in ['descripcion', 'descripción', 'problema', 'reparacion']:
                col_desc = col
                break
        
        if col_desc is None:
            st.error(f"No se encontró una columna de descripción. Columnas disponibles: {list(df.columns)}")
            st.info("Asegúrate de que tu CSV tenga una columna llamada 'Descripcion' (puede ser con mayúsculas o minúsculas)")
            st.stop()
        
        st.info(f"✅ Usando columna: **{col_desc}**")
        
        # Procesar todas las descripciones
        with st.spinner("🔄 Procesando descripciones... Esto puede tomar unos segundos"):
            predicciones = []
            for idx, desc in enumerate(df[col_desc]):
                if idx % 100 == 0:  # Mostrar progreso cada 100 filas
                    st.progress(idx / len(df))
                cat, _ = predict_category(desc, model, vectorizer)
                predicciones.append(cat if cat else "Error")
            
            df['Categoria_Predicha'] = predicciones
            st.success("✅ Categorización completada exitosamente!")
        
        # Mostrar resultados
        st.subheader("📊 Resultados")
        st.dataframe(df.head(20))
        
        # Estadísticas y gráficos
        st.subheader("📈 Análisis de distribuciones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Conteo por categoría
            counts = df['Categoria_Predicha'].value_counts()
            primary_colors = ['#1f77b4', '#ff0000', '#ffcc00', '#6f42c1', '#8e5edd']
            fig_pie = px.pie(
                values=counts.values, 
                names=counts.index, 
                title="Proporción por categoría",
                color_discrete_sequence=primary_colors
            )
            fig_pie.update_layout(
                paper_bgcolor='#2e2e3c',
                plot_bgcolor='#2e2e3c',
                font_color='#f4edff'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Gráfico de barras
            primary_colors = ['#1f77b4', '#ff0000', '#ffcc00', '#6f42c1', '#8e5edd']
            fig_bar = px.bar(
                x=counts.index, 
                y=counts.values,
                title="Frecuencia por categoría",
                labels={'x': 'Categoría', 'y': 'Cantidad'},
                color=counts.index,
                color_discrete_sequence=primary_colors
            )
            fig_bar.update_layout(
                paper_bgcolor='#2e2e3c',
                plot_bgcolor='#2e2e3c',
                font_color='#f4edff'
            )
            fig_bar.update_traces(marker_line_color='#ffffff', marker_line_width=0.8)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Top reparaciones más comunes
        st.subheader("🏆 Top reparaciones más frecuentes")
        top_n = st.slider("Mostrar top N reparaciones:", 5, 30, 10)
        
        descripciones_frecuentes = df[col_desc].value_counts().head(top_n).reset_index()
        descripciones_frecuentes.columns = ['Descripción', 'Frecuencia']
        
        # Mostrar tabla
        st.dataframe(descripciones_frecuentes, use_container_width=True)
        
        # Análisis por categoría (si el usuario quiere profundizar)
        st.subheader("🔍 Análisis detallado por categoría")
        categoria_seleccionada = st.selectbox("Selecciona una categoría para ver detalles:", counts.index)
        
        if categoria_seleccionada:
            df_filtrado = df[df['Categoria_Predicha'] == categoria_seleccionada]
            st.write(f"**{len(df_filtrado)}** reparaciones en esta categoría")
            
            top_desc_cat = df_filtrado[col_desc].value_counts().head(10).reset_index()
            top_desc_cat.columns = ['Descripción', 'Frecuencia']
            st.dataframe(top_desc_cat, use_container_width=True)
        
        # Botón de descarga
        csv_resultado = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="💾 Descargar resultados (CSV)",
            data=csv_resultado,
            file_name="reparaciones_categorizadas.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
        st.exception(e)

# ============================
# 6. EJEMPLOS DE PRUEBA (OPCIONAL)
# ============================
with st.sidebar.expander("🎯 Ejemplos de prueba"):
    st.markdown("""
    **Copia y pega estas descripciones para probar:**
    
    - `cambio de filtro de aire y bujias`
    - `reparacion completa del motor`
    - `instalacion de luces led`
    - `revisión del sistema eléctrico`
    - `diagnóstico de ruido en el motor`
    - `reparación de la transmisión automática`
    - `problema de encendido`
    - `evaluación del motor`
    - `instalacion de sistema de sonido`
    - `el motor no arranca`
    - `revisar calentamiento`
    - `pintar carroceria`
    - `chequeo de frenos y pastillas`
    - `cambio de balatas delanteras`
    - `actualización de software de motor`
    """)

# ============================
# 7. INFORMACIÓN DEL SISTEMA
# ============================
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Información del modelo:**
- Tipo: Support Vector Machine (SVM)
- Vectorizador: TF-IDF (max_features=1000)
- Preprocesamiento: Limpieza + stopwords español
- Categorías: {len(model.classes_) if hasattr(model, 'classes_') else 'N/A'}
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado con  para categorización de reparaciones")