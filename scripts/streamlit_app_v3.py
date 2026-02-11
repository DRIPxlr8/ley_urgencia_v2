"""
Sistema de Predicción - Ley de Urgencia (Decreto 34)
Versión 2.0 - Modelo Random Forest optimizado
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from io import BytesIO
import os
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Predictor Ley de Urgencia",
    page_icon="🏥",
    layout="wide"
)

# Obtener ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / 'modelos'


@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado y sus componentes"""
    try:
        modelo = joblib.load(MODELOS_DIR / 'modelo_ley_urgencia.pkl')
        preprocessor = joblib.load(MODELOS_DIR / 'preprocessor.pkl')
        scaler = joblib.load(MODELOS_DIR / 'scaler.pkl')
        metadata = joblib.load(MODELOS_DIR / 'modelo_metadata.pkl')
        return modelo, preprocessor, scaler, metadata
    except FileNotFoundError as e:
        st.error(f"❌ Error al cargar modelo: {e}")
        st.info(f"Buscando en: {MODELOS_DIR}")
        return None, None, None, None


def preparar_features(df):
    """Prepara las features derivadas igual que en entrenamiento"""
    df_prep = df.copy()
    
    # Conversión de numéricas
    columnas_numericas = ['FC', 'FR', 'PAS', 'PAD', 'SatO2', 'Temp', 'Glasgow', 'Triage']
    for col in columnas_numericas:
        if col in df_prep.columns:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')
    
    # Conversión de binarias
    columnas_binarias = ['HipertencionArterial', 'DiabetesMellitus', 'Cardiopatia']
    for col in columnas_binarias:
        if col in df_prep.columns:
            df_prep[col] = df_prep[col].astype(str).str.strip().str.lower()
            df_prep[col] = df_prep[col].map({
                'si': 1, 'sí': 1, 's': 1, '1': 1, 1: 1,
                'no': 0, 'n': 0, '0': 0, 0: 0
            })
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce').fillna(0)
    
    # Features derivadas
    if 'SatO2' in df_prep.columns and 'FR' in df_prep.columns:
        df_prep['Ratio_SatO2_FR'] = df_prep['SatO2'] / (df_prep['FR'] + 1)
    
    if 'PAS' in df_prep.columns and 'PAD' in df_prep.columns:
        df_prep['Flag_Hipotension'] = ((df_prep['PAS'] < 90) | (df_prep['PAD'] < 60)).astype(int)
        df_prep['Ratio_PAM'] = (df_prep['PAS'] + 2*df_prep['PAD']) / 3
    
    if 'FC' in df_prep.columns:
        df_prep['Flag_Taquicardia'] = (df_prep['FC'] > 100).astype(int)
    
    if 'Temp' in df_prep.columns:
        df_prep['Flag_Fiebre'] = (df_prep['Temp'] > 38).astype(int)
    
    if 'Glasgow' in df_prep.columns:
        df_prep['Flag_GlasgowBajo'] = (df_prep['Glasgow'] < 13).astype(int)
    
    # Score gravedad
    score_components = []
    for flag in ['Flag_Hipotension', 'Flag_Taquicardia', 'Flag_Fiebre', 'Flag_GlasgowBajo']:
        if flag in df_prep.columns:
            score_components.append(df_prep[flag])
    
    if score_components:
        df_prep['Score_Gravedad'] = sum(score_components)
    
    if 'Triage' in df_prep.columns:
        df_prep['Flag_TriageCritico'] = (df_prep['Triage'] <= 2).astype(int)
    
    return df_prep


def predecir(df_input, modelo, preprocessor, scaler, metadata):
    """Genera predicciones para un DataFrame"""
    
    # Preparar features
    df_prep = preparar_features(df_input)
    
    # Obtener features del modelo
    features_modelo = metadata['features']
    
    # Asegurar que existan todas las features
    for feat in features_modelo:
        if feat not in df_prep.columns:
            df_prep[feat] = 0  # Valor por defecto
    
    # Seleccionar solo las features del modelo
    X = df_prep[features_modelo]
    
    # Aplicar pipeline
    X_prep = preprocessor.transform(X)
    X_scaled = scaler.transform(X_prep)
    
    # Predecir
    predicciones = modelo.predict(X_scaled)
    probabilidades = modelo.predict_proba(X_scaled)
    
    # Convertir a labels
    pred_labels = pd.Series(predicciones).map({1: 'PERTINENTE', 0: 'NO PERTINENTE'})
    confianza = probabilidades.max(axis=1) * 100
    
    return pred_labels, confianza, probabilidades


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.title("🏥 Sistema de Clasificación - Ley de Urgencia")
st.markdown("**Decreto 34 - Clasificación de Atenciones de Urgencia**")
st.markdown("---")

# Cargar modelo
modelo, preprocessor, scaler, metadata = cargar_modelo()

if modelo is None:
    st.stop()

# Mostrar información del modelo
with st.sidebar:
    st.markdown("### 📊 Información del Modelo")
    st.markdown(f"**Fecha entrenamiento:**  \n{metadata['fecha_entrenamiento'][:10]}")
    st.markdown(f"**Casos entrenamiento:**  \n{metadata['casos_totales']:,}")
    st.markdown(f"**Accuracy test:**  \n{metadata['accuracy_test']:.2%}")
    st.markdown(f"**Features:**  \n{len(metadata['features'])}")
    st.markdown("---")
    st.markdown("**Modelo:** Random Forest")
    st.markdown(f"**Árboles:** {metadata['n_estimators']}")
    st.markdown(f"**Profundidad:** {metadata['max_depth']}")
    st.markdown("---")
    st.markdown("### ⚙️ Umbral de Confianza")
    umbral_confianza = st.slider(
        "Confianza mínima (%)",
        min_value=50, max_value=90, value=65, step=5,
        help="Predicciones con confianza menor a este valor se marcarán como INDETERMINADO"
    )
    st.caption(f"Predicciones con < {umbral_confianza}% de confianza se mostrarán como indeterminadas")

# Selector de modo
modo = st.radio(
    "Seleccione modo de operación:",
    ["📝 Formulario Individual", "📤 Carga Masiva (Excel)"],
    horizontal=True
)

st.markdown("---")

# ============================================================
# MODO 1: FORMULARIO INDIVIDUAL
# ============================================================

if modo == "📝 Formulario Individual":
    
    st.markdown("### Ingreso de Datos del Paciente")
    
    # Información de campos requeridos
    with st.expander("ℹ️ Campos Requeridos por el Modelo", expanded=False):
        st.markdown("""
        **El modelo utiliza 19 características en total:**
        
        **📋 Campos Base (11) - Debe ingresar:**
        - ✅ Frecuencia Cardíaca (FC)
        - ✅ Frecuencia Respiratoria (FR)
        - ✅ Presión Arterial Sistólica (PAS)
        - ✅ Presión Arterial Diastólica (PAD)
        - ✅ Saturación de Oxígeno (SatO2)
        - ✅ Temperatura
        - ✅ Escala de Glasgow
        - ✅ Triage (1-5)
        - ✅ Hipertensión Arterial (antecedente)
        - ✅ Diabetes Mellitus (antecedente)
        - ✅ Cardiopatía (antecedente)
        
        **🔄 Campos Derivados (8) - Se calculan automáticamente:**
        - ⚙️ Ratio SatO2/FR
        - ⚙️ Flag Hipotensión
        - ⚙️ Flag Taquicardia
        - ⚙️ Flag Fiebre
        - ⚙️ Flag Glasgow Bajo
        - ⚙️ Score de Gravedad
        - ⚙️ Presión Arterial Media
        - ⚙️ Flag Triage Crítico
        """)
    
    st.markdown("---")
    
    # Usar formulario para evitar recargas constantes
    with st.form("form_prediccion"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Signos Vitales** ⭐")
            fc = st.number_input("Frecuencia Cardíaca (lpm)*", value=80, min_value=0, max_value=250, help="Latidos por minuto")
            fr = st.number_input("Frecuencia Respiratoria (rpm)*", value=16, min_value=0, max_value=60, help="Respiraciones por minuto")
            pas = st.number_input("PA Sistólica (mmHg)*", value=120, min_value=0, max_value=300, help="Presión arterial sistólica")
            pad = st.number_input("PA Diastólica (mmHg)*", value=80, min_value=0, max_value=200, help="Presión arterial diastólica")
        
        with col2:
            st.markdown("**Parámetros Clínicos** ⭐")
            sato2 = st.number_input("Saturación O₂ (%)*", value=98, min_value=0, max_value=100, help="Saturación de oxígeno")
            temp = st.number_input("Temperatura (°C)*", value=36.5, min_value=30.0, max_value=45.0, step=0.1, help="Temperatura corporal")
            glasgow = st.number_input("Glasgow*", value=15, min_value=3, max_value=15, help="Escala de coma de Glasgow (3-15)")
            triage = st.selectbox("Triage*", [1, 2, 3, 4, 5], index=2, help="1=Crítico, 5=Menor urgencia")
        
        with col3:
            st.markdown("**Antecedentes** ⭐")
            hta = st.selectbox("Hipertensión Arterial*", ["No", "Si"], index=0, help="Antecedente de HTA")
            diabetes = st.selectbox("Diabetes Mellitus*", ["No", "Si"], index=0, help="Antecedente de diabetes")
            cardiopatia = st.selectbox("Cardiopatía*", ["No", "Si"], index=0, help="Antecedente de enfermedad cardíaca")
        
        st.caption("⭐ Todos los campos son obligatorios para generar la predicción")
        
        st.markdown("---")
        
        submitted = st.form_submit_button("🔮 GENERAR PREDICCIÓN", type="primary", use_container_width=True)
    
    if submitted:
        
        # Validaciones básicas
        validaciones = []
        
        if fc <= 0:
            validaciones.append("⚠️ Frecuencia Cardíaca debe ser mayor a 0")
        if fr <= 0:
            validaciones.append("⚠️ Frecuencia Respiratoria debe ser mayor a 0")
        if pas <= 0 or pad <= 0:
            validaciones.append("⚠️ Presiones arteriales deben ser mayores a 0")
        if sato2 <= 0 or sato2 > 100:
            validaciones.append("⚠️ Saturación O₂ debe estar entre 1-100%")
        if temp < 30 or temp > 45:
            validaciones.append("⚠️ Temperatura fuera de rango normal")
        
        if validaciones:
            for val in validaciones:
                st.error(val)
            st.stop()
        
        # Crear DataFrame
        data = {
            'FC': [fc],
            'FR': [fr],
            'PAS': [pas],
            'PAD': [pad],
            'SatO2': [sato2],
            'Temp': [temp],
            'Glasgow': [glasgow],
            'Triage': [triage],
            'HipertencionArterial': [1 if hta == "Si" else 0],
            'DiabetesMellitus': [1 if diabetes == "Si" else 0],
            'Cardiopatia': [1 if cardiopatia == "Si" else 0]
        }
        
        df = pd.DataFrame(data)
        
        # Predecir
        with st.spinner("Analizando..."):
            pred_labels, confianza, probabilidades = predecir(df, modelo, preprocessor, scaler, metadata)
        
        # Mostrar resultados
        st.markdown("### 📋 Resultado de la Predicción")
        
        resultado = pred_labels.iloc[0]
        conf = confianza[0]
        
        # Determinar si la confianza es suficiente
        es_indeterminado = conf < umbral_confianza
        
        if es_indeterminado:
            st.warning("### ⚠️ NO ES POSIBLE DETERMINAR CON CERTEZA")
            st.markdown(
                f"""<div style='background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; color: #856404;'>
                <b>El modelo no tiene suficiente confianza para emitir un resultado definitivo.</b><br><br>
                La confianza de la predicción es <b>{conf:.1f}%</b>, inferior al umbral mínimo de <b>{umbral_confianza}%</b>.<br>
                Se recomienda <b>revisión manual por un profesional médico</b> para determinar la pertinencia de esta atención.
                </div>""",
                unsafe_allow_html=True
            )
            st.markdown("")
            st.info(f"💡 **Tendencia del modelo:** {resultado} ({conf:.1f}% confianza) — *pero no es concluyente*")
        else:
            # Color según resultado
            if resultado == "PERTINENTE":
                st.success(f"### ✅ {resultado}")
            else:
                st.info(f"### ℹ️ {resultado}")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confianza", f"{conf:.1f}%", 
                      delta=f"{'⚠️ Bajo umbral' if es_indeterminado else '✓ Confiable'}")
        
        with col2:
            prob_pert = probabilidades[0][1] * 100
            st.metric("P(PERTINENTE)", f"{prob_pert:.1f}%")
        
        with col3:
            prob_no_pert = probabilidades[0][0] * 100
            st.metric("P(NO PERTINENTE)", f"{prob_no_pert:.1f}%")
        
        # Detalles clínicos
        st.markdown("### 🔍 Análisis Clínico")
        
        alertas = []
        
        if pas < 90 or pad < 60:
            alertas.append("⚠️ Hipotensión detectada")
        
        if fc > 100:
            alertas.append("⚠️ Taquicardia detectada")
        
        if temp > 38:
            alertas.append("⚠️ Fiebre detectada")
        
        if glasgow < 13:
            alertas.append("⚠️ Glasgow bajo - Compromiso de conciencia")
        
        if triage <= 2:
            alertas.append("⚠️ Triage crítico")
        
        if sato2 < 90:
            alertas.append("⚠️ Hipoxemia")
        
        if alertas:
            for alerta in alertas:
                st.warning(alerta)
        else:
            st.success("✓ Sin alertas clínicas críticas")

# ============================================================
# MODO 2: CARGA MASIVA
# ============================================================

else:
    st.markdown("### 📤 Carga Masiva de Datos")
    
    st.info("""
    **📋 Formato del archivo Excel (.xlsx):**
    
    **Columnas OBLIGATORIAS (11):**
    
    | Columna | Tipo | Ejemplo |
    |---------|------|---------|
    | FC | Numérico | 80 |
    | FR | Numérico | 16 |
    | PAS | Numérico | 120 |
    | PAD | Numérico | 80 |
    | SatO2 | Numérico | 98 |
    | Temp | Numérico | 36.5 |
    | Glasgow | Numérico (3-15) | 15 |
    | Triage | Numérico (1-5) | 3 |
    | HipertencionArterial | Si/No o 1/0 | Si |
    | DiabetesMellitus | Si/No o 1/0 | No |
    | Cardiopatia | Si/No o 1/0 | No |
    
    **Columnas OPCIONALES (se preservan en el resultado):**
    - Episodio, Fecha, Centro, RUT, Nombre, etc.
    
    **💡 Las 8 características derivadas se calculan automáticamente:**
    - Ratio_SatO2_FR, Flag_Hipotension, Flag_Taquicardia, Flag_Fiebre,
    - Flag_GlasgowBajo, Score_Gravedad, Ratio_PAM, Flag_TriageCritico
    """)
    
    # Botón para descargar plantilla
    col_a, col_b = st.columns([3, 1])
    with col_b:
        plantilla = pd.DataFrame({
            'Episodio': ['12345678'],
            'FC': [80],
            'FR': [16],
            'PAS': [120],
            'PAD': [80],
            'SatO2': [98],
            'Temp': [36.5],
            'Glasgow': [15],
            'Triage': [3],
            'HipertencionArterial': ['No'],
            'DiabetesMellitus': ['No'],
            'Cardiopatia': ['No']
        })
        
        output_plantilla = BytesIO()
        with pd.ExcelWriter(output_plantilla, engine='openpyxl') as writer:
            plantilla.to_excel(writer, index=False, sheet_name='Plantilla')
        
        st.download_button(
            label="📥 Descargar Plantilla Excel",
            data=output_plantilla.getvalue(),
            file_name="plantilla_ley_urgencia.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Descarga una plantilla con el formato correcto"
        )
    
    st.markdown("---")
    
    archivo = st.file_uploader(
        "Seleccione archivo Excel (.xlsx)",
        type=['xlsx'],
        help="El archivo debe contener las columnas requeridas"
    )
    
    if archivo is not None:
        
        try:
            # Leer archivo
            df = pd.read_excel(archivo)
            
            st.success(f"✅ Archivo cargado: {len(df):,} registros")
            
            # Mostrar preview
            with st.expander("👁️ Vista previa de datos"):
                st.dataframe(df.head(10))
            
            # Verificar columnas mínimas
            cols_requeridas = ['FC', 'FR', 'PAS', 'PAD', 'SatO2', 'Temp', 'Glasgow', 'Triage']
            cols_faltantes = [c for c in cols_requeridas if c not in df.columns]
            
            # Verificar columnas de antecedentes (pueden tener nombres alternativos)
            cols_antecedentes = ['HipertencionArterial', 'DiabetesMellitus', 'Cardiopatia']
            antecedentes_faltantes = []
            
            for col in cols_antecedentes:
                if col not in df.columns:
                    antecedentes_faltantes.append(col)
            
            if cols_faltantes or antecedentes_faltantes:
                st.error("❌ **Faltan columnas obligatorias:**")
                if cols_faltantes:
                    st.error(f"**Signos vitales:** {', '.join(cols_faltantes)}")
                if antecedentes_faltantes:
                    st.error(f"**Antecedentes:** {', '.join(antecedentes_faltantes)}")
                
                st.warning("💡 **Solución:** Descarga la plantilla Excel arriba y úsala como referencia.")
            else:
                
                if st.button("🚀 Procesar Archivo", type="primary"):
                    
                    with st.spinner(f"Procesando {len(df):,} registros..."):
                        
                        # Predecir
                        pred_labels, confianza, probabilidades = predecir(df, modelo, preprocessor, scaler, metadata)
                        
                        # Crear DataFrame de resultados
                        df_resultado = df.copy()
                        df_resultado['Prediccion_Modelo'] = pred_labels.values
                        df_resultado['Confianza_%'] = confianza.round(2)
                        
                        # Aplicar umbral de confianza: marcar como INDETERMINADO si confianza < umbral
                        df_resultado['Prediccion'] = df_resultado.apply(
                            lambda row: 'INDETERMINADO - Requiere revisión manual'
                            if row['Confianza_%'] < umbral_confianza
                            else row['Prediccion_Modelo'],
                            axis=1
                        )
                        
                        df_resultado['Prob_PERTINENTE_%'] = (probabilidades[:, 1] * 100).round(2)
                        df_resultado['Prob_NO_PERTINENTE_%'] = (probabilidades[:, 0] * 100).round(2)
                        df_resultado['Fecha_Prediccion'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Mostrar resultados
                        st.success("✅ Procesamiento completado")
                        
                        # Estadísticas
                        st.markdown("### 📊 Resumen de Resultados")
                        
                        total = len(df_resultado)
                        pertinentes = (df_resultado['Prediccion'] == 'PERTINENTE').sum()
                        no_pertinentes = (df_resultado['Prediccion'] == 'NO PERTINENTE').sum()
                        indeterminados = (df_resultado['Prediccion'].str.startswith('INDETERMINADO')).sum()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Casos", f"{total:,}")
                        
                        with col2:
                            pct_pert = (pertinentes / total * 100) if total > 0 else 0
                            st.metric("✅ PERTINENTE", f"{pertinentes:,} ({pct_pert:.1f}%)")
                        
                        with col3:
                            pct_no = (no_pertinentes / total * 100) if total > 0 else 0
                            st.metric("ℹ️ NO PERTINENTE", f"{no_pertinentes:,} ({pct_no:.1f}%)")
                        
                        with col4:
                            pct_ind = (indeterminados / total * 100) if total > 0 else 0
                            st.metric("⚠️ INDETERMINADO", f"{indeterminados:,} ({pct_ind:.1f}%)")
                        
                        if indeterminados > 0:
                            st.warning(f"⚠️ **{indeterminados} caso(s)** tienen confianza menor a {umbral_confianza}% y requieren revisión manual. "
                                      f"La columna 'Prediccion_Modelo' contiene la tendencia del modelo para referencia.")
                        
                        # Mostrar resultados
                        st.markdown("### 📋 Resultados Detallados")
                        st.dataframe(df_resultado)
                        
                        # Descargar resultados
                        st.markdown("### 💾 Descargar Resultados")
                        
                        # Convertir a Excel
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_resultado.to_excel(writer, index=False, sheet_name='Resultados')
                        
                        st.download_button(
                            label="📥 Descargar Resultados (Excel)",
                            data=output.getvalue(),
                            file_name=f"predicciones_ley_urgencia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        
        except Exception as e:
            st.error(f"❌ Error al procesar archivo: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Sistema de Clasificación Ley de Urgencia - Decreto 34<br>
    UC CHRISTUS Chile | Versión 2.0 | Modelo Random Forest
</div>
""", unsafe_allow_html=True)
