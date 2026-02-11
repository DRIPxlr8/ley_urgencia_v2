"""
Sistema Inteligente de Predicción - Ley de Urgencia (Decreto 34)
Versión actualizada con modelo robusto y carga de archivos Excel
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from io import BytesIO

# Importar funciones del módulo principal
from resolucion_ml_v2 import prepare_dataframe, normalize_columns


@st.cache_resource
def cargar_modelo_entrenado():
    """Carga el modelo robusto ya entrenado"""
    try:
        modelo = joblib.load('modelo_ley_urgencia.pkl')
        metadata = joblib.load('modelo_metadata.pkl')
        return modelo, metadata
    except FileNotFoundError:
        st.error("❌ No se encontró el modelo entrenado. Ejecuta primero: `python entrenar_modelo_robusto.py`")
        return None, None


def procesar_archivo_excel(archivo_subido, modelo, metadata):
    """Procesa un archivo Excel y genera predicciones"""
    
    # Leer archivo Excel
    df_raw = pd.read_excel(archivo_subido)
    
    st.info(f"📊 Archivo cargado: {len(df_raw):,} filas")
    
    # Preparar dataframe (normalizar columnas, calcular features)
    try:
        # Crear archivo temporal para prepare_dataframe
        temp_path = Path("temp_upload.xlsx")
        df_raw.to_excel(temp_path, index=False)
        
        df, alert_cols, decree_cols, df_raw_full = prepare_dataframe(temp_path)
        temp_path.unlink()  # Eliminar temporal
        
    except Exception as e:
        st.error(f"❌ Error al preparar datos: {e}")
        return None
    
    # Asegurar que existan todas las features necesarias
    binary_cols = metadata.get('all_binary', [])
    
    for col in metadata['feature_cols']:
        if col not in df.columns:
            if col in binary_cols:
                df[col] = 0
            elif col in metadata['categorical_cols']:
                df[col] = "desconocido"
            else:
                df[col] = np.nan
    
    X = df[metadata['feature_cols']].copy()
    
    # Generar predicciones
    with st.spinner("🤖 Generando predicciones con IA..."):
        preds = modelo.predict(X)
        preds_labels = pd.Series(preds).map(metadata['int_to_label'])
        proba = modelo.predict_proba(X)
        confianza = proba.max(axis=1)
    
    # Crear DataFrame de resultados
    df_resultados = df_raw_full.copy()
    df_resultados['Prediccion_IA'] = preds_labels.values
    df_resultados['Confianza_IA'] = (confianza * 100).round(2)
    
    # Aplicar lógica híbrida con Decreto 34 si existe
    if "CUMPLE_CRITERIO_DECRETO" in df.columns:
        cumple_decreto = df["CUMPLE_CRITERIO_DECRETO"].astype(bool)
        
        ETIQUETA_POSITIVA = "PERTINENTE"
        UMBRAL_CONFIANZA_ALTA = 0.55
        
        decisiones_finales = []
        for i in range(len(df)):
            cumple_d = cumple_decreto.iloc[i]
            pred_ml = preds_labels.iloc[i]
            conf = confianza[i]
            
            # PRIORIDAD 1: Alta confianza ML
            if conf >= UMBRAL_CONFIANZA_ALTA:
                decision = pred_ml
                metodo = "ML (alta confianza)"
            # PRIORIDAD 2: Decreto 34 como tiebreaker
            elif cumple_d:
                decision = ETIQUETA_POSITIVA
                metodo = "Decreto 34"
            # PRIORIDAD 3: Conservador (NO PERTINENTE)
            else:
                decision = "NO PERTINENTE"
                metodo = "ML (conservador)"
            
            decisiones_finales.append((decision, metodo))
        
        df_resultados['Decision_Final'] = [d[0] for d in decisiones_finales]
        df_resultados['Metodo_Decision'] = [d[1] for d in decisiones_finales]
    else:
        df_resultados['Decision_Final'] = preds_labels.values
        df_resultados['Metodo_Decision'] = "ML puro"
    
    return df_resultados


def modo_formulario_individual():
    """Modo de predicción con formulario individual (modo original)"""
    
    st.markdown("### 📋 Ingreso de Datos del Paciente")
    
    # Tabs para organizar el formulario
    tab1, tab2, tab3 = st.tabs([
        "🩺 Signos Vitales & Lab", 
        "💉 Procedimientos & Soporte",
        "📝 Información Adicional"
    ])
    
    # TAB 1: Signos vitales y laboratorio
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Signos Vitales**")
            pas = st.number_input("PA Sistólica (mmHg)", value=120.0, step=1.0)
            pad = st.number_input("PA Diastólica (mmHg)", value=80.0, step=1.0)
            temp = st.number_input("Temperatura (°C)", value=36.5, step=0.1)
            sato2 = st.number_input("Saturación O₂ (%)", value=98.0, step=1.0)
            fc = st.number_input("Frecuencia Cardíaca (lpm)", value=80.0, step=1.0)
            fr = st.number_input("Frecuencia Respiratoria (rpm)", value=16.0, step=1.0)
            glasgow = st.number_input("Escala de Glasgow", min_value=3, max_value=15, value=15)
        
        with col2:
            st.markdown("**Laboratorio**")
            hemo = st.number_input("Hemoglobina (g/dL)", value=14.0, step=0.1)
            creat = st.number_input("Creatinina (mg/dL)", value=1.0, step=0.1)
            bun = st.number_input("BUN (mg/dL)", value=15.0, step=1.0)
            sodio = st.number_input("Sodio (mEq/L)", value=140.0, step=1.0)
            potasio = st.number_input("Potasio (mEq/L)", value=4.0, step=0.1)
            pcr = st.number_input("PCR (mg/L)", value=5.0, step=0.5)
    
    # TAB 2: Procedimientos
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Soporte Vital**")
            vm = st.checkbox("Ventilación Mecánica")
            dva = st.checkbox("Drogas Vasoactivas")
            fio2_alto = st.checkbox("FiO₂ ≥ 50%")
            
            st.markdown("**Procedimientos**")
            cirugia = st.checkbox("Cirugía Realizada")
            hemodinamia = st.checkbox("Hemodinamia")
            dialisis = st.checkbox("Diálisis")
        
        with col2:
            st.markdown("**Estudios Complementarios**")
            troponinas = st.checkbox("Troponinas Alteradas")
            ecg_alt = st.checkbox("ECG Alterado")
            rnm = st.checkbox("RNM Stroke")
            
            st.markdown("**Estado General**")
            comp_conc = st.checkbox("Compromiso de Conciencia")
            transfusiones = st.checkbox("Transfusiones")
    
    # TAB 3: Adicional
    with tab3:
        st.markdown("**Antecedentes**")
        col1, col2, col3 = st.columns(3)
        with col1:
            ant_card = st.checkbox("Cardíaco")
        with col2:
            ant_diab = st.checkbox("Diabético")
        with col3:
            ant_hta = st.checkbox("HTA")
        
        triage = st.selectbox("Triage", options=[1,2,3,4,5], index=4,
                            format_func=lambda x: f"C{x}")
        tipo_cama = st.selectbox("Tipo de Cama", 
                                options=["UCI Adulto", "UCI Pediátrico", "Intermedio", 
                                        "Básico", "Box Urgencia", "desconocido"],
                                index=5)
        
        st.markdown("**Motivo de Consulta / Diagnóstico**")
        texto_libre = st.text_area(
            "Descripción clínica (motivo, síntomas, diagnósticos)",
            placeholder="Ej: Paciente con dolor torácico, disnea de esfuerzo, antecedente de IAM...",
            height=100,
            help="El modelo analiza palabras clave como: infarto, stroke, pancreatitis, sepsis, etc."
        )
    
    st.markdown("---")
    
    if st.button("🔮 GENERAR PREDICCIÓN", type="primary", use_container_width=True):
        
        # Crear DataFrame con los datos ingresados
        data = {
            'PA_Sistolica': [pas],
            'PA_Diastolica': [pad],
            'Temperatura': [temp],
            'SatO2': [sato2],
            'FC': [fc],
            'FR': [fr],
            'Glasgow': [glasgow],
            'Hemoglobina': [hemo],
            'Creatinina': [creat],
            'BUN': [bun],
            'Sodio': [sodio],
            'Potasio': [potasio],
            'PCR': [pcr],
            'Ventilacion_Mecanica': [1 if vm else 0],
            'DVA': [1 if dva else 0],
            'FiO2_ge50_flag': [1 if fio2_alto else 0],
            'Cirugia': [1 if cirugia else 0],
            'Hemodinamia': [1 if hemodinamia else 0],
            'Dialisis': [1 if dialisis else 0],
            'Troponinas_Alteradas': [1 if troponinas else 0],
            'ECG_Alterado': [1 if ecg_alt else 0],
            'RNM_Stroke': [1 if rnm else 0],
            'Compromiso_Conciencia': [1 if comp_conc else 0],
            'Transfusiones': [1 if transfusiones else 0],
            'Antecedente_Cardiaco': [1 if ant_card else 0],
            'Antecedente_Diabetico': [1 if ant_diab else 0],
            'Antecedente_HTA': [1 if ant_hta else 0],
            'Triage': [triage],
            'Tipo_Cama': [tipo_cama],
            'Texto_Libre': [texto_libre]
        }
        
        # Cargar modelo
        modelo, metadata = cargar_modelo_entrenado()
        if modelo is None:
            return
        
        # Crear DataFrame y asegurar todas las columnas
        df = pd.DataFrame(data)
        
        # Procesar texto libre para generar alertas y flags de decreto
        from resolucion_ml_v2 import add_text_alerts, add_decree_flags
        df, _ = add_text_alerts(df)
        df, _ = add_decree_flags(df)
        
        binary_cols = metadata.get('all_binary', [])
        for col in metadata['feature_cols']:
            if col not in df.columns:
                if col in binary_cols:
                    df[col] = 0
                elif col in metadata['categorical_cols']:
                    df[col] = "desconocido"
                else:
                    df[col] = np.nan
        
        X = df[metadata['feature_cols']]
        
        # Predecir
        with st.spinner("🤖 Procesando..."):
            pred = modelo.predict(X)[0]
            proba = modelo.predict_proba(X)[0]
            pred_label = metadata['int_to_label'][pred]
            confianza = proba.max() * 100
        
        # Mostrar resultado
        st.markdown("---")
        st.markdown("### 🎯 Resultado")
        
        if pred_label == "PERTINENTE":
            st.error(f"""
            ### ⚠️ SÍ ES LEY DE URGENCIA
            
            **El paciente CUMPLE con los criterios del Decreto 34**
            
            Confianza: {confianza:.1f}%
            """)
        else:
            st.success(f"""
            ### ✅ NO ES LEY DE URGENCIA
            
            **El paciente NO cumple con los criterios del Decreto 34**
            
            Confianza: {confianza:.1f}%
            """)
        
        # Mostrar alertas detectadas del texto libre si existen
        alertas_texto = []
        cols_alerta_texto = [
            'alert_infarto', 'alert_stroke', 'alert_pancreatitis', 
            'alert_sepsis', 'alert_meningitis', 'alert_hemorragia'
        ]
        for col in cols_alerta_texto:
            if col in df.columns and df[col].iloc[0] == 1:
                alertas_texto.append(col.replace('alert_', '').capitalize())
        
        if alertas_texto:
            st.info(f"🔍 **Palabras clave detectadas en texto libre:** {', '.join(alertas_texto)}")
        
        # Mostrar si cumple criterios Decreto 34
        if 'CUMPLE_CRITERIO_DECRETO' in df.columns and df['CUMPLE_CRITERIO_DECRETO'].iloc[0] == 1:
            st.warning("⚖️ **Cumple criterios del Decreto 34** (basado en reglas clínicas)")


def modo_carga_archivo():
    """Modo de predicción masiva desde archivo Excel"""
    
    st.markdown("### 📤 Cargar Archivo Excel para Predicción Masiva")
    
    st.info("""
    **Instrucciones:**
    1. Sube un archivo Excel (.xlsx) con los datos de pacientes
    2. El archivo debe contener columnas como: Presión Arterial, Temperatura, Glasgow, etc.
    3. El sistema detectará automáticamente las columnas y generará predicciones
    """)
    
    archivo_subido = st.file_uploader(
        "Selecciona archivo Excel",
        type=['xlsx'],
        help="Formato: .xlsx (Microsoft Excel)"
    )
    
    if archivo_subido is not None:
        # Cargar modelo
        modelo, metadata = cargar_modelo_entrenado()
        if modelo is None:
            return
        
        # Procesar archivo
        df_resultados = procesar_archivo_excel(archivo_subido, modelo, metadata)
        
        if df_resultados is not None:
            # Mostrar resultados
            st.success(f"✅ Predicciones generadas: {len(df_resultados):,} casos")
            
            # Estadísticas
            col1, col2, col3 = st.columns(3)
            
            total = len(df_resultados)
            pertinentes = (df_resultados['Decision_Final'] == 'PERTINENTE').sum()
            no_pertinentes = total - pertinentes
            
            with col1:
                st.metric("Total Casos", f"{total:,}")
            with col2:
                st.metric("PERTINENTE", f"{pertinentes:,}", 
                         delta=f"{(pertinentes/total*100):.1f}%")
            with col3:
                st.metric("NO PERTINENTE", f"{no_pertinentes:,}",
                         delta=f"{(no_pertinentes/total*100):.1f}%")
            
            # Mostrar tabla de resultados
            st.markdown("### 📋 Resultados Detallados")
            
            # Seleccionar columnas relevantes para mostrar
            cols_mostrar = []
            for col in ['ID', 'Episodio', 'Triage', 'Glasgow', 'PA_Sistolica', 
                       'Temperatura', 'Prediccion_IA', 'Confianza_IA', 
                       'Decision_Final', 'Metodo_Decision']:
                if col in df_resultados.columns:
                    cols_mostrar.append(col)
            
            if not cols_mostrar:
                cols_mostrar = list(df_resultados.columns[-5:])  # Últimas 5 columnas
            
            st.dataframe(
                df_resultados[cols_mostrar],
                use_container_width=True,
                height=400
            )
            
            # Botón de descarga
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_resultados.to_excel(writer, index=False, sheet_name='Predicciones')
            
            output.seek(0)
            
            st.download_button(
                label="📥 Descargar Resultados (Excel)",
                data=output,
                file_name="predicciones_ley_urgencia.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Validación si existe columna de resolución real
            if 'Resolucion' in df_resultados.columns:
                st.markdown("---")
                st.markdown("### 📊 Validación con Datos Reales")
                
                df_val = df_resultados[df_resultados['Resolucion'].notna()].copy()
                
                if len(df_val) > 0:
                    correctos = (df_val['Decision_Final'] == df_val['Resolucion']).sum()
                    accuracy = (correctos / len(df_val)) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Casos con Resolución", f"{len(df_val):,}")
                    with col2:
                        st.metric("Accuracy", f"{accuracy:.2f}%", 
                                 delta=f"{correctos}/{len(df_val)} correctos")
                    
                    # Matriz de confusión
                    st.markdown("**Matriz de Confusión:**")
                    matriz = pd.crosstab(
                        df_val['Resolucion'], 
                        df_val['Decision_Final'],
                        rownames=['Real'],
                        colnames=['Predicción']
                    )
                    st.dataframe(matriz, use_container_width=True)


def main():
    # Configuración de página
    st.set_page_config(
        page_title="Sistema IA - Ley de Urgencia",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personalizado
    st.markdown("""
        <style>
        .main {
            padding: 1rem 2rem;
        }
        h1 {
            color: #1e3a8a;
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1.5rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0 2rem 0;'>
            <h1 style='color: #1e3a8a; font-size: 2.5rem; margin-bottom: 0.5rem;'>
                🏥 Sistema Inteligente - Ley de Urgencia
            </h1>
            <p style='color: #64748b; font-size: 1.1rem; margin: 0;'>
                Evaluación de Criterios del Decreto 34 con IA
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Información del modelo
    with st.sidebar:
        st.markdown("### ℹ️ Información del Modelo")
        
        modelo, metadata = cargar_modelo_entrenado()
        
        if modelo is not None and metadata is not None:
            st.success("✅ Modelo cargado")
            
            with st.expander("📊 Detalles del Modelo"):
                st.write(f"**Versión:** {metadata.get('version', 'N/A')}")
                st.write(f"**Accuracy validación:** {metadata.get('accuracy_validacion', 0)*100:.2f}%")
                
                # Calcular total de casos (train + validación)
                casos_train = metadata.get('casos_entrenamiento', 0)
                casos_val = metadata.get('casos_validacion', 0)
                casos_total = casos_train + casos_val
                
                st.write(f"**Total casos dataset:** {casos_total:,}")
                st.write(f"  - Entrenamiento (80%): {casos_train:,}")
                st.write(f"  - Validación (20%): {casos_val:,}")
                st.write(f"**Features:** {len(metadata.get('feature_cols', []))}")
        else:
            st.error("❌ Modelo no disponible")
            st.info("Entrena el modelo con:\n```python entrenar_modelo_robusto.py```")
        
        st.markdown("---")
        st.markdown("### 🔧 Opciones")
        
        # Selector de modo
        modo = st.radio(
            "Modo de predicción:",
            options=["📋 Formulario Individual", "📤 Carga Masiva (Excel)"],
            index=1
        )
        
        st.markdown("---")
        st.markdown("""
        ### 📚 Guía Rápida
        
        **Formulario Individual:**
        - Ingresa datos de un paciente
        - Obtén predicción instantánea
        
        **Carga Masiva:**
        - Sube archivo Excel
        - Procesa múltiples casos
        - Descarga resultados
        """)
    
    # Contenido principal según modo seleccionado
    st.markdown("---")
    
    if modo == "📋 Formulario Individual":
        modo_formulario_individual()
    else:
        modo_carga_archivo()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #64748b; padding: 2rem 0;'>
            <p style='font-size: 0.9rem;'>
                🏥 <strong>Sistema Inteligente de Predicción - Ley de Urgencia (Decreto 34)</strong>
            </p>
            <p style='font-size: 0.85rem;'>
                Modelo: XGBoost Robusto | Versión: 3.0 | Entrenado con 26,606 casos
            </p>
            <p style='font-size: 0.8rem; color: #94a3b8;'>
                ⚠️ Herramienta de apoyo. Decisión final debe ser tomada por personal médico.
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
