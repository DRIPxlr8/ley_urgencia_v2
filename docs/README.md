# Sistema ML Ley de Urgencia - Decreto 34

Sistema de Machine Learning para clasificar atenciones de urgencia como PERTINENTE o NO PERTINENTE según Decreto 34 (Chile).

## 📁 Estructura del Proyecto

```
ley_urgencia/
├── data/                              # Datos de entrenamiento
│   ├── Base.xlsx                      # 5,250 casos etiquetados (entrenamiento)
│   └── form_MPP.xlsx                  # 1,253 casos producción (525 validados)
│
├── tools/                             # Herramientas auxiliares
│   └── conversor_archivos.py          # Convertir Excel/CSV/JSON
│
├── entrenar_con_form_mpp.py          # ⭐ SCRIPT PRINCIPAL DE ENTRENAMIENTO
├── evaluar_modelo_final.py           # ⭐ SCRIPT DE EVALUACIÓN
├── streamlit_app_v3.py               # Interfaz web interactiva
├── ejecutar_streamlit.py             # Launcher de Streamlit
│
├── modelo_ley_urgencia.pkl           # Modelo entrenado (Random Forest)
├── preprocessor.pkl                  # Pipeline de preprocesamiento
├── scaler.pkl                        # Escalador de features
├── modelo_metadata.pkl               # Metadata del modelo
│
├── resultados_form_mpp_FINAL.xlsx    # Últimos resultados
├── errores_form_mpp_FINAL.csv        # Casos donde el modelo falla
├── RESUMEN_MODELO_FINAL.md           # Documentación técnica
├── requirements.txt                  # Dependencias Python
└── README.md                         # Este archivo
```

## 🚀 Uso Rápido

### 1. Entrenar Modelo

Entrena con Base.xlsx (5,250 casos) + form_MPP.xlsx (525 validados):

```bash
python entrenar_con_form_mpp.py
```

**Salida:**
- `modelo_ley_urgencia.pkl` - Modelo Random Forest
- `preprocessor.pkl` - Pipeline de preprocesamiento
- `scaler.pkl` - Escalador
- `modelo_metadata.pkl` - Metadata

### 2. Evaluar Modelo

Evalúa el modelo en los 525 casos validados de form_MPP.xlsx:

```bash
python evaluar_modelo_final.py
```

**Salida:**
- `resultados_form_mpp_FINAL.xlsx` - Predicciones completas
- `errores_form_mpp_FINAL.csv` - Casos mal clasificados

### 3. Interfaz Web

Lanza la aplicación Streamlit (actualizada con modelo 2.0):

```bash
python ejecutar_streamlit.py
# o directamente:
streamlit run streamlit_app_v3.py
```

**Características:**
- **Modo Individual:** Formulario para casos únicos
- **Modo Masivo:** Carga de archivos Excel para predicciones en lote
- **Métricas en tiempo real:** Confianza y probabilidades
- **Alertas clínicas:** Detección automática de signos vitales críticos
- **Descarga de resultados:** Export a Excel con todas las predicciones

## 📊 Resultados Actuales

**Modelo:** Random Forest (500 árboles, profundidad 20)
**Entrenamiento:** 5,775 casos (Base.xlsx + form_MPP.xlsx validados)
**Features:** 19 características (vitales + flags + scores)

### Métricas en Test (interno)
- Accuracy: **82.60%**
- Precision NO PERTINENTE: 79%
- Precision PERTINENTE: 85%
- Recall NO PERTINENTE: 81%
- Recall PERTINENTE: 84%

### Métricas en Producción (form_MPP.xlsx - 525 casos)
- Accuracy: **80.57%**
- Aciertos: 423/525
- Errores: 102/525
- Precision NO PERTINENTE: 77%
- Precision PERTINENTE: 83%
- Recall NO PERTINENTE: 72%
- Recall PERTINENTE: 86%

### Evolución
- Modelo inicial (solo Base.xlsx): 69.71%
- **Modelo actual (Base + form_MPP): 80.57%**
- **Mejora: +10.86 puntos porcentuales**

## 🔧 Características (Features)

El modelo utiliza 19 features:

**Signos Vitales:**
- FC (Frecuencia Cardíaca)
- FR (Frecuencia Respiratoria)
- PAS (Presión Arterial Sistólica)
- PAD (Presión Arterial Diastólica)
- SatO2 (Saturación de Oxígeno)
- Temp (Temperatura)
- Glasgow (Escala de Glasgow)
- Triage (Clasificación ESI)

**Antecedentes:**
- HipertencionArterial
- DiabetesMellitus
- Cardiopatia

**Features Derivadas:**
- Ratio_SatO2_FR
- Flag_Hipotension (PAS<90 o PAD<60)
- Flag_Taquicardia (FC>100)
- Flag_Fiebre (Temp>38)
- Flag_GlasgowBajo (Glasgow<13)
- Score_Gravedad (suma de flags)
- Ratio_PAM (Presión Arterial Media)
- Flag_TriageCritico (Triage≤2)

## 📝 Notas Técnicas

### Pipeline de Entrenamiento
1. Carga Base.xlsx + form_MPP.xlsx validados
2. Normaliza nombres de columnas
3. Convierte binarias (Si/No → 1/0)
4. Crea features derivadas
5. Aplica KNN Imputer (5 vecinos)
6. Escala con StandardScaler
7. Entrena Random Forest
8. Valida con cross-validation (5-fold)

### Pipeline de Predicción
1. Carga form_MPP.xlsx
2. Mapea columnas a formato modelo
3. Genera features derivadas
4. Aplica preprocessor + scaler
5. Predice con Random Forest
6. Retorna clase + confianza

## 📂 Archivos Archivados

- `archive_modelos_antiguos/` - Modelos anteriores
- `archive_excel_antiguos/` - Datos históricos
- `archive_scripts_analisis/` - Scripts de análisis/optimización

## 🔄 Flujo de Trabajo

```
┌─────────────────┐
│   Base.xlsx     │ 5,250 casos
│ (entrenamiento) │
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
         v                  v
┌─────────────────┐   ┌─────────────────┐
│  form_MPP.xlsx  │   │  entrenar_con_  │
│ (525 validados) │──▶│   form_mpp.py   │
└─────────────────┘   └────────┬────────┘
                               │
                               v
                      ┌─────────────────┐
                      │ modelo_ley_     │
                      │ urgencia.pkl    │
                      └────────┬────────┘
                               │
                               v
                      ┌─────────────────┐
                      │ evaluar_modelo_ │
                      │    final.py     │
                      └────────┬────────┘
                               │
                               v
                      ┌─────────────────┐
                      │  resultados_    │
                      │ form_mpp_FINAL  │
                      └─────────────────┘
```

## 📞 Información del Proyecto

**Organización:** UC CHRISTUS Chile  
**Tema:** Clasificación Ley de Urgencia - Decreto 34  
**Última actualización:** Febrero 6, 2026  
**Modelo actual:** Random Forest (80.57% accuracy)
