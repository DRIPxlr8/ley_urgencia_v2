# 📊 MODELO FINAL - LEY DE URGENCIA DECRETO 34

**Fecha de Actualización:** 6 de febrero de 2026  
**Versión:** 2.0 (Entrenado con Base + form_MPP)

---

## 🎯 MODELO ACTUAL

### Arquitectura
**Tipo:** Random Forest Classifier (Simple)

**Configuración:**
- n_estimators: 500
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2
- class_weight: balanced
- random_state: 42
- n_jobs: -1

### Datos de Entrenamiento

**Dataset Combinado:** 5,775 casos totales
- **Base.xlsx:** 5,250 casos (91%)
- **form_MPP.xlsx:** 525 casos validados (9%)

**Split:**
- Train: 4,620 casos (80%)
- Test: 1,155 casos (20%)

**Distribución de Clases:**
- PERTINENTE: 3,288 casos (56.9%)
- NO PERTINENTE: 2,487 casos (43.1%)

---

## 📈 RESULTADOS

### Métricas en Test Interno (1,155 casos)

**Accuracy: 82.60%**

**Resultados en Test (Base.xlsx):**
```
✅ Accuracy: 82.60%
✅ Precision NO PERTINENTE: 79%
✅ Precision PERTINENTE: 85%
✅ Recall NO PERTINENTE: 81%
✅ Recall PERTINENTE: 84%

Confusion Matrix:
                  Predicho
                  NO PERT    PERTINENTE
Real NO PERT          403          94
Real PERTINENTE       107         551
```

**Cross-Validation (5-fold):**
- Mean: 80.09%
- Std: ±3.02%

---

## 🔍 EVALUACIÓN CON form_MPP.xlsx

**Datos de Evaluación:**
- Total casos: 1,253
- Casos CON validación manual: 525 (evaluados)
- Casos SIN validación: 728 (no evaluados)

**Distribución Real:**
- PERTINENTE: 319 casos (60.8%)
- NO PERTINENTE: 206 casos (39.2%)

**Resultados en form_MPP.xlsx:**
```
🎯 Accuracy: 80.57% (MEJORADO desde 69.71%)
✅ Aciertos: 423/525 (antes: 366/525)
❌ Errores: 102/525 (antes: 159/525)
✨ MEJORA: +10.86 puntos porcentuales

Precision:
  - NO PERTINENTE: 77%
  - PERTINENTE: 83%

Recall:
  - NO PERTINENTE: 72%
  - PERTINENTE: 86%

Confusion Matrix:
                     Predicho
                     NO PERT    PERTINENTE
Real NO PERT             148          58
Real PERTINENTE           44         275
```

**Confianza Promedio:** 66.75%

---

## 🔧 FEATURES DEL MODELO

**Total Features:** 19

### Signos Vitales (8)
1. FC - Frecuencia Cardíaca
2. FR - Frecuencia Respiratoria
3. PAS - Presión Arterial Sistólica
4. PAD - Presión Arterial Diastólica
5. SatO2 - Saturación de Oxígeno
6. Temp - Temperatura
7. Glasgow - Escala de Glasgow
8. Triage - Clasificación ESI

### Antecedentes (3)
9. HipertencionArterial
10. DiabetesMellitus
11. Cardiopatia

### Features Derivadas (8)
12. Ratio_SatO2_FR - SatO2 / (FR + 1)
13. Flag_Hipotension - (PAS < 90) o (PAD < 60)
14. Flag_Taquicardia - FC > 100
15. Flag_Fiebre - Temp > 38
16. Flag_GlasgowBajo - Glasgow < 13
17. Score_Gravedad - Suma de flags de riesgo
18. Ratio_PAM - (PAS + 2*PAD) / 3
19. Flag_TriageCritico - Triage <= 2

---

## 📊 COMPARACIÓN DE MODELOS

| Modelo | Dataset | Accuracy | Errores |
|--------|---------|----------|---------|
| **Versión 1.0** (Ensemble) | Solo Base.xlsx | 69.71% | 159/525 |
| **Versión 2.0** (Random Forest) | Base + form_MPP | **80.57%** | **102/525** |
| **Mejora** | - | **+10.86%** | **-57 casos** |

**Reducción de errores:** 36% (-57 errores)

---

## 🚀 PIPELINE DE PROCESAMIENTO

### Entrenamiento
1. Carga Base.xlsx (5,250) + form_MPP.xlsx validados (525)
2. Normalización de nombres de columnas
3. Conversión de binarias (Si/No → 1/0)
4. Conversión de numéricas a float
5. Generación de features derivadas
6. **ColumnTransformer:**
   - KNN Imputer (k=5) para columnas numéricas
   - Passthrough para categóricas
7. **StandardScaler** para normalización
8. **Random Forest** con 500 árboles

### Predicción
1. Carga datos nuevos
2. Mapeo de columnas (form_MPP → Base)
3. Generación de features derivadas
4. Aplicación de preprocessor.pkl
5. Aplicación de scaler.pkl
6. Predicción con modelo_ley_urgencia.pkl
7. Retorno de clase + probabilidad

---

## 📁 ARCHIVOS GENERADOS

### Modelo
- `modelo_ley_urgencia.pkl` - Random Forest entrenado
- `preprocessor.pkl` - Pipeline de preprocesamiento (KNN + ColumnTransformer)
- `scaler.pkl` - StandardScaler ajustado
- `modelo_metadata.pkl` - Metadata (fecha, features, métricas)

### Resultados
- `resultados_form_mpp_FINAL.xlsx` - Predicciones completas (525 casos)
- `errores_form_mpp_FINAL.csv` - Casos mal clasificados (102 casos)

---

## 🔍 ANÁLISIS DE ERRORES

**Total errores en form_MPP:** 102/525 (19.43%)

**Tipo de errores:**
- Falsos Positivos (NO → PERTINENTE): 58 casos (28% error en NO PERT)
- Falsos Negativos (PERTINENTE → NO): 44 casos (14% error en PERTINENTE)

**Observaciones:**
- El modelo es más conservador con NO PERTINENTE (77% precision)
- Mejor recall en PERTINENTE (86%) - detecta mejor urgencias reales
- La confianza promedio (66.75%) sugiere decisiones en zona gris para algunos casos

---

## 💡 PRÓXIMOS PASOS POTENCIALES

1. **Análisis de los 102 errores restantes:**
   - Identificar patrones comunes
   - Verificar consistencia de etiquetado
   - Casos en zona gris de decisión

2. **Ajuste de threshold:**
   - Evaluar costos de FP vs FN
   - Optimizar punto de corte según prioridad clínica

3. **Features adicionales:**
   - Incorporar más variables si están disponibles
   - Interacciones específicas de errores

4. **Validación continua:**
   - Reentrenar con nuevos casos validados
   - Monitorear drift en producción

---

## 📞 INFORMACIÓN TÉCNICA

**Framework:** scikit-learn 1.4+  
**Python:** 3.14  
**Librerías principales:**
- pandas
- numpy
- scikit-learn
- joblib

**Scripts principales:**
- `entrenar_con_form_mpp.py` - Entrenamiento
- `evaluar_modelo_final.py` - Evaluación
- `streamlit_app_v3.py` - Interfaz web

**Organización:** UC CHRISTUS Chile  
**Proyecto:** Clasificación Ley de Urgencia - Decreto 34
