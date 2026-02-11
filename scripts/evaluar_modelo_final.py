"""
EVALUACION MODELO ENTRENADO CON BASE + FORM_MPP
================================================
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Cargar modelo y componentes
print("Cargando modelo...")
modelo = joblib.load('../modelos/modelo_ley_urgencia.pkl')
preprocessor = joblib.load('../modelos/preprocessor.pkl')
scaler = joblib.load('../modelos/scaler.pkl')
metadata = joblib.load('../modelos/modelo_metadata.pkl')

print(f"  Modelo entrenado: {metadata['fecha_entrenamiento']}")
print(f"  Casos entrenamiento: {metadata['casos_totales']}")
print(f"  Test accuracy: {metadata['accuracy_test']:.4f}")

# Cargar form_MPP.xlsx
print("\nCargando form_MPP.xlsx...")
df = pd.read_excel('../data/form_MPP.xlsx')

# Filtrar solo casos validados
df_validados = df[df['Validacion'].notna()].copy()
print(f"  Total casos: {len(df)}")
print(f"  Casos validados: {len(df_validados)}")

# Mapeo de columnas
mapeo_columnas = {
    'Frecuencia Cardíaca': 'FC',
    'Frecuencia Respiratoria': 'FR',
    'Presión Arterial Sistólica': 'PAS',
    'Presión Arterial Diastólica': 'PAD',
    'Saturación Oxígeno': 'SatO2',
    'Temperatura en °C': 'Temp',
    'Glasgow': 'Glasgow',
    'Triage': 'Triage',
    'Antecedentes de Hipertensión Arterial': 'HipertencionArterial',
    'Antecedentes Diabéticos': 'DiabetesMellitus',
    'Antecedentes Cardíacos': 'Cardiopatia'
}

df_validados = df_validados.rename(columns=mapeo_columnas)

# Convertir columnas numéricas
columnas_numericas = ['FC', 'FR', 'PAS', 'PAD', 'SatO2', 'Temp', 'Glasgow', 'Triage']

for col in columnas_numericas:
    if col in df_validados.columns:
        df_validados[col] = pd.to_numeric(df_validados[col], errors='coerce')

# Convertir columnas binarias
columnas_binarias = ['HipertencionArterial', 'DiabetesMellitus', 'Cardiopatia']

for col in columnas_binarias:
    if col in df_validados.columns:
        df_validados[col] = df_validados[col].astype(str).str.strip().str.lower()
        df_validados[col] = df_validados[col].map({
            'si': 1, 'sí': 1, 's': 1, '1': 1, 
            'no': 0, 'n': 0, '0': 0
        })
        df_validados[col] = pd.to_numeric(df_validados[col], errors='coerce')

# Features derivadas
if 'SatO2' in df_validados.columns and 'FR' in df_validados.columns:
    df_validados['Ratio_SatO2_FR'] = df_validados['SatO2'] / (df_validados['FR'] + 1)

if 'PAS' in df_validados.columns and 'PAD' in df_validados.columns:
    df_validados['Flag_Hipotension'] = ((df_validados['PAS'] < 90) | (df_validados['PAD'] < 60)).astype(int)
    df_validados['Ratio_PAM'] = (df_validados['PAS'] + 2*df_validados['PAD']) / 3

if 'FC' in df_validados.columns:
    df_validados['Flag_Taquicardia'] = (df_validados['FC'] > 100).astype(int)

if 'Temp' in df_validados.columns:
    df_validados['Flag_Fiebre'] = (df_validados['Temp'] > 38).astype(int)

if 'Glasgow' in df_validados.columns:
    df_validados['Flag_GlasgowBajo'] = (df_validados['Glasgow'] < 13).astype(int)

# Score gravedad
score_components = []
for flag in ['Flag_Hipotension', 'Flag_Taquicardia', 'Flag_Fiebre', 'Flag_GlasgowBajo']:
    if flag in df_validados.columns:
        score_components.append(df_validados[flag])

if score_components:
    df_validados['Score_Gravedad'] = sum(score_components)

if 'Triage' in df_validados.columns:
    df_validados['Flag_TriageCritico'] = (df_validados['Triage'] <= 2).astype(int)

# Preparar X (solo features del modelo)
features_modelo = metadata['features']
X = df_validados[features_modelo]

# Preparar y (target)
y_true = df_validados['Validacion'].astype(str).str.strip().str.upper()
y_true = y_true.map({'PERTINENTE': 1, 'NO PERTINENTE': 0})

# Aplicar pipeline de preprocesamiento
X_prep = preprocessor.transform(X)
X_scaled = scaler.transform(X_prep)

# Predecir
print("\nRealizando predicciones...")
y_pred = modelo.predict(X_scaled)
y_proba = modelo.predict_proba(X_scaled)

# Métricas
accuracy = accuracy_score(y_true, y_pred)

print("\n" + "="*70)
print("RESULTADOS EN form_MPP.xlsx (525 CASOS VALIDADOS)")
print("="*70)

print(f"\nACCURACY: {accuracy:.4f} ({accuracy:.2%})")
print(f"Aciertos: {(y_true == y_pred).sum()}/{len(y_true)}")
print(f"Errores: {(y_true != y_pred).sum()}/{len(y_true)}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['NO PERTINENTE', 'PERTINENTE']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(f"\n                Predicho")
print(f"                NO PERT    PERTINENTE")
print(f"Real NO PERT    {cm[0][0]:6d}      {cm[0][1]:6d}")
print(f"Real PERTINENTE {cm[1][0]:6d}      {cm[1][1]:6d}")

# Confianza
confianza_promedio = y_proba.max(axis=1).mean()
print(f"\nConfianza promedio: {confianza_promedio:.2%}")

# Guardar resultados
df_resultados = df_validados[['Episodio']].copy()
df_resultados['Real'] = y_true.map({1: 'PERTINENTE', 0: 'NO PERTINENTE'})
df_resultados['Prediccion'] = pd.Series(y_pred).map({1: 'PERTINENTE', 0: 'NO PERTINENTE'})
df_resultados['Confianza'] = y_proba.max(axis=1)
df_resultados['Correcto'] = (y_true == y_pred)

df_resultados.to_excel('../resultados/resultados_form_mpp_FINAL.xlsx', index=False)
print(f"\nResultados guardados en: resultados_form_mpp_FINAL.xlsx")

# Errores
errores = df_resultados[~df_resultados['Correcto']].copy()
if len(errores) > 0:
    errores.to_csv('../resultados/errores_form_mpp_FINAL.csv', index=False)
    print(f"Errores guardados en: errores_form_mpp_FINAL.csv ({len(errores)} casos)")

print("\n" + "="*70)
print("COMPARACION CON MODELO ANTERIOR")
print("="*70)
print(f"\nModelo ANTERIOR (solo Base.xlsx):     69.71% (366/525 correctos, 159 errores)")
print(f"Modelo NUEVO (Base + form_MPP):       {accuracy:.2%} ({(y_true == y_pred).sum()}/525 correctos, {(y_true != y_pred).sum()} errores)")
print(f"\nMEJORA: {accuracy - 0.6971:.2%} ({(y_true == y_pred).sum() - 366:+d} casos)")
