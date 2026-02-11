"""
REENTRENAMIENTO CON BASE.xlsx + form_MPP.xlsx CASOS VALIDADOS
=============================================================
Combina ambos datasets para que el modelo aprenda de los casos de producción
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
import joblib
from datetime import datetime

def convertir_binarios(df, columnas_binarias):
    """Convierte columnas Si/No a 1/0"""
    df_copy = df.copy()
    for col in columnas_binarias:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.strip().str.lower()
            df_copy[col] = df_copy[col].map({
                'si': 1, 'sí': 1, 's': 1, '1': 1, 
                'no': 0, 'n': 0, '0': 0
            })
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    return df_copy

def preparar_form_mpp(filepath):
    """Lee y prepara form_MPP.xlsx con mapeo de columnas"""
    df = pd.read_excel(filepath)
    
    # Filtrar solo casos CON validación manual PRIMERO
    df = df[df['Validacion'].notna()].copy()
    
    # Mapeo COMPLETO de columnas form_MPP -> Base
    mapeo_columnas = {
        # Variables básicas
        'Frecuencia Cardiaca': 'FC',
        'Frecuencia Respiratoria': 'FR', 
        'Presión Arterial Sistólica': 'PAS',
        'Presión Arterial Diastólica': 'PAD',
        'Saturación de Oxígeno': 'SatO2',
        'Temperatura en °C': 'Temp',
        'Escala de Dolor (EVA)': 'Dolor',
        'Glasgow': 'Glasgow',
        'Triage': 'Triage',
        'Diagnóstico de Ingreso': 'Diagnostico',
        'Horas de Observación': 'HorasObservacion',
        'Número de Exámenes de Laboratorio': 'NumExamenes',
        'Número de Procedimientos Realizados': 'NumProcedimientos',
        'Número de Diagnósticos de Ingreso': 'NumDiagnosticos',
        'Número de Diagnósticos de Egreso': 'NumDiagnostEgreso',
        'Antecedentes Mórbidos': 'AntMorbidos',
        'Cantidad de Medicamentos administrados': 'CantMedicamentos',
        
        # Diagnósticos por sistema
        'Diagnóstico Cardiovascular': 'DiagnosticoCardiovascular',
        'Diagnóstico Respiratorio': 'DiagnosticoRespiratorio',
        'Diagnóstico Digestivo': 'DiagnosticoDigestivo',
        'Diagnóstico Neurológico': 'DiagnosticoNeurologico',
        'Diagnóstico Traumatológico': 'DiagnosticoTraumatologico',
        'Diagnóstico Infeccioso': 'DiagnosticoInfeccioso',
        'Diagnóstico Oncológico': 'DiagnosticoOncologico',
        'Diagnóstico Psiquiátrico': 'DiagnosticoPsiquiatrico',
        'Diagnóstico Ginecológico': 'DiagnosticoGinecologico',
        'Diagnóstico Urológico': 'DiagnosticoUrologico',
        'Diagnóstico Dermatológico': 'DiagnosticoDermatologico',
        'Diagnóstico Oftalmológico': 'DiagnosticoOftalmologico',
        
        # Antecedentes
        'Antecedentes de Hipertensión Arterial': 'HipertencionArterial',
        'Antecedentes Diabéticos': 'DiabetesMellitus',
        'Antecedentes Cardíacos': 'Cardiopatia',
        'Asma': 'Asma',
        'Epilepsia': 'Epilepsia',
        'Artrosis': 'Artrosis',
        'Tabaquismo': 'DislipidemiaTabaquismo',
        'Cirugía Previa': 'CirugiaPrevia',
        'Alergias a Medicamentos': 'AlergiasMedicamentos',
        'Hospitalización Reciente': 'HospitalizacionReciente'
    }
    
    # Renombrar columnas
    df = df.rename(columns=mapeo_columnas)
    
    print(f"OK form_MPP.xlsx: {len(df)} casos validados")
    
    return df

def preparar_base(filepath):
    """Lee Base.xlsx y normaliza nombres de columnas"""
    df = pd.read_excel(filepath)
    
    # Normalizar nombres de columnas para que coincidan con el modelo
    mapeo_normalizacion = {
        'Saturacion_O2': 'SatO2',
        'Temperatura en °C': 'Temp',
        'Horas_Observacion': 'HorasObservacion',
        'Numero_Examenes': 'NumExamenes',
        'Numero_Procedimientos': 'NumProcedimientos',
        'Numero_Diagnosticos_Ingreso': 'NumDiagnosticos',
        'Numero_Diagnosticos_Egreso': 'NumDiagnostEgreso',
        'Antecedentes_Morbidos': 'AntMorbidos',
        'Cantidad_Medicamentos': 'CantMedicamentos',
        'Resolucion': 'Validacion'  # En Base.xlsx se llama Resolucion
    }
    
    df = df.rename(columns=mapeo_normalizacion)
    
    print(f"OK Base.xlsx: {len(df)} casos")
    
    return df

def prepare_dataframe(df):
    """Prepara features igual que en entrenar_modelo_final.py"""
    df_prep = df.copy()
    
    # Convertir numéricas
    columnas_numericas = ['FC', 'FR', 'PAS', 'PAD', 'SatO2', 'Temp', 'Dolor', 
                          'Glasgow', 'Triage', 'HorasObservacion', 'NumExamenes',
                          'NumProcedimientos', 'NumDiagnosticos', 'NumDiagnostEgreso',
                          'CantMedicamentos']
    
    for col in columnas_numericas:
        if col in df_prep.columns:
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')
    
    # Convertir binarias
    columnas_binarias = [
        'DiagnosticoCardiovascular', 'DiagnosticoRespiratorio', 'DiagnosticoDigestivo',
        'DiagnosticoNeurologico', 'DiagnosticoTraumatologico', 'DiagnosticoInfeccioso',
        'DiagnosticoOncologico', 'DiagnosticoPsiquiatrico', 'DiagnosticoGinecologico',
        'DiagnosticoUrologico', 'DiagnosticoDermatologico', 'DiagnosticoOftalmologico',
        'HipertencionArterial', 'DiabetesMellitus', 'Cardiopatia', 'Asma', 
        'Epilepsia', 'Artrosis', 'DislipidemiaTabaquismo', 'AntMorbidos',
        'CirugiaPrevia', 'AlergiasMedicamentos', 'HospitalizacionReciente'
    ]
    
    df_prep = convertir_binarios(df_prep, columnas_binarias)
    
    # Features derivadas - solo si las columnas existen
    if 'SatO2' in df_prep.columns and 'FR' in df_prep.columns:
        df_prep['Ratio_SatO2_FR'] = df_prep['SatO2'] / (df_prep['FR'] + 1)
    
    if 'PAS' in df_prep.columns and 'PAD' in df_prep.columns:
        df_prep['Flag_Hipotension'] = ((df_prep['PAS'] < 90) | (df_prep['PAD'] < 60)).astype(int)
        df_prep['Ratio_PAM'] = (df_prep['PAS'] + 2*df_prep['PAD']) / 3
    
    if 'FC' in df_prep.columns:
        df_prep['Flag_Taquicardia'] = (df_prep['FC'] > 100).astype(int)
    
    if 'Temp' in df_prep.columns:
        df_prep['Flag_Fiebre'] = (df_prep['Temp'] > 38).astype(int)
    
    if 'Dolor' in df_prep.columns:
        df_prep['Flag_DolorSevero'] = (df_prep['Dolor'] >= 7).astype(int)
    
    if 'Glasgow' in df_prep.columns:
        df_prep['Flag_GlasgowBajo'] = (df_prep['Glasgow'] < 13).astype(int)
    
    # Score gravedad solo con flags existentes
    score_components = []
    for flag in ['Flag_Hipotension', 'Flag_Taquicardia', 'Flag_Fiebre', 
                  'Flag_DolorSevero', 'Flag_GlasgowBajo']:
        if flag in df_prep.columns:
            score_components.append(df_prep[flag])
    
    if score_components:
        df_prep['Score_Gravedad'] = sum(score_components)
    
    if all(col in df_prep.columns for col in ['NumExamenes', 'NumProcedimientos', 'CantMedicamentos']):
        df_prep['TotalActividad'] = (
            df_prep['NumExamenes'].fillna(0) + 
            df_prep['NumProcedimientos'].fillna(0) +
            df_prep['CantMedicamentos'].fillna(0)
        )
        df_prep['Flag_AltaActividad'] = (df_prep['TotalActividad'] > 5).astype(int)
    
    if 'NumExamenes' in df_prep.columns and 'HorasObservacion' in df_prep.columns:
        df_prep['Ratio_Examen_Hora'] = df_prep['NumExamenes'] / (df_prep['HorasObservacion'] + 1)
    
    if 'Triage' in df_prep.columns:
        df_prep['Flag_TriageCritico'] = (df_prep['Triage'] <= 2).astype(int)
    
    return df_prep

# ============================================================
# CARGA Y COMBINACIÓN DE DATOS
# ============================================================

print("="*70)
print("ENTRENAMIENTO COMBINADO: Base.xlsx + form_MPP.xlsx")
print("="*70)

# Cargar ambos datasets
df_base = preparar_base('../data/Base.xlsx')
df_form = preparar_form_mpp('../data/form_MPP.xlsx')

# Ya no necesitamos buscar columnas comunes - form_MPP ya está mapeado

# Combinar datasets usando solo columnas comunes
# Usar todas las columnas necesarias para el modelo
df_combinado = pd.concat([
    df_base,
    df_form
], ignore_index=True)

print(f"\nDATASET COMBINADO: {len(df_combinado)} casos totales")
print(f"   - De Base.xlsx: {len(df_base)}")
print(f"   - De form_MPP.xlsx: {len(df_form)}")
print(f"   - Columnas totales: {len(df_combinado.columns)}")

# Preparar features
df_prep = prepare_dataframe(df_combinado)

# Convertir target
df_prep['Validacion'] = df_prep['Validacion'].astype(str).str.strip().str.upper()
df_prep['Validacion'] = df_prep['Validacion'].map({'PERTINENTE': 1, 'NO PERTINENTE': 0})
df_prep = df_prep.dropna(subset=['Validacion'])

print(f"\nOK Casos finales: {len(df_prep)}")
print("\nDistribucion:")
print(df_prep['Validacion'].value_counts())

# ============================================================
# PREPARAR FEATURES Y TARGET
# ============================================================

features_modelo = [
    'FC', 'FR', 'PAS', 'PAD', 'SatO2', 'Temp', 'Dolor', 'Glasgow', 'Triage',
    'HorasObservacion', 'NumExamenes', 'NumProcedimientos', 'NumDiagnosticos',
    'NumDiagnostEgreso', 'CantMedicamentos',
    'DiagnosticoCardiovascular', 'DiagnosticoRespiratorio', 'DiagnosticoDigestivo',
    'DiagnosticoNeurologico', 'DiagnosticoTraumatologico', 'DiagnosticoInfeccioso',
    'DiagnosticoOncologico', 'DiagnosticoPsiquiatrico', 'DiagnosticoGinecologico',
    'DiagnosticoUrologico', 'DiagnosticoDermatologico', 'DiagnosticoOftalmologico',
    'HipertencionArterial', 'DiabetesMellitus', 'Cardiopatia', 'Asma', 
    'Epilepsia', 'Artrosis', 'DislipidemiaTabaquismo', 'AntMorbidos',
    'CirugiaPrevia', 'AlergiasMedicamentos', 'HospitalizacionReciente',
    'Ratio_SatO2_FR', 'Flag_Hipotension', 'Flag_Taquicardia', 'Flag_Fiebre',
    'Flag_DolorSevero', 'Flag_GlasgowBajo', 'Score_Gravedad', 'Ratio_PAM',
    'TotalActividad', 'Flag_AltaActividad', 'Ratio_Examen_Hora', 'Flag_TriageCritico'
]

# Filtrar solo features que existan
features_disponibles = [f for f in features_modelo if f in df_prep.columns]
print(f"\nFeatures usadas: {len(features_disponibles)}")

X = df_prep[features_disponibles]
y = df_prep['Validacion']

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ============================================================
# PIPELINE Y ENTRENAMIENTO
# ============================================================

# Separar columnas numéricas y categóricas
columnas_numericas = X_train.select_dtypes(include=[np.number]).columns.tolist()
columnas_categoricas = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Pipeline de preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', KNNImputer(n_neighbors=5), columnas_numericas),
        ('cat', 'passthrough', columnas_categoricas)
    ],
    remainder='drop'
)

# Preprocesar datos
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_prep)
X_test_scaled = scaler.transform(X_test_prep)

# ============================================================
# MODELO RANDOM FOREST
# ============================================================

print("\nEntrenando Random Forest...")

modelo = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

modelo.fit(X_train_scaled, y_train)

# ============================================================
# EVALUACIÓN
# ============================================================

print("\n" + "="*70)
print("RESULTADOS")
print("="*70)

# Predicciones
y_pred_train = modelo.predict(X_train_scaled)
y_pred_test = modelo.predict(X_test_scaled)

# Métricas Train
acc_train = accuracy_score(y_train, y_pred_train)
print(f"\nTRAIN Accuracy: {acc_train:.4f}")

# Métricas Test
acc_test = accuracy_score(y_test, y_pred_test)
print(f"\nTEST Accuracy: {acc_test:.4f}")

print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test, 
                          target_names=['NO PERTINENTE', 'PERTINENTE']))

print("\nConfusion Matrix (Test):")
cm = confusion_matrix(y_test, y_pred_test)
print(f"\n                Predicho")
print(f"                NO PERT    PERTINENTE")
print(f"Real NO PERT    {cm[0][0]:6d}      {cm[0][1]:6d}")
print(f"Real PERTINENTE {cm[1][0]:6d}      {cm[1][1]:6d}")

# Cross-validation
print("\nCross-Validation (5-fold):")
cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5, n_jobs=-1)
print(f"   Scores: {cv_scores}")
print(f"   Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================
# GUARDAR MODELO
# ============================================================

print("\nGuardando modelo...")

# Mover modelo anterior
import os
from datetime import datetime

if os.path.exists('../modelos/modelo_ley_urgencia.pkl'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.rename('../modelos/modelo_ley_urgencia.pkl', 
              f'../archive_modelos_antiguos/modelo_ANTES_FORM_MPP_{timestamp}.pkl')
    print("   OK Modelo anterior archivado")

# Guardar nuevo modelo y componentes
joblib.dump(modelo, '../modelos/modelo_ley_urgencia.pkl')
joblib.dump(preprocessor, '../modelos/preprocessor.pkl')
joblib.dump(scaler, '../modelos/scaler.pkl')

# Metadata
metadata = {
    'fecha_entrenamiento': datetime.now().isoformat(),
    'casos_totales': len(df_combinado),
    'casos_base': len(df_base),
    'casos_form_mpp': len(df_form),
    'accuracy_train': float(acc_train),
    'accuracy_test': float(acc_test),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'features': features_disponibles,
    'modelo': 'RandomForestClassifier',
    'n_estimators': 500,
    'max_depth': 20
}

joblib.dump(metadata, '../modelos/modelo_metadata.pkl')

print("   OK modelo_ley_urgencia.pkl")
print("   OK preprocessor.pkl")
print("   OK scaler.pkl")
print("   OK modelo_metadata.pkl")

print("\n" + "="*70)
print("ENTRENAMIENTO COMPLETADO")
print("="*70)
print(f"\nTest Accuracy: {acc_test:.2%}")
print(f"CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
