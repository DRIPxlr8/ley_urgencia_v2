"""
EVALUACIÓN DEL MODELO CON form_MPP.xlsx
Evalúa accuracy solo en los casos que tienen validación manual
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔍 EVALUACIÓN DEL MODELO CON form_MPP.xlsx")
print("="*80)

# ============================================================
# 1. CARGAR MODELO
# ============================================================
print("\n📂 Cargando modelo entrenado...")

try:
    modelo = joblib.load('modelo_ley_urgencia.pkl')
    metadata = joblib.load('modelo_metadata.pkl')
    print("   ✅ Modelo cargado exitosamente")
except FileNotFoundError:
    print("   ❌ ERROR: No se encontró el modelo entrenado")
    print("   Ejecuta primero: python entrenar_modelo_final.py")
    exit(1)

print(f"\n   Modelo: {metadata.get('model_type', 'N/A')}")
print(f"   Features: {len(metadata['features'])}")
print(f"   Accuracy en entrenamiento: {metadata.get('accuracy_test', 0)*100:.2f}%")

# ============================================================
# 2. CARGAR DATOS DE EVALUACIÓN
# ============================================================
print("\n📂 Cargando form_MPP.xlsx...")

df = pd.read_excel('data/form_MPP.xlsx')

print(f"   Total registros: {len(df):,}")

# Verificar columna de validación
if 'Validacion' not in df.columns:
    print("   ❌ ERROR: No se encontró columna 'Validacion'")
    exit(1)

# Filtrar solo casos CON validación manual
df_con_validacion = df[df['Validacion'].notna()].copy()

print(f"   Casos CON validación: {len(df_con_validacion):,}")
print(f"   Casos SIN validación: {len(df) - len(df_con_validacion):,}")

if len(df_con_validacion) == 0:
    print("   ❌ No hay casos con validación para evaluar")
    exit(1)

print(f"\n   Distribución de validaciones:")
print(df_con_validacion['Validacion'].value_counts())

# ============================================================
# 3. PREPARAR FEATURES
# ============================================================
print("\n🔧 Preparando features...")

# Mapear nombres de columnas de form_MPP a Base.xlsx
mapeo_columnas = {
    'Presión Arterial Sistólica': 'PAS',
    'Presión Arterial Diastólica': 'PAD',
    'Presión Arterial Media': 'PAM',
    'Saturación Oxígeno': 'Saturacion_O2',
    'Frecuencia Cardíaca': 'FC',
    'Frecuencia Respiratoria': 'FR',
    'Nitrógeno Ureico': 'BUN',
    'Antecedentes Cardíacos': 'Antecedentes_Cardiacos',
    'Antecedentes Diabéticos': 'Antecedentes_Diabeticos',
    'Antecedentes de Hipertensión Arterial': 'Antecedentes_HTA',
    'FIO2 > o igual a 50%': 'FIO2 > o igual a 50%',
    'Ventilación Mecánica': 'Ventilacion_Mecanica',
    'Cirugía Realizada': 'Cirugia',
    'Cirugía mismo día ingreso': 'Cirugia_mismo_dia',
    'Hemodinamia Realizada': 'Hemodinamia',
    'Hemodinamia mismo día ingreso ': 'Hemodinamia_mismo_dia',
    'Endoscopia mismo día ingreso': 'Endoscopia_mismo_dia',
    'Diálisis': 'Dialisis',
    'Trombólisis': 'Trombólisis',
    'Trombólisis mismo día ingreso': 'Trombólisis mismo día ingreso',
    'Troponinas Alteradas': 'Troponinas',
    'ECG Alterado': 'ECG_alterado',
    'RNM Protocolo Stroke': 'RNM_Stroke',
    'Compromiso Conciencia': 'Compromiso_Conciencia'
}

# Crear DataFrame de trabajo
df_work = df_con_validacion.copy()

# Renombrar columnas según mapeo
for col_form, col_base in mapeo_columnas.items():
    if col_form in df_work.columns:
        df_work[col_base] = df_work[col_form]

# Asegurar que existen las columnas que el modelo espera
for feature in metadata['features']:
    if feature not in df_work.columns:
        # Si es feature derivada, la calcularemos
        if feature in metadata.get('feature_engineering', []):
            continue
        # Si no existe, crear con NaN
        df_work[feature] = np.nan

# ============================================================
# 4. FEATURE ENGINEERING (IGUAL QUE EN ENTRENAMIENTO)
# ============================================================
print("\n⚙️  Feature Engineering...")

# Normalizar nombres para PAS, PAD, etc. si ya existen
if 'Presión Arterial Sistólica' in df_work.columns and 'PAS' not in df_work.columns:
    df_work['PAS'] = df_work['Presión Arterial Sistólica']
if 'Presión Arterial Diastólica' in df_work.columns and 'PAD' not in df_work.columns:
    df_work['PAD'] = df_work['Presión Arterial Diastólica']
if 'Saturación Oxígeno' in df_work.columns and 'Saturacion_O2' not in df_work.columns:
    df_work['Saturacion_O2'] = df_work['Saturación Oxígeno']
if 'Frecuencia Cardíaca' in df_work.columns and 'FC' not in df_work.columns:
    df_work['FC'] = df_work['Frecuencia Cardíaca']
if 'Frecuencia Respiratoria' in df_work.columns and 'FR' not in df_work.columns:
    df_work['FR'] = df_work['Frecuencia Respiratoria']

# Convertir columnas numéricas a numeric
numeric_cols = ['PAS', 'PAD', 'Saturacion_O2', 'FC', 'FR', 'Glasgow', 
                'PCR', 'Hemoglobina', 'Creatinina', 'BUN', 'Sodio', 'Potasio',
                'FIO2', 'Temperatura en °C']
for col in numeric_cols:
    if col in df_work.columns:
        df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

# Convertir binarias Si/No a 1/0
binary_cols = list(metadata['binary_features'])
for col in binary_cols:
    if col in df_work.columns and df_work[col].dtype == 'object':
        df_work[col] = df_work[col].map({'Si': 1, 'Sí': 1, 'si': 1, 'sí': 1, 'No': 0, 'no': 0, 'NO': 0})
        df_work[col] = pd.to_numeric(df_work[col], errors='coerce').fillna(0).astype(int)

# Crear features derivadas
df_work['Ratio_SatO2_FR'] = df_work['Saturacion_O2'] / (df_work['FR'] + 1)
df_work['Ratio_PAS_Glasgow'] = df_work['PAS'] / (df_work['Glasgow'] + 1)
df_work['Presion_Pulso'] = df_work['PAS'] - df_work['PAD']
df_work['Presion_Media_Calc'] = (df_work['PAS'] + 2 * df_work['PAD']) / 3

df_work['Flag_Hipotension'] = (df_work['PAS'] < 100).astype(int)
df_work['Flag_Hipertension_Critica'] = (df_work['PAS'] > 180).astype(int)
df_work['Flag_Hipoxemia'] = (df_work['Saturacion_O2'] < 92).astype(int)
df_work['Flag_Taquipnea'] = (df_work['FR'] > 24).astype(int)
df_work['Flag_Glasgow_Bajo'] = (df_work['Glasgow'] < 13).astype(int)

df_work['Score_Gravedad'] = (
    df_work['Flag_Hipotension'] * 2 +
    df_work['Flag_Hipertension_Critica'] * 1.5 +
    df_work['Flag_Hipoxemia'] * 2 +
    df_work['Flag_Taquipnea'] * 1.5 +
    df_work['Flag_Glasgow_Bajo'] * 2
)

df_work['SatO2_x_Glasgow'] = df_work['Saturacion_O2'] * df_work['Glasgow']
df_work['PA_x_FR'] = df_work['PAS'] * df_work['FR']

print(f"   ✓ Features derivadas creadas")

# ============================================================
# 5. PREPARAR MATRIZ X, y_true
# ============================================================
print("\n📊 Preparando datos para predicción...")

# Extraer features que el modelo espera
X = df_work[metadata['features']].copy()

# Etiquetas verdaderas
label_to_int = metadata['label_to_int']
int_to_label = metadata['int_to_label']

y_true = df_work['Validacion'].map(label_to_int).values

print(f"   ✓ Features preparadas: {len(metadata['features'])}")
print(f"   ✓ Casos a evaluar: {len(X)}")

# ============================================================
# 6. PREDICCIÓN
# ============================================================
print("\n🤖 Generando predicciones...")

y_pred = modelo.predict(X)
y_pred_proba = modelo.predict_proba(X)

# Confianza (probabilidad máxima)
confianza = y_pred_proba.max(axis=1)

print(f"   ✅ Predicciones completadas")
print(f"   Confianza promedio: {confianza.mean()*100:.1f}%")

# ============================================================
# 7. EVALUACIÓN
# ============================================================
print("\n" + "="*80)
print("📊 RESULTADOS DE EVALUACIÓN")
print("="*80)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

print(f"\n🎯 ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Cantidad de aciertos
aciertos = (y_true == y_pred).sum()
total = len(y_true)

print(f"\n   ✅ Aciertos: {aciertos}/{total}")
print(f"   ❌ Errores: {total - aciertos}/{total}")

# Classification Report
print(f"\n📋 CLASSIFICATION REPORT:")
print("="*80)
print(classification_report(y_true, y_pred, target_names=['NO PERTINENTE', 'PERTINENTE']))

# Confusion Matrix
print(f"\n🔢 CONFUSION MATRIX:")
print("="*80)
cm = confusion_matrix(y_true, y_pred)

print(f"\n                     Predicho")
print(f"                     NO PERT    PERTINENTE")
print(f"   Real NO PERT        {cm[0][0]:>6}     {cm[0][1]:>6}")
print(f"   Real PERTINENTE     {cm[1][0]:>6}     {cm[1][1]:>6}")

# Métricas por clase
print(f"\n📈 MÉTRICAS DETALLADAS:")
print("="*80)

# Para NO PERTINENTE
vn = cm[0][0]  # Verdaderos Negativos
fp = cm[0][1]  # Falsos Positivos
fn = cm[1][0]  # Falsos Negativos
vp = cm[1][1]  # Verdaderos Positivos

# NO PERTINENTE
if (vn + fp) > 0:
    precision_no_pert = vn / (vn + fn) if (vn + fn) > 0 else 0
    recall_no_pert = vn / (vn + fp) if (vn + fp) > 0 else 0
    print(f"\n   NO PERTINENTE:")
    print(f"      Casos reales: {vn + fp}")
    print(f"      Bien clasificados: {vn}")
    print(f"      Mal clasificados: {fp} (predichos como PERTINENTE)")
    print(f"      Precisión: {precision_no_pert*100:.1f}%")

# PERTINENTE
if (vp + fn) > 0:
    precision_pert = vp / (vp + fp) if (vp + fp) > 0 else 0
    recall_pert = vp / (vp + fn) if (vp + fn) > 0 else 0
    print(f"\n   PERTINENTE:")
    print(f"      Casos reales: {vp + fn}")
    print(f"      Bien clasificados: {vp}")
    print(f"      Mal clasificados: {fn} (predichos como NO PERTINENTE)")
    print(f"      Recall: {recall_pert*100:.1f}%")

# ============================================================
# 8. ANÁLISIS DE ERRORES
# ============================================================
print(f"\n" + "="*80)
print("🔍 ANÁLISIS DE ERRORES")
print("="*80)

# Casos mal clasificados
errores = df_work[y_true != y_pred].copy()
errores['Prediccion'] = [int_to_label[p] for p in y_pred[y_true != y_pred]]
errores['Real'] = errores['Validacion']
errores['Confianza'] = confianza[y_true != y_pred]

if len(errores) > 0:
    print(f"\n   Total errores: {len(errores)}")
    print(f"\n   Primeros 10 errores:")
    print(f"   {'Episodio':<15} {'Real':<15} {'Predicho':<15} {'Confianza':<10}")
    print("   " + "-"*60)
    
    for i, row in errores.head(10).iterrows():
        episodio = str(row.get('Episodio', 'N/A'))[:13]
        real = row['Real']
        pred = row['Prediccion']
        conf = row['Confianza']
        print(f"   {episodio:<15} {real:<15} {pred:<15} {conf*100:>6.1f}%")
    
    # Guardar errores en CSV
    errores[['Episodio', 'Real', 'Prediccion', 'Confianza']].to_csv(
        'errores_form_mpp.csv', index=False
    )
    print(f"\n   💾 Errores guardados en: errores_form_mpp.csv")
else:
    print(f"\n   ✅ ¡Sin errores! Predicción perfecta")

# ============================================================
# 9. GUARDAR PREDICCIONES COMPLETAS
# ============================================================
print(f"\n💾 Guardando predicciones completas...")

df_resultados = df.copy()
df_resultados['Prediccion_IA'] = np.nan
df_resultados['Confianza_IA'] = np.nan

# Asignar predicciones solo a los casos evaluados
indices_evaluados = df_con_validacion.index
df_resultados.loc[indices_evaluados, 'Prediccion_IA'] = [int_to_label[p] for p in y_pred]
df_resultados.loc[indices_evaluados, 'Confianza_IA'] = confianza * 100

df_resultados.to_excel('form_MPP_con_predicciones.xlsx', index=False)

print(f"   ✅ Predicciones guardadas: form_MPP_con_predicciones.xlsx")

# ============================================================
# 10. RESUMEN FINAL
# ============================================================
print("\n" + "="*80)
print("✅ EVALUACIÓN COMPLETADA")
print("="*80)

print(f"""
RESUMEN:
   • Total casos en form_MPP.xlsx: {len(df):,}
   • Casos CON validación (evaluados): {len(df_con_validacion):,}
   • Casos SIN validación: {len(df) - len(df_con_validacion):,}
   
RESULTADOS:
   • Accuracy: {accuracy*100:.2f}%
   • Aciertos: {aciertos}/{total}
   • Errores: {total - aciertos}/{total}
   
ARCHIVOS GENERADOS:
   • form_MPP_con_predicciones.xlsx (todas las predicciones)
   • errores_form_mpp.csv (análisis de errores)
""")

print("="*80)
