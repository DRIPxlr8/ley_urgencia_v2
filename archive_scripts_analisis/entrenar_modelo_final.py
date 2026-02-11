"""
MODELO FINAL - Ley de Urgencia
Entrenamiento con data/Base.xlsx
Basado en el mejor modelo anterior (ensemble robusto)
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 MODELO FINAL - LEY DE URGENCIA")
print("="*80)

# ============================================================
# 1. CARGAR DATOS
# ============================================================
print("\n📂 Cargando datos...")
df = pd.read_excel('data/Base.xlsx')

print(f"   Total registros: {len(df):,}")
print(f"   Columnas: {len(df.columns)}")

# Verificar validaciones
if 'Resolucion' not in df.columns:
    print("\n❌ ERROR: No se encontró columna 'Resolucion'")
    exit(1)

print(f"\n   Distribución de clases:")
print(df['Resolucion'].value_counts())

# ============================================================
# 2. PREPARAR FEATURES
# ============================================================
print("\n🔧 Preparando features...")

# Mapeo de etiquetas
label_to_int = {'NO PERTINENTE': 0, 'PERTINENTE': 1}
int_to_label = {0: 'NO PERTINENTE', 1: 'PERTINENTE'}

df['Target'] = df['Resolucion'].map(label_to_int)

# Variables numéricas base
numeric_base = [
    'PAS', 'PAD', 'Temperatura en °C', 'Saturacion_O2', 'FC', 'FR', 'Glasgow',
    'PCR', 'Hemoglobina', 'Creatinina', 'BUN', 'Sodio', 'Potasio', 'FIO2'
]

# Variables binarias
binary_base = [
    'FIO2 > o igual a 50%', 'Ventilacion_Mecanica', 'Cirugia', 'Cirugia_mismo_dia',
    'Hemodinamia', 'Hemodinamia_mismo_dia', 'Endoscopia', 'Endoscopia_mismo_dia',
    'Dialisis', 'Trombólisis', 'Trombólisis mismo día ingreso', 'DVA', 'Transfusiones',
    'Troponinas', 'ECG_alterado', 'RNM_Stroke', 'Compromiso_Conciencia',
    'Antecedentes_Cardiacos', 'Antecedentes_Diabeticos', 'Antecedentes_HTA'
]

# Variables categóricas
categorical_base = ['Tipo_Cama']

# Verificar y ajustar columnas disponibles
present_numeric = [c for c in numeric_base if c in df.columns]
present_binary = [c for c in binary_base if c in df.columns]
present_categorical = [c for c in categorical_base if c in df.columns]

# CONVERTIR BINARIAS DE SI/NO A 1/0
print("\n🔄 Convirtiendo variables binarias...")
for col in present_binary:
    if df[col].dtype == 'object':
        df[col] = df[col].map({'Si': 1, 'Sí': 1, 'si': 1, 'sí': 1, 'No': 0, 'no': 0, 'NO': 0})
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

print(f"   Numéricas: {len(present_numeric)}")
print(f"   Binarias: {len(present_binary)}")
print(f"   Categóricas: {len(present_categorical)}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n⚙️  Feature Engineering...")

# Normalizar nombres para trabajar uniformemente
df_work = df.copy()

# Crear features derivadas
# Ratios críticos
df_work['Ratio_SatO2_FR'] = df_work['Saturacion_O2'] / (df_work['FR'] + 1)
df_work['Ratio_PAS_Glasgow'] = df_work['PAS'] / (df_work['Glasgow'] + 1)
df_work['Presion_Pulso'] = df_work['PAS'] - df_work['PAD']
df_work['Presion_Media_Calc'] = (df_work['PAS'] + 2 * df_work['PAD']) / 3

# Flags de riesgo
df_work['Flag_Hipotension'] = (df_work['PAS'] < 100).astype(int)
df_work['Flag_Hipertension_Critica'] = (df_work['PAS'] > 180).astype(int)
df_work['Flag_Hipoxemia'] = (df_work['Saturacion_O2'] < 92).astype(int)
df_work['Flag_Taquipnea'] = (df_work['FR'] > 24).astype(int)
df_work['Flag_Glasgow_Bajo'] = (df_work['Glasgow'] < 13).astype(int)

# Score de gravedad compuesto
df_work['Score_Gravedad'] = (
    df_work['Flag_Hipotension'] * 2 +
    df_work['Flag_Hipertension_Critica'] * 1.5 +
    df_work['Flag_Hipoxemia'] * 2 +
    df_work['Flag_Taquipnea'] * 1.5 +
    df_work['Flag_Glasgow_Bajo'] * 2
)

# Interacciones
df_work['SatO2_x_Glasgow'] = df_work['Saturacion_O2'] * df_work['Glasgow']
df_work['PA_x_FR'] = df_work['PAS'] * df_work['FR']

# Agregar features derivadas a las listas
feature_engineering = [
    'Ratio_SatO2_FR', 'Ratio_PAS_Glasgow', 'Presion_Pulso', 'Presion_Media_Calc',
    'Flag_Hipotension', 'Flag_Hipertension_Critica', 'Flag_Hipoxemia',
    'Flag_Taquipnea', 'Flag_Glasgow_Bajo', 'Score_Gravedad',
    'SatO2_x_Glasgow', 'PA_x_FR'
]

print(f"   Features derivadas creadas: {len(feature_engineering)}")

# ============================================================
# 4. PREPARAR MATRIZ X, y
# ============================================================
print("\n📊 Preparando conjunto de datos...")

# Todas las features
all_features = present_numeric + present_binary + present_categorical + feature_engineering

X = df_work[all_features].copy()
y = df_work['Target'].values

print(f"   Features totales: {len(all_features)}")
print(f"   Casos totales: {len(X)}")

# ============================================================
# 5. SPLIT TRAIN/TEST
# ============================================================
print("\n✂️  Dividiendo datos...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {len(X_train):,} casos")
print(f"   Test: {len(X_test):,} casos")

print(f"\n   Distribución Train:")
unique, counts = np.unique(y_train, return_counts=True)
for val, count in zip(unique, counts):
    label = int_to_label[val]
    print(f"      {label}: {count} ({count/len(y_train)*100:.1f}%)")

print(f"\n   Distribución Test:")
unique, counts = np.unique(y_test, return_counts=True)
for val, count in zip(unique, counts):
    label = int_to_label[val]
    print(f"      {label}: {count} ({count/len(y_test)*100:.1f}%)")

# ============================================================
# 6. PREPROCESAMIENTO
# ============================================================
print("\n🔨 Configurando preprocesamiento...")

# Features numéricas (incluye derivadas numéricas)
numeric_features = present_numeric + [f for f in feature_engineering if f not in [
    'Flag_Hipotension', 'Flag_Hipertension_Critica', 'Flag_Hipoxemia',
    'Flag_Taquipnea', 'Flag_Glasgow_Bajo'
]]

# Features binarias (incluye flags derivados)
binary_features = present_binary + [
    'Flag_Hipotension', 'Flag_Hipertension_Critica', 'Flag_Hipoxemia',
    'Flag_Taquipnea', 'Flag_Glasgow_Bajo'
]

# Pipeline para numéricas: KNNImputer + StandardScaler
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# Pipeline para binarias: SimpleImputer (rellena con 0)
binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

# Pipeline para categóricas: SimpleImputer + OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('bin', binary_transformer, binary_features),
        ('cat', categorical_transformer, present_categorical)
    ],
    remainder='drop'
)

print(f"   ✓ Numéricas: {len(numeric_features)} features")
print(f"   ✓ Binarias: {len(binary_features)} features")
print(f"   ✓ Categóricas: {len(present_categorical)} features")

# ============================================================
# 7. MODELO ENSEMBLE (VOTING CLASSIFIER)
# ============================================================
print("\n🤖 Configurando modelo Ensemble...")

# Configurar estimadores base
estimators = [
    ('xgb1', xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    )),
    ('xgb2', xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=123,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    )),
    ('rf', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight='balanced'
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    ))
]

# VotingClassifier con soft voting
ensemble = VotingClassifier(
    estimators=estimators,
    voting='soft',
    n_jobs=-1
)

print(f"   ✓ {len(estimators)} modelos base configurados")
print(f"   ✓ Voting: soft (promedio de probabilidades)")
print(f"   ✓ Balance de clases: aplicado en XGB y RF")

# ============================================================
# 8. PIPELINE COMPLETO
# ============================================================
pipeline_final = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', ensemble)
])

# ============================================================
# 9. ENTRENAMIENTO
# ============================================================
print("\n🚀 Entrenando modelo...")
print("   (Esto puede tomar varios minutos...)")

pipeline_final.fit(X_train, y_train)

print("   ✅ Entrenamiento completado!")

# ============================================================
# 10. EVALUACIÓN
# ============================================================
print("\n📈 Evaluando modelo...")

# Predicciones
y_pred_train = pipeline_final.predict(X_train)
y_pred_test = pipeline_final.predict(X_test)

# Accuracy
acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"\n   ACCURACY:")
print(f"      Train: {acc_train:.4f} ({acc_train*100:.2f}%)")
print(f"      Test:  {acc_test:.4f} ({acc_test*100:.2f}%)")

# Classification report
print(f"\n   CLASSIFICATION REPORT (Test):")
print(classification_report(y_test, y_pred_test, target_names=['NO PERTINENTE', 'PERTINENTE']))

# Confusion matrix
print(f"\n   CONFUSION MATRIX (Test):")
cm = confusion_matrix(y_test, y_pred_test)
print(f"\n                  Predicho")
print(f"                  NO PERT    PERTINENTE")
print(f"      Real NO PERT    {cm[0][0]:>6}     {cm[0][1]:>6}")
print(f"      Real PERTINENTE {cm[1][0]:>6}     {cm[1][1]:>6}")

# ============================================================
# 11. GUARDAR MODELO
# ============================================================
print("\n💾 Guardando modelo...")

# Guardar pipeline
joblib.dump(pipeline_final, 'modelo_ley_urgencia.pkl')

# Guardar metadata
metadata = {
    'features': all_features,
    'numeric_features': numeric_features,
    'binary_features': binary_features,
    'categorical_features': present_categorical,
    'label_to_int': label_to_int,
    'int_to_label': int_to_label,
    'classes': ['NO PERTINENTE', 'PERTINENTE'],
    'n_train': len(X_train),
    'n_test': len(X_test),
    'accuracy_train': acc_train,
    'accuracy_test': acc_test,
    'model_type': 'ensemble_final_base',
    'feature_engineering': feature_engineering
}

joblib.dump(metadata, 'modelo_metadata.pkl')

print(f"   ✅ Modelo guardado: modelo_ley_urgencia.pkl")
print(f"   ✅ Metadata guardada: modelo_metadata.pkl")

print("\n" + "="*80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*80)
print(f"""
RESUMEN:
   • Datos de entrenamiento: {len(X_train):,} casos
   • Datos de test: {len(X_test):,} casos
   • Features utilizadas: {len(all_features)}
   • Accuracy en test: {acc_test*100:.2f}%
   • Modelo: Ensemble de 4 algoritmos
   
SIGUIENTE PASO:
   Ejecutar: python evaluar_form_mpp.py
   Para evaluar el modelo con form_MPP.xlsx
""")
print("="*80)
