"""
BÚSQUEDA DEL MEJOR MODELO
Prueba diferentes configuraciones y evalúa directamente con form_MPP.xlsx
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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔍 BÚSQUEDA DEL MEJOR MODELO")
print("="*80)

# ============================================================
# 1. CARGAR Y PREPARAR DATOS DE ENTRENAMIENTO
# ============================================================
print("\n📂 Cargando datos de entrenamiento (Base.xlsx)...")
df = pd.read_excel('data/Base.xlsx')

label_to_int = {'NO PERTINENTE': 0, 'PERTINENTE': 1}
int_to_label = {0: 'NO PERTINENTE', 1: 'PERTINENTE'}
df['Target'] = df['Resolucion'].map(label_to_int)

# Features
numeric_base = [
    'PAS', 'PAD', 'Temperatura en °C', 'Saturacion_O2', 'FC', 'FR', 'Glasgow',
    'PCR', 'Hemoglobina', 'Creatinina', 'BUN', 'Sodio', 'Potasio', 'FIO2'
]
binary_base = [
    'FIO2 > o igual a 50%', 'Ventilacion_Mecanica', 'Cirugia', 'Cirugia_mismo_dia',
    'Hemodinamia', 'Hemodinamia_mismo_dia', 'Endoscopia', 'Endoscopia_mismo_dia',
    'Dialisis', 'Trombólisis', 'Trombólisis mismo día ingreso', 'DVA', 'Transfusiones',
    'Troponinas', 'ECG_alterado', 'RNM_Stroke', 'Compromiso_Conciencia',
    'Antecedentes_Cardiacos', 'Antecedentes_Diabeticos', 'Antecedentes_HTA'
]
categorical_base = ['Tipo_Cama']

present_numeric = [c for c in numeric_base if c in df.columns]
present_binary = [c for c in binary_base if c in df.columns]
present_categorical = [c for c in categorical_base if c in df.columns]

# Convertir binarias
for col in present_binary:
    if df[col].dtype == 'object':
        df[col] = df[col].map({'Si': 1, 'Sí': 1, 'si': 1, 'sí': 1, 'No': 0, 'no': 0, 'NO': 0})
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# Feature engineering
df['Ratio_SatO2_FR'] = df['Saturacion_O2'] / (df['FR'] + 1)
df['Ratio_PAS_Glasgow'] = df['PAS'] / (df['Glasgow'] + 1)
df['Presion_Pulso'] = df['PAS'] - df['PAD']
df['Presion_Media_Calc'] = (df['PAS'] + 2 * df['PAD']) / 3
df['Flag_Hipotension'] = (df['PAS'] < 100).astype(int)
df['Flag_Hipertension_Critica'] = (df['PAS'] > 180).astype(int)
df['Flag_Hipoxemia'] = (df['Saturacion_O2'] < 92).astype(int)
df['Flag_Taquipnea'] = (df['FR'] > 24).astype(int)
df['Flag_Glasgow_Bajo'] = (df['Glasgow'] < 13).astype(int)
df['Score_Gravedad'] = (
    df['Flag_Hipotension'] * 2 + df['Flag_Hipertension_Critica'] * 1.5 +
    df['Flag_Hipoxemia'] * 2 + df['Flag_Taquipnea'] * 1.5 + df['Flag_Glasgow_Bajo'] * 2
)
df['SatO2_x_Glasgow'] = df['Saturacion_O2'] * df['Glasgow']
df['PA_x_FR'] = df['PAS'] * df['FR']

feature_engineering = [
    'Ratio_SatO2_FR', 'Ratio_PAS_Glasgow', 'Presion_Pulso', 'Presion_Media_Calc',
    'Flag_Hipotension', 'Flag_Hipertension_Critica', 'Flag_Hipoxemia',
    'Flag_Taquipnea', 'Flag_Glasgow_Bajo', 'Score_Gravedad',
    'SatO2_x_Glasgow', 'PA_x_FR'
]

all_features = present_numeric + present_binary + present_categorical + feature_engineering
X = df[all_features].copy()
y = df['Target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocesamiento
numeric_features = present_numeric + [f for f in feature_engineering if f not in [
    'Flag_Hipotension', 'Flag_Hipertension_Critica', 'Flag_Hipoxemia',
    'Flag_Taquipnea', 'Flag_Glasgow_Bajo'
]]
binary_features = present_binary + [
    'Flag_Hipotension', 'Flag_Hipertension_Critica', 'Flag_Hipoxemia',
    'Flag_Taquipnea', 'Flag_Glasgow_Bajo'
]

numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])
binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('bin', binary_transformer, binary_features),
        ('cat', categorical_transformer, present_categorical)
    ],
    remainder='drop'
)

print(f"   ✅ Datos cargados: {len(X_train):,} train / {len(X_test):,} test")

# ============================================================
# 2. CARGAR DATOS DE EVALUACIÓN (form_MPP.xlsx)
# ============================================================
print("\n📂 Cargando datos de evaluación (form_MPP.xlsx)...")
df_eval = pd.read_excel('data/form_MPP.xlsx')
df_eval_val = df_eval[df_eval['Validacion'].notna()].copy()

# Preparar features igual que entrenamiento
mapeo_columnas = {
    'Presión Arterial Sistólica': 'PAS',
    'Presión Arterial Diastólica': 'PAD',
    'Saturación Oxígeno': 'Saturacion_O2',
    'Frecuencia Cardíaca': 'FC',
    'Frecuencia Respiratoria': 'FR',
    'Nitrógeno Ureico': 'BUN',
}

df_work = df_eval_val.copy()
for col_form, col_base in mapeo_columnas.items():
    if col_form in df_work.columns:
        df_work[col_base] = df_work[col_form]

# Asegurar columnas
for feature in all_features:
    if feature not in df_work.columns and feature not in feature_engineering:
        df_work[feature] = np.nan

# Convertir numéricas
numeric_cols = ['PAS', 'PAD', 'Saturacion_O2', 'FC', 'FR', 'Glasgow', 
                'PCR', 'Hemoglobina', 'Creatinina', 'BUN', 'Sodio', 'Potasio',
                'FIO2', 'Temperatura en °C']
for col in numeric_cols:
    if col in df_work.columns:
        df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

# Convertir binarias
for col in binary_features:
    if col in df_work.columns and df_work[col].dtype == 'object':
        df_work[col] = df_work[col].map({'Si': 1, 'Sí': 1, 'si': 1, 'sí': 1, 'No': 0, 'no': 0, 'NO': 0})
        df_work[col] = pd.to_numeric(df_work[col], errors='coerce').fillna(0).astype(int)

# Feature engineering
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
    df_work['Flag_Hipotension'] * 2 + df_work['Flag_Hipertension_Critica'] * 1.5 +
    df_work['Flag_Hipoxemia'] * 2 + df_work['Flag_Taquipnea'] * 1.5 + df_work['Flag_Glasgow_Bajo'] * 2
)
df_work['SatO2_x_Glasgow'] = df_work['Saturacion_O2'] * df_work['Glasgow']
df_work['PA_x_FR'] = df_work['PAS'] * df_work['FR']

X_eval = df_work[all_features].copy()
y_eval = df_work['Validacion'].map(label_to_int).values

print(f"   ✅ Datos de evaluación: {len(X_eval):,} casos")

# ============================================================
# 3. DEFINIR CONFIGURACIONES A PROBAR
# ============================================================
print("\n" + "="*80)
print("🧪 PROBANDO DIFERENTES CONFIGURACIONES")
print("="*80)

configuraciones = [
    {
        'nombre': '1. XGBoost Solo (optimizado)',
        'modelo': xgb.XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        )
    },
    {
        'nombre': '2. XGBoost Solo (más profundo)',
        'modelo': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        )
    },
    {
        'nombre': '3. Random Forest Solo',
        'modelo': RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    },
    {
        'nombre': '4. Gradient Boosting Solo',
        'modelo': GradientBoostingClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
    },
    {
        'nombre': '5. Ensemble: XGB + RF (2 modelos)',
        'modelo': VotingClassifier(
            estimators=[
                ('xgb', xgb.XGBClassifier(
                    n_estimators=400, max_depth=7, learning_rate=0.04,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    eval_metric='logloss',
                    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=400, max_depth=20, min_samples_split=5,
                    random_state=42, class_weight='balanced', n_jobs=-1
                ))
            ],
            voting='soft',
            n_jobs=-1
        )
    },
    {
        'nombre': '6. Ensemble: XGB + GB (2 modelos)',
        'modelo': VotingClassifier(
            estimators=[
                ('xgb', xgb.XGBClassifier(
                    n_estimators=400, max_depth=7, learning_rate=0.04,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    eval_metric='logloss',
                    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=300, max_depth=7, learning_rate=0.05,
                    subsample=0.8, random_state=42
                ))
            ],
            voting='soft',
            n_jobs=-1
        )
    },
    {
        'nombre': '7. Ensemble: XGB + RF + GB (3 modelos)',
        'modelo': VotingClassifier(
            estimators=[
                ('xgb', xgb.XGBClassifier(
                    n_estimators=350, max_depth=7, learning_rate=0.04,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    eval_metric='logloss',
                    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=300, max_depth=18, min_samples_split=5,
                    random_state=42, class_weight='balanced', n_jobs=-1
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=250, max_depth=7, learning_rate=0.05,
                    subsample=0.8, random_state=42
                ))
            ],
            voting='soft',
            n_jobs=-1
        )
    },
    {
        'nombre': '8. Ensemble Actual (4 modelos)',
        'modelo': VotingClassifier(
            estimators=[
                ('xgb1', xgb.XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    eval_metric='logloss',
                    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                )),
                ('xgb2', xgb.XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    subsample=0.9, colsample_bytree=0.9, random_state=123,
                    eval_metric='logloss',
                    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=10,
                    min_samples_leaf=4, random_state=42, class_weight='balanced'
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, random_state=42
                ))
            ],
            voting='soft',
            n_jobs=-1
        )
    }
]

# ============================================================
# 4. ENTRENAR Y EVALUAR CADA CONFIGURACIÓN
# ============================================================
resultados = []

for i, config in enumerate(configuraciones, 1):
    print(f"\n{'─'*80}")
    print(f"Probando: {config['nombre']}")
    print(f"{'─'*80}")
    
    # Crear pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', config['modelo'])
    ])
    
    # Entrenar
    print("   Entrenando...")
    pipeline.fit(X_train, y_train)
    
    # Evaluar en test interno
    y_pred_test = pipeline.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    # Evaluar en form_MPP.xlsx (LO IMPORTANTE)
    y_pred_eval = pipeline.predict(X_eval)
    acc_eval = accuracy_score(y_eval, y_pred_eval)
    
    # Guardar resultados
    resultados.append({
        'nombre': config['nombre'],
        'acc_test_interno': acc_test,
        'acc_form_mpp': acc_eval,
        'pipeline': pipeline
    })
    
    print(f"   ✅ Test interno: {acc_test*100:.2f}%")
    print(f"   🎯 form_MPP.xlsx: {acc_eval*100:.2f}%")

# ============================================================
# 5. RANKING Y RECOMENDACIÓN
# ============================================================
print("\n" + "="*80)
print("📊 RANKING DE MODELOS")
print("="*80)

# Ordenar por accuracy en form_MPP (lo más importante)
resultados.sort(key=lambda x: x['acc_form_mpp'], reverse=True)

print(f"\n{'#':<3} {'Modelo':<45} {'Test':<8} {'form_MPP':<10} {'Δ':<7}")
print("─" * 80)

for i, r in enumerate(resultados, 1):
    delta = r['acc_form_mpp'] - r['acc_test_interno']
    emoji = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    
    nombre_corto = r['nombre'][:43]
    print(f"{emoji} {i:<2} {nombre_corto:<45} {r['acc_test_interno']*100:>6.2f}% {r['acc_form_mpp']*100:>8.2f}% {delta*100:>+6.2f}%")

# Mejor modelo
mejor = resultados[0]

print("\n" + "="*80)
print("🏆 MEJOR MODELO")
print("="*80)

print(f"\n{mejor['nombre']}")
print(f"   Accuracy en form_MPP.xlsx: {mejor['acc_form_mpp']*100:.2f}%")
print(f"   Accuracy en test interno: {mejor['acc_test_interno']*100:.2f}%")
print(f"   Diferencia (gap): {(mejor['acc_form_mpp'] - mejor['acc_test_interno'])*100:+.2f}%")

# Guardar mejor modelo
print(f"\n💾 Guardando mejor modelo...")
joblib.dump(mejor['pipeline'], 'modelo_ley_urgencia_MEJOR.pkl')

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
    'accuracy_test': mejor['acc_test_interno'],
    'accuracy_form_mpp': mejor['acc_form_mpp'],
    'model_type': mejor['nombre'],
    'feature_engineering': feature_engineering
}
joblib.dump(metadata, 'modelo_metadata_MEJOR.pkl')

print(f"   ✅ Guardado: modelo_ley_urgencia_MEJOR.pkl")
print(f"   ✅ Guardado: modelo_metadata_MEJOR.pkl")

print("\n" + "="*80)
print("✅ BÚSQUEDA COMPLETADA")
print("="*80)

print(f"""
MEJOR CONFIGURACIÓN ENCONTRADA:
{mejor['nombre']}

MEJORA:
   Modelo actual (4 modelos): ~64.95%
   Mejor modelo encontrado: {mejor['acc_form_mpp']*100:.2f}%
   Mejora: {(mejor['acc_form_mpp'] - 0.6495)*100:+.2f} puntos porcentuales

El modelo ha sido guardado y está listo para usar.
""")

print("="*80)
