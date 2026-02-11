"""
Analiza con qué datos se entrenó el modelo actual
"""
import pandas as pd
import numpy as np

print("="*80)
print("DATOS DE ENTRENAMIENTO DEL MODELO ACTUAL")
print("="*80)

# 1. Cargar MPP
print("\n📂 ARCHIVO 1: Base MPP mes Octubre 2025.xlsx")
mpp = pd.read_excel('Base MPP mes Octubre 2025.xlsx')
print(f"   Total registros: {len(mpp):,}")

# Filtrar validados
validados = mpp[mpp['VALIDACIÓN'].notna()].copy()
print(f"   Casos validados: {len(validados):,}")

# Convertir fecha
mpp['Fecha Alta'] = pd.to_datetime(mpp['Fecha Alta'], errors='coerce')
validados['Fecha Alta'] = pd.to_datetime(validados['Fecha Alta'], errors='coerce')

# Separar octubre del resto
octubre = validados[
    (validados['Fecha Alta'].dt.year == 2025) & 
    (validados['Fecha Alta'].dt.month == 10)
].copy()

resto = validados[
    ~((validados['Fecha Alta'].dt.year == 2025) & 
      (validados['Fecha Alta'].dt.month == 10))
].copy()

print(f"\n   OCTUBRE 2025 (usado para VALIDACIÓN):")
print(f"      • {len(octubre):,} casos")
if len(octubre) > 0:
    print(f"      • Distribución:")
    for val, count in octubre['VALIDACIÓN'].value_counts().items():
        print(f"         - {val}: {count}")

print(f"\n   RESTO DE MESES (usado para ENTRENAMIENTO):")
print(f"      • {len(resto):,} casos")
if len(resto) > 0:
    print(f"      • Distribución:")
    for val, count in resto['VALIDACIÓN'].value_counts().items():
        print(f"         - {val}: {count}")
    
    # Analizar meses
    resto['Mes'] = resto['Fecha Alta'].dt.to_period('M')
    print(f"\n      • Meses incluidos:")
    for mes, count in resto['Mes'].value_counts().sort_index().items():
        print(f"         - {mes}: {count} casos")

# 2. Cargar query
print(f"\n📂 ARCHIVO 2: query.xlsx")
query = pd.read_excel('query.xlsx')
print(f"   Total registros: {len(query):,}")
print(f"   Columnas: {len(query.columns)}")

# Verificar cruce
query['EPISODIO_LIMPIO'] = query['Episodio'].astype(str).str.strip().str.upper()
resto['EPISODIO_LIMPIO'] = resto['ESTADIA/EPISODIO'].astype(str).str.strip().str.upper()
octubre['EPISODIO_LIMPIO'] = octubre['ESTADIA/EPISODIO'].astype(str).str.strip().str.upper()

# Cruzar
train_cruzado = query.merge(
    resto[['EPISODIO_LIMPIO', 'VALIDACIÓN']],
    on='EPISODIO_LIMPIO',
    how='inner'
)

val_cruzado = query.merge(
    octubre[['EPISODIO_LIMPIO', 'VALIDACIÓN']],
    on='EPISODIO_LIMPIO',
    how='inner'
)

print(f"\n🔗 CRUCE MPP + QUERY:")
print(f"\n   SET DE ENTRENAMIENTO:")
print(f"      • {len(train_cruzado):,} casos (episodios cruzados exitosamente)")
print(f"      • Distribución:")
for val, count in train_cruzado['VALIDACIÓN'].value_counts().items():
    pct = count / len(train_cruzado) * 100
    print(f"         - {val}: {count} ({pct:.1f}%)")

print(f"\n   SET DE VALIDACIÓN:")
print(f"      • {len(val_cruzado):,} casos (octubre 2025)")
print(f"      • Distribución:")
for val, count in val_cruzado['VALIDACIÓN'].value_counts().items():
    pct = count / len(val_cruzado) * 100
    print(f"         - {val}: {count} ({pct:.1f}%)")

print(f"\n📊 VARIABLES CLÍNICAS DISPONIBLES EN QUERY:")
clinical_vars = [
    'Presión Arterial Sistólica', 'Presión Arterial Diastólica',
    'Temperatura en °C', 'Saturación Oxígeno', 'Frecuencia Cardíaca',
    'Frecuencia Respiratoria', 'Glasgow', 'PCR', 'Hemoglobina',
    'Creatinina', 'Nitrógeno Ureico', 'Sodio', 'Potasio', 'FIO2'
]

procedimientos = [
    'Ventilación Mecánica', 'Cirugía Realizada', 'Hemodinamia Realizada',
    'Endoscopia', 'Diálisis', 'Trombólisis', 'DVA', 'Transfusiones',
    'Troponinas Alteradas', 'ECG Alterado', 'RNM Protocolo Stroke',
    'Compromiso Conciencia'
]

antecedentes = [
    'Antecedentes Cardíacos', 'Antecedentes Diabéticos',
    'Antecedentes de Hipertensión Arterial'
]

print(f"\n   Signos vitales y laboratorio ({len(clinical_vars)}):")
for var in clinical_vars:
    if var in query.columns:
        no_nulos = query[var].notna().sum()
        pct = no_nulos / len(query) * 100
        print(f"      ✓ {var}: {pct:.1f}% completo")

print(f"\n   Procedimientos ({len(procedimientos)}):")
for var in procedimientos:
    if var in query.columns:
        no_nulos = query[var].notna().sum()
        pct = no_nulos / len(query) * 100
        print(f"      ✓ {var}: {pct:.1f}% completo")

print(f"\n   Antecedentes ({len(antecedentes)}):")
for var in antecedentes:
    if var in query.columns:
        no_nulos = query[var].notna().sum()
        pct = no_nulos / len(query) * 100
        print(f"      ✓ {var}: {pct:.1f}% completo")

print(f"\n" + "="*80)
print("RESUMEN")
print("="*80)
print(f"""
El modelo actual fue entrenado con:

1. DATOS FUENTE:
   • Base MPP mes Octubre 2025.xlsx (validaciones manuales)
   • query.xlsx (variables clínicas completas)

2. ESTRATEGIA:
   • Se cruzan los episodios entre MPP y query
   • MPP aporta las etiquetas (VALIDACIÓN: PERTINENTE/NO PERTINENTE)
   • Query aporta las variables clínicas (signos vitales, procedimientos, etc.)

3. DIVISIÓN:
   • ENTRENAMIENTO: {len(train_cruzado)} casos (meses anteriores a octubre 2025)
   • VALIDACIÓN: {len(val_cruzado)} casos (octubre 2025)

4. FEATURE ENGINEERING:
   • 70 features totales
   • Incluye ratios derivados (SatO2/FR, PAS/Glasgow)
   • Flags de riesgo (hipotensión, hipoxemia, Glasgow bajo)
   • Scores de gravedad compuestos

5. MODELO:
   • Ensemble de 4 algoritmos (XGBoost x2, Random Forest, Gradient Boosting)
   • Threshold optimizado: 0.60
""")
print("="*80)
