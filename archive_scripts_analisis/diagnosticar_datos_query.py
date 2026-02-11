"""
Diagnostica diferencias entre datos de query y datos de entrenamiento
"""
import pandas as pd
import numpy as np

print("="*80)
print("DIAGNÓSTICO DE DATOS QUERY vs ENTRENAMIENTO")
print("="*80)

# Cargar datos
print("\n📂 Cargando archivos...")
query = pd.read_excel('query_octubre_2025_con_validacion.xlsx')
entrenamiento = pd.read_excel('Dataset_Combinado_Entrenamiento.xlsx')

print(f"   Query: {len(query)} casos")
print(f"   Entrenamiento: {len(entrenamiento)} casos")

# Mapeo de columnas
mapeo = {
    'PA_Sistolica': 'Presión Arterial Sistólica',
    'PA_Diastolica': 'Presión Arterial Diastólica',
    'Temperatura': 'Temperatura en °C',
    'SatO2': 'Saturación Oxígeno',
    'FC': 'Frecuencia Cardíaca',
    'FR': 'Frecuencia Respiratoria',
    'Glasgow': 'Glasgow',
    'PCR': 'PCR',
    'Hemoglobina': 'Hemoglobina',
    'Creatinina': 'Creatinina',
    'BUN': 'Nitrógeno Ureico',
    'Sodio': 'Sodio',
    'Potasio': 'Potasio',
    'FiO2': 'FIO2',
    'FiO2_ge50_flag': 'FIO2 > o igual a 50%',
    'Ventilacion_Mecanica': 'Ventilación Mecánica',
    'Cirugia': 'Cirugía Realizada',
    'Cirugia_mismo_dia': 'Cirugía mismo día ingreso',
    'Hemodinamia': 'Hemodinamia Realizada',
    'Hemodinamia_mismo_dia': 'Hemodinamia mismo día ingreso ',
    'Endoscopia': 'Endoscopia',
    'Endoscopia_mismo_dia': 'Endoscopia mismo día ingreso',
    'Dialisis': 'Diálisis',
    'Trombolisis': 'Trombólisis',
    'Trombolisis_mismo_dia': 'Trombólisis mismo día ingreso',
    'DVA': 'DVA',
    'Transfusiones': 'Transfusiones',
    'Troponinas_Alteradas': 'Troponinas Alteradas',
    'ECG_Alterado': 'ECG Alterado',
    'RNM_Stroke': 'RNM Protocolo Stroke',
    'Compromiso_Conciencia': 'Compromiso Conciencia',
    'Antecedente_Cardiaco': 'Antecedentes Cardíacos',
    'Antecedente_Diabetico': 'Antecedentes Diabéticos',
    'Antecedente_HTA': 'Antecedentes de Hipertensión Arterial',
    'Tipo_Cama': 'Tipo de Cama',
}

print("\n📊 ANÁLISIS DE COMPLETITUD DE DATOS")
print("="*80)

# Analizar datos faltantes
print("\nDatos faltantes en QUERY:")
for col_modelo, col_query in mapeo.items():
    if col_query in query.columns:
        missing_pct = query[col_query].isna().sum() / len(query) * 100
        print(f"  {col_modelo:25s}: {missing_pct:5.1f}% faltante")
    else:
        print(f"  {col_modelo:25s}: COLUMNA NO EXISTE")

print("\nDatos faltantes en ENTRENAMIENTO:")
for col_modelo, col_query in mapeo.items():
    if col_modelo in entrenamiento.columns:
        missing_pct = entrenamiento[col_modelo].isna().sum() / len(entrenamiento) * 100
        print(f"  {col_modelo:25s}: {missing_pct:5.1f}% faltante")

# Comparar valores únicos en columnas binarias
print("\n\n📋 ANÁLISIS DE VALORES EN COLUMNAS BINARIAS")
print("="*80)

binary_cols = {
    'FiO2_ge50_flag': 'FIO2 > o igual a 50%',
    'Ventilacion_Mecanica': 'Ventilación Mecánica',
    'Cirugia': 'Cirugía Realizada',
}

for col_modelo, col_query in binary_cols.items():
    if col_query in query.columns:
        valores = query[col_query].dropna().unique()
        print(f"\n{col_modelo} (query):")
        print(f"  Valores únicos: {valores}")
        print(f"  Conteo: {query[col_query].value_counts().to_dict()}")
    
    if col_modelo in entrenamiento.columns:
        valores = entrenamiento[col_modelo].dropna().unique()
        print(f"\n{col_modelo} (entrenamiento):")
        print(f"  Valores únicos: {valores}")
        print(f"  Distribución: {entrenamiento[col_modelo].value_counts().to_dict()}")

# Comparar distribución de casos PERTINENTE vs NO PERTINENTE
print("\n\n🎯 DISTRIBUCIÓN DE CLASES")
print("="*80)

print("\nQuery:")
print(query['VALIDACIÓN'].value_counts())
print(f"% PERTINENTE: {query['VALIDACIÓN'].value_counts()['PERTINENTE'] / len(query) * 100:.1f}%")

print("\nEntrenamiento:")
print(entrenamiento['Resolucion'].value_counts())
print(f"% PERTINENTE: {entrenamiento['Resolucion'].value_counts()['PERTINENTE'] / len(entrenamiento) * 100:.1f}%")

# Analizar casos PERTINENTES en query - ver qué características tienen
print("\n\n🔍 CARACTERÍSTICAS DE CASOS PERTINENTES EN QUERY")
print("="*80)

pertinentes = query[query['VALIDACIÓN'] == 'PERTINENTE']
print(f"\nTotal PERTINENTES: {len(pertinentes)}")

# Ver columnas con más datos completos en pertinentes
print("\nDatos completos en PERTINENTES:")
for col_modelo, col_query in mapeo.items():
    if col_query in query.columns:
        missing_pct = pertinentes[col_query].isna().sum() / len(pertinentes) * 100
        if missing_pct < 50:  # Solo mostrar las que tienen menos de 50% faltante
            print(f"  {col_modelo:25s}: {100-missing_pct:5.1f}% completo")

# Ver valores numéricos promedio en pertinentes vs no pertinentes
print("\n\n📈 PROMEDIOS NUMÉRICOS: PERTINENTE vs NO PERTINENTE (query)")
print("="*80)

numeric_cols = ['Presión Arterial Sistólica', 'Presión Arterial Diastólica', 
                'Temperatura en °C', 'Saturación Oxígeno', 'Frecuencia Cardíaca',
                'Frecuencia Respiratoria', 'Glasgow']

pert = query[query['VALIDACIÓN'] == 'PERTINENTE']
no_pert = query[query['VALIDACIÓN'] == 'NO PERTINENTE']

for col in numeric_cols:
    if col in query.columns:
        # Convertir a numérico
        pert_vals = pd.to_numeric(pert[col].astype(str).str.replace(',', '.'), errors='coerce')
        no_pert_vals = pd.to_numeric(no_pert[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        if pert_vals.notna().sum() > 0 and no_pert_vals.notna().sum() > 0:
            print(f"\n{col}:")
            print(f"  PERTINENTE:     {pert_vals.mean():.2f} (n={pert_vals.notna().sum()})")
            print(f"  NO PERTINENTE:  {no_pert_vals.mean():.2f} (n={no_pert_vals.notna().sum()})")
