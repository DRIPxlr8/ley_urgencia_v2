"""
Análisis profundo de errores para identificar patrones y mejorar modelo
"""
import pandas as pd
import numpy as np

print("="*80)
print("ANÁLISIS DE ERRORES DEL MODELO")
print("="*80)

# Cargar resultados
df = pd.read_excel('validacion_octubre_2025_resultados.xlsx')

print(f"\n📊 Resumen:")
print(f"   Total casos: {len(df)}")
print(f"   Correctos: {df['CORRECTO'].sum()} ({df['CORRECTO'].sum()/len(df)*100:.1f}%)")
print(f"   Errores: {(~df['CORRECTO']).sum()} ({(~df['CORRECTO']).sum()/len(df)*100:.1f}%)")

# Separar errores
errores = df[~df['CORRECTO']].copy()
correctos = df[df['CORRECTO']].copy()

print(f"\n❌ Tipos de errores:")
falsos_positivos = errores[errores['PREDICCIÓN'] == 'PERTINENTE']
falsos_negativos = errores[errores['PREDICCIÓN'] == 'NO PERTINENTE']

print(f"   Falsos Positivos (predice PERT, es NO PERT): {len(falsos_positivos)}")
print(f"   Falsos Negativos (predice NO PERT, es PERT): {len(falsos_negativos)}")

# Analizar características de falsos negativos (el problema principal)
if len(falsos_negativos) > 0:
    print(f"\n🔍 ANÁLISIS DE FALSOS NEGATIVOS (casos PERTINENTES no detectados)")
    print("="*80)
    
    # Convertir columnas numéricas
    numeric_cols = ['Presión Arterial Sistólica', 'Presión Arterial Diastólica', 
                    'Temperatura en °C', 'Saturación Oxígeno', 'Frecuencia Cardíaca',
                    'Frecuencia Respiratoria', 'Glasgow', 'PCR', 'Hemoglobina',
                    'Creatinina', 'Nitrógeno Ureico', 'Sodio', 'Potasio']
    
    for col in numeric_cols:
        if col in falsos_negativos.columns:
            falsos_negativos[col] = pd.to_numeric(falsos_negativos[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    for col in numeric_cols:
        if col in correctos.columns:
            correctos[col] = pd.to_numeric(correctos[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Solo PERTINENTES correctos
    pert_correctos = correctos[correctos['VALIDACIÓN'] == 'PERTINENTE']
    
    print(f"\nComparación: Falsos Negativos vs PERTINENTES bien detectados")
    print("─"*80)
    
    for col in ['Presión Arterial Sistólica', 'Presión Arterial Diastólica', 
                'Saturación Oxígeno', 'Frecuencia Cardíaca', 'Frecuencia Respiratoria', 
                'Glasgow', 'Temperatura en °C']:
        if col in falsos_negativos.columns:
            fn_mean = falsos_negativos[col].mean()
            pc_mean = pert_correctos[col].mean()
            
            if pd.notna(fn_mean) and pd.notna(pc_mean):
                diff = fn_mean - pc_mean
                print(f"\n{col}:")
                print(f"  Falsos Negativos: {fn_mean:.2f}")
                print(f"  PERT Correctos:   {pc_mean:.2f}")
                print(f"  Diferencia:       {diff:+.2f}")
    
    # Características binarias
    print(f"\n\nCaracterísticas binarias:")
    print("─"*80)
    
    binary_cols = ['Ventilación Mecánica', 'Cirugía Realizada', 'Hemodinamia Realizada',
                   'Diálisis', 'Trombólisis', 'DVA', 'Transfusiones']
    
    for col in binary_cols:
        if col in falsos_negativos.columns:
            fn_pct = (falsos_negativos[col] == 'Sí').sum() / len(falsos_negativos) * 100 if len(falsos_negativos) > 0 else 0
            pc_pct = (pert_correctos[col] == 'Sí').sum() / len(pert_correctos) * 100 if len(pert_correctos) > 0 else 0
            
            print(f"\n{col}:")
            print(f"  Falsos Negativos: {fn_pct:.1f}% Sí")
            print(f"  PERT Correctos:   {pc_pct:.1f}% Sí")

    # Ver casos específicos
    print(f"\n\n📋 CASOS ESPECÍFICOS - Falsos Negativos")
    print("="*80)
    
    for idx, row in falsos_negativos.iterrows():
        print(f"\nEpisodio {row['Episodio']}:")
        print(f"  PA: {row.get('Presión Arterial Sistólica', 'N/A')}/{row.get('Presión Arterial Diastólica', 'N/A')}")
        print(f"  SatO2: {row.get('Saturación Oxígeno', 'N/A')}%")
        print(f"  Glasgow: {row.get('Glasgow', 'N/A')}")
        print(f"  FR: {row.get('Frecuencia Respiratoria', 'N/A')}")
        print(f"  VM: {row.get('Ventilación Mecánica', 'N/A')}")
        print(f"  Cirugía: {row.get('Cirugía Realizada', 'N/A')}")
        print(f"  Hemodinamia: {row.get('Hemodinamia Realizada', 'N/A')}")

print(f"\n\n✅ PATRONES EN CASOS CORRECTOS")
print("="*80)

pert_ok = correctos[correctos['VALIDACIÓN'] == 'PERTINENTE']
no_pert_ok = correctos[correctos['VALIDACIÓN'] == 'NO PERTINENTE']

# Convertir numéricas
for col in numeric_cols:
    if col in pert_ok.columns:
        pert_ok[col] = pd.to_numeric(pert_ok[col].astype(str).str.replace(',', '.'), errors='coerce')
    if col in no_pert_ok.columns:
        no_pert_ok[col] = pd.to_numeric(no_pert_ok[col].astype(str).str.replace(',', '.'), errors='coerce')

print("\nDiferencias clave entre PERTINENTE y NO PERTINENTE (correctamente clasificados):")

for col in ['Presión Arterial Sistólica', 'Saturación Oxígeno', 'Glasgow', 'Frecuencia Respiratoria']:
    if col in pert_ok.columns and col in no_pert_ok.columns:
        pert_mean = pert_ok[col].mean()
        no_pert_mean = no_pert_ok[col].mean()
        
        if pd.notna(pert_mean) and pd.notna(no_pert_mean):
            diff = pert_mean - no_pert_mean
            print(f"\n{col}:")
            print(f"  PERTINENTE:    {pert_mean:.2f}")
            print(f"  NO PERTINENTE: {no_pert_mean:.2f}")
            print(f"  Diferencia:    {diff:+.2f}")

print("\n\n💡 RECOMENDACIONES PARA MEJORAR")
print("="*80)
print("\nBasado en el análisis de errores:")
print("1. Feature engineering para capturar mejor los patrones de falsos negativos")
print("2. Ajustar umbrales de decisión (reducir threshold para detectar más PERTINENTES)")
print("3. Crear features de interacción entre variables vitales")
print("4. Ponderar más los casos PERTINENTES en el entrenamiento")
