import pandas as pd

print("="*80)
print("ANÁLISIS DETALLADO: Dataset_Combinado_Entrenamiento.xlsx")
print("="*80)

df = pd.read_excel('Dataset_Combinado_Entrenamiento.xlsx')

print(f"\n📊 INFORMACIÓN GENERAL:")
print(f"   Total filas: {len(df):,}")
print(f"   Total columnas: {len(df.columns)}")

print(f"\n📋 COLUMNAS (primeras 40):")
for i, col in enumerate(df.columns[:40], 1):
    print(f"   {i:2}. {col}")

if len(df.columns) > 40:
    print(f"   ... (+{len(df.columns)-40} columnas más)")

print(f"\n🏷️  VALIDACIONES:")
if 'Resolucion' in df.columns:
    print(df['Resolucion'].value_counts())
    validados = df['Resolucion'].notna().sum()
    print(f"\n   Total validados: {validados:,} ({validados/len(df)*100:.1f}%)")
    print(f"   Sin validar: {len(df) - validados:,}")

print(f"\n📊 CALIDAD DE DATOS (primeras variables clínicas):")
variables = ['PAS', 'PAD', 'Temperatura', 'Saturacion_O2', 'FC', 'FR', 'Glasgow', 
             'PCR', 'Hemoglobina', 'Creatinina']

for var in variables:
    if var in df.columns:
        completitud = df[var].notna().sum() / len(df) * 100
        print(f"   {var:<20}: {completitud:>5.1f}% completo")

print("\n" + "="*80)
