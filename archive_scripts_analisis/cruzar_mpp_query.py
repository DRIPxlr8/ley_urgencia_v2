"""
Cruza datos de Base MPP mes Octubre 2025 con query
por episodio, manteniendo datos de octubre y agregando VALIDACIÓN
"""
import pandas as pd
from datetime import datetime

print("📂 Cargando archivos...")
mpp = pd.read_excel('Base MPP mes Octubre 2025.xlsx')
query = pd.read_excel('query.xlsx')

print(f"   Base MPP: {len(mpp):,} filas")
print(f"   Query: {len(query):,} filas")

# Filtrar MPP por Fecha Alta en octubre 2025
print("\n📅 Filtrando MPP por Fecha Alta octubre 2025...")
mpp['Fecha Alta'] = pd.to_datetime(mpp['Fecha Alta'], errors='coerce')
mpp_octubre = mpp[
    (mpp['Fecha Alta'].dt.year == 2025) & 
    (mpp['Fecha Alta'].dt.month == 10) &
    (mpp['VALIDACIÓN'].notna())
].copy()

print(f"   MPP octubre 2025 con validación: {len(mpp_octubre):,} filas")
print(f"   Distribución validación MPP octubre:")
print(mpp_octubre['VALIDACIÓN'].value_counts().to_string())

# Limpiar columnas de episodio para el cruce
print("\n🔗 Preparando cruce por episodio...")
mpp_octubre['EPISODIO_LIMPIO'] = mpp_octubre['ESTADIA/EPISODIO'].astype(str).str.strip().str.upper()
query['EPISODIO_LIMPIO'] = query['Episodio'].astype(str).str.strip().str.upper()

# Cruce: query completo + validación de MPP octubre
print("\n⚙️ Realizando cruce...")
resultado = query.merge(
    mpp_octubre[['EPISODIO_LIMPIO', 'VALIDACIÓN']],
    on='EPISODIO_LIMPIO',
    how='left'
)

# Eliminar columna auxiliar
resultado = resultado.drop(columns=['EPISODIO_LIMPIO'])

# Estadísticas del cruce
total = len(resultado)
con_validacion = resultado['VALIDACIÓN'].notna().sum()
sin_validacion = total - con_validacion

print(f"\n📊 Resultados del cruce:")
print(f"   Total registros query: {total:,}")
print(f"   Con VALIDACIÓN: {con_validacion:,} ({con_validacion/total*100:.1f}%)")
print(f"   Sin VALIDACIÓN: {sin_validacion:,} ({sin_validacion/total*100:.1f}%)")

if 'VALIDACIÓN' in resultado.columns and resultado['VALIDACIÓN'].notna().sum() > 0:
    print(f"\n   Distribución VALIDACIÓN:")
    print(resultado['VALIDACIÓN'].value_counts().to_string())

# Filtrar solo registros con validación
resultado_final = resultado[resultado['VALIDACIÓN'].notna()].copy()

print(f"\n📋 Registros a guardar (solo con VALIDACIÓN): {len(resultado_final):,}")

# Guardar resultado
output_file = 'query_octubre_2025_con_validacion.xlsx'
print(f"\n💾 Guardando resultado en {output_file}...")
resultado_final.to_excel(output_file, index=False)

print(f"\n✅ Cruce completado exitosamente")
print(f"   Archivo generado: {output_file}")
