"""
Analiza TODOS los archivos Excel y compara con form_MPP.csv
"""
import pandas as pd
import os
from pathlib import Path

print("="*100)
print("ANÁLISIS COMPLETO DE ARCHIVOS - COMPARACIÓN CON FORM_MPP.CSV")
print("="*100)

# 1. Cargar form_MPP.csv como REFERENCIA
print("\n📋 FORMATO DE REFERENCIA: form_MPP.csv")
form_mpp = pd.read_csv('data/form_MPP.csv')
print(f"   Filas: {len(form_mpp):,}")
print(f"   Columnas: {len(form_mpp.columns)}")

# Columnas esperadas (sin las de validación agregadas)
columnas_validacion = ['validaciona', 'validacion2', 'validacion 3', 'validacion4', 
                       'validacion5', 'validacion6', 'Validacion']
columnas_esperadas = [col for col in form_mpp.columns if col not in columnas_validacion]

print(f"\n   Columnas base (sin validación): {len(columnas_esperadas)}")
print(f"   Columnas de validación agregadas: {columnas_validacion}")

# 2. Analizar todos los Excel
archivos_excel = [
    'Base MPP mes Octubre 2025.xlsx',
    'Base MPP Actualizada.xlsx',
    'Base MPP 2024-2025.xlsx',
    'query.xlsx',
    'query_octubre_2025_con_validacion.xlsx',
    'Data.xlsx',
    'Dataset_Combinado_Entrenamiento.xlsx',
    'Propuesta base MPP.xlsx',
    'Actividad LU HUC.xlsx',
    'Actividad LU HUC - Con Validacion.xlsx',
    'Actividad LU HUC - Con Validacion - Formato Modelo.xlsx',
    'validacion_octubre_2025_resultados.xlsx',
    'data/form_MPP.xlsx'
]

print("\n" + "="*100)
print("ANÁLISIS DE ARCHIVOS EXCEL")
print("="*100)

resultados = []

for archivo in archivos_excel:
    if not os.path.exists(archivo):
        continue
    
    try:
        df = pd.read_excel(archivo)
        
        # Analizar compatibilidad
        columnas_comunes = set(df.columns) & set(columnas_esperadas)
        columnas_faltantes = set(columnas_esperadas) - set(df.columns)
        columnas_extras = set(df.columns) - set(form_mpp.columns)
        
        # Verificar si tiene columna de validación
        tiene_validacion = any(col in df.columns for col in 
                              ['VALIDACIÓN', 'Validacion', 'validacion', 'PERTINENTE'])
        
        # Calcular % de compatibilidad
        compatibilidad = (len(columnas_comunes) / len(columnas_esperadas)) * 100
        
        resultado = {
            'archivo': archivo,
            'filas': len(df),
            'columnas': len(df.columns),
            'compatibilidad': compatibilidad,
            'columnas_comunes': len(columnas_comunes),
            'columnas_faltantes': len(columnas_faltantes),
            'tiene_validacion': tiene_validacion,
            'columnas_extras': len(columnas_extras),
            'faltantes_criticas': columnas_faltantes
        }
        resultados.append(resultado)
        
    except Exception as e:
        print(f"\n❌ ERROR en {archivo}: {str(e)[:50]}")

# Ordenar por compatibilidad
resultados.sort(key=lambda x: x['compatibilidad'], reverse=True)

print("\n📊 RANKING DE COMPATIBILIDAD CON form_MPP.csv:")
print("-" * 100)
print(f"{'Archivo':<60} {'Filas':>8} {'Cols':>5} {'Match':>6} {'Valid':>6}")
print("-" * 100)

for r in resultados:
    validacion_icon = "✅" if r['tiene_validacion'] else "❌"
    match_pct = f"{r['compatibilidad']:.0f}%"
    
    archivo_corto = r['archivo'][:58] if len(r['archivo']) > 58 else r['archivo']
    
    print(f"{archivo_corto:<60} {r['filas']:>8,} {r['columnas']:>5} {match_pct:>6} {validacion_icon:>6}")

# Detalles de los mejores candidatos
print("\n" + "="*100)
print("ANÁLISIS DETALLADO DE LOS MEJORES CANDIDATOS")
print("="*100)

# Top 3
for i, r in enumerate(resultados[:3], 1):
    print(f"\n{i}. {r['archivo']}")
    print(f"   {'─'*95}")
    print(f"   📊 Filas: {r['filas']:,}")
    print(f"   📋 Columnas: {r['columnas']}")
    print(f"   ✓ Compatibilidad: {r['compatibilidad']:.1f}%")
    print(f"   ✓ Columnas coincidentes: {r['columnas_comunes']}/{len(columnas_esperadas)}")
    print(f"   ⚠ Columnas faltantes: {r['columnas_faltantes']}")
    print(f"   ➕ Columnas extras: {r['columnas_extras']}")
    print(f"   🏷️  Tiene validación: {'SÍ ✅' if r['tiene_validacion'] else 'NO ❌'}")
    
    if r['columnas_faltantes'] > 0 and r['columnas_faltantes'] <= 10:
        print(f"\n   ⚠ Columnas faltantes críticas:")
        for col in list(r['faltantes_criticas'])[:10]:
            print(f"      - {col}")

# Verificar cuál archivo tiene formato idéntico a form_MPP
print("\n" + "="*100)
print("VERIFICACIÓN ESPECÍFICA: ¿ALGÚN EXCEL ES IDÉNTICO A form_MPP.csv?")
print("="*100)

form_mpp_xlsx = None
if os.path.exists('data/form_MPP.xlsx'):
    form_mpp_xlsx = pd.read_excel('data/form_MPP.xlsx')
    
    columnas_xlsx = set([c for c in form_mpp_xlsx.columns if c not in columnas_validacion])
    columnas_csv = set(columnas_esperadas)
    
    if columnas_xlsx == columnas_csv:
        print("\n✅ data/form_MPP.xlsx tiene FORMATO IDÉNTICO a form_MPP.csv")
        print(f"   Filas: {len(form_mpp_xlsx):,}")
        print(f"   Columnas base: {len(columnas_xlsx)}")
        
        # Verificar si tiene validación
        tiene_val = any(col in form_mpp_xlsx.columns for col in columnas_validacion)
        print(f"   Tiene columnas de validación: {'SÍ' if tiene_val else 'NO'}")
    else:
        diff = columnas_xlsx.symmetric_difference(columnas_csv)
        print(f"\n⚠️ data/form_MPP.xlsx tiene {len(diff)} columnas diferentes")

# RECOMENDACIÓN FINAL
print("\n" + "="*100)
print("🎯 RECOMENDACIÓN FINAL")
print("="*100)

# Encontrar el mejor candidato
mejor = resultados[0] if resultados else None

if mejor:
    print(f"\n📌 MEJOR ARCHIVO PARA ENTRENAMIENTO:")
    print(f"   Archivo: {mejor['archivo']}")
    print(f"   Razón: {mejor['compatibilidad']:.0f}% compatible con form_MPP.csv")
    print(f"   Filas disponibles: {mejor['filas']:,}")
    print(f"   Tiene validaciones: {'SÍ ✅' if mejor['tiene_validacion'] else 'NO ❌'}")
    
    if mejor['compatibilidad'] == 100 and mejor['tiene_validacion']:
        print(f"\n   ✅ ¡PERFECTO! Este archivo:")
        print(f"      • Tiene el mismo formato que form_MPP.csv")
        print(f"      • Incluye etiquetas de validación para entrenar")
        print(f"      • Está listo para usar")
    elif mejor['compatibilidad'] >= 90 and mejor['tiene_validacion']:
        print(f"\n   ⚠️ CASI PERFECTO, pero:")
        print(f"      • Le faltan {mejor['columnas_faltantes']} columnas")
        print(f"      • Se puede usar con ajustes menores")
    elif mejor['compatibilidad'] >= 90 and not mejor['tiene_validacion']:
        print(f"\n   ⚠️ PROBLEMA: Alta compatibilidad pero NO tiene validaciones")
        print(f"      • No se puede usar para entrenar (falta columna de etiquetas)")
    else:
        print(f"\n   ❌ PROBLEMA: Baja compatibilidad")
        print(f"      • Formato muy diferente a form_MPP.csv")

print("\n" + "="*100)
print("SITUACIÓN ACTUAL DEL MODELO")
print("="*100)

print("""
⚠️ PROBLEMA DETECTADO:

El modelo actual fue entrenado con:
   • query.xlsx (variables clínicas)
   • Base MPP mes Octubre 2025.xlsx (validaciones)

Pero el formato de trabajo real es:
   • form_MPP.csv (formato unificado con todo)

OPCIONES:

1. Entrenar con archivo que coincida 100% con form_MPP.csv
   ✅ Modelo usará exactamente las mismas columnas
   ❌ Requiere archivo con validaciones en formato form_MPP

2. Convertir form_MPP.csv + agregar validaciones
   ✅ Formato perfecto
   ❌ Requiere etiquetar manualmente los casos

3. Adaptar modelo actual para trabajar con form_MPP.csv
   ✅ Aprovecha modelo existente
   ⚠️ Requiere mapeo de columnas
""")

print("="*100)
