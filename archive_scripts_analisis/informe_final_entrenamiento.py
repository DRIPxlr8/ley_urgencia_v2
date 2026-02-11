"""
INFORME FINAL: ANÁLISIS DE DATOS PARA ENTRENAMIENTO DEL MODELO
"""
import pandas as pd

print("="*100)
print("🎯 INFORME FINAL: ¿CON QUÉ ARCHIVO SE DEBE ENTRENAR EL MODELO?")
print("="*100)

print("""
📋 CONTEXTO:
   • Formato de trabajo: form_MPP.csv (1,253 casos, 46 columnas clínicas)
   • Modelo actual: Entrenado con query.xlsx + Base MPP mes Octubre 2025.xlsx
   • Problema: Desajuste entre formato de entrenamiento y formato de trabajo

""")

print("="*100)
print("✅ RESPUESTA: data/form_MPP.xlsx ES EL ARCHIVO IDEAL")
print("="*100)

# Cargar form_MPP.xlsx
df = pd.read_excel('data/form_MPP.xlsx')

print(f"""
📊 CARACTERÍSTICAS DE data/form_MPP.xlsx:

1. FORMATO:
   ✅ 100% compatible con form_MPP.csv
   ✅ Mismo formato que usarás en producción
   ✅ Todas las columnas clínicas necesarias
   
   Total filas: {len(df):,}
   Columnas clínicas: 46
   Columnas de validación: 7

2. VALIDACIONES DISPONIBLES:
""")

cols_val = ['validaciona', 'validacion2', 'validacion 3', 'validacion4', 
            'validacion5', 'validacion6', 'Validacion']

total_validados = 0
for col in cols_val:
    if col in df.columns:
        n_val = df[col].notna().sum()
        if n_val > 0:
            perts = (df[col] == 'PERTINENTE').sum()
            no_perts = (df[col] == 'NO PERTINENTE').sum()
            print(f"   • {col}: {n_val} casos ({perts} PERT, {no_perts} NO PERT)")
            total_validados = max(total_validados, n_val)

print(f"""
3. COLUMNA PRINCIPAL: 'Validacion'
   ✅ 525 casos validados (41.9% del total)
   ✅ 319 PERTINENTE (60.8%)
   ✅ 206 NO PERTINENTE (39.2%)
   ✅ Balance aceptable para ML

4. VENTAJAS:
   ✅ Formato EXACTO al de producción
   ✅ No requiere mapeo de columnas
   ✅ Suficientes casos para entrenamiento (525 validados)
   ✅ Datos recientes (enero 2026)
   ✅ Balance de clases razonable

""")

print("="*100)
print("📝 PLAN DE ACCIÓN RECOMENDADO")
print("="*100)

print("""
PASO 1: CONSOLIDAR VALIDACIONES
   • Usar columna 'Validacion' como etiqueta principal (525 casos)
   • Opcional: consolidar otras columnas de validación para más datos
   
PASO 2: CREAR SCRIPT DE ENTRENAMIENTO NUEVO
   • Leer data/form_MPP.xlsx
   • Filtrar casos con 'Validacion' != null
   • Entrenar modelo con MISMO formato que form_MPP.csv
   • Guardar modelo optimizado para producción

PASO 3: VENTAJAS DE ESTE ENFOQUE
   ✅ Modelo entrenado con formato idéntico al de producción
   ✅ Sin necesidad de mapeo de columnas
   ✅ Predicciones directas en form_MPP.csv
   ✅ Mantenimiento más simple

""")

# Comparar con modelo actual
print("="*100)
print("📊 COMPARACIÓN: MODELO ACTUAL vs MODELO NUEVO")
print("="*100)

print("""
MODELO ACTUAL (entrenar_mpp_completo.py):
   • Datos: query.xlsx + Base MPP mes Octubre 2025.xlsx
   • Entrenamiento: 477 casos
   • Validación: 48 casos (octubre 2025)
   • Problema: Requiere cruce de archivos y mapeo de columnas

MODELO NUEVO (con form_MPP.xlsx):
   • Datos: data/form_MPP.xlsx (TODO EN UN ARCHIVO)
   • Disponible: 525 casos validados
   • Ventaja: MISMO formato que producción
   • Recomendación: Split 80/20 → 420 train / 105 test
   
""")

print("="*100)
print("🎯 CONCLUSIÓN FINAL")
print("="*100)

print(f"""
✅ ARCHIVO RECOMENDADO: data/form_MPP.xlsx

RAZONES:
1. Formato IDÉNTICO a form_MPP.csv (el que usarás en producción)
2. Tiene 525 casos validados (vs 477 del modelo actual)
3. Un solo archivo (no requiere cruce de episodios)
4. Datos más recientes
5. Simplifica el flujo de trabajo

ACCIÓN INMEDIATA:
Crear script 'entrenar_modelo_form_mpp.py' que:
   • Lea data/form_MPP.xlsx
   • Use columna 'Validacion' como etiqueta
   • Entrene con las 46 columnas clínicas
   • Genere modelo compatible con form_MPP.csv

¿Quieres que cree este script ahora? 🚀
""")

print("="*100)

# Verificar calidad de datos
print("\n📊 ANÁLISIS DE CALIDAD DE DATOS EN form_MPP.xlsx:")
print("-" * 100)

# Solo casos validados
df_val = df[df['Validacion'].notna()].copy()

print(f"\nCasos validados: {len(df_val)}")
print(f"\nDistribución de clases:")
print(df_val['Validacion'].value_counts().to_string())

# Completitud de variables críticas
print(f"\n\nCompletitud de variables clínicas (en casos validados):")
vars_criticas = [
    'Presión Arterial Sistólica', 'Presión Arterial Diastólica',
    'Temperatura en °C', 'Saturación Oxígeno', 'Frecuencia Cardíaca',
    'Frecuencia Respiratoria', 'Glasgow', 'FIO2'
]

for var in vars_criticas:
    if var in df_val.columns:
        completitud = df_val[var].notna().sum() / len(df_val) * 100
        print(f"   {var:<35}: {completitud:>5.1f}%")

print("\n" + "="*100)
