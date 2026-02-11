"""
Análisis PROFUNDO de TODOS los Excel
Compara CONTENIDO y SIGNIFICADO, no solo nombres de columnas
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

print("="*100)
print("🔍 ANÁLISIS PROFUNDO DE CONTENIDO - TODOS LOS ARCHIVOS EXCEL")
print("="*100)

# Archivos a analizar
archivos = [
    'Base MPP mes Octubre 2025.xlsx',
    'Base MPP Actualizada.xlsx',
    'Base MPP 2024-2025.xlsx',
    'query.xlsx',
    'query_octubre_2025_con_validacion.xlsx',
    'Data.xlsx',
    'Dataset_Combinado_Entrenamiento.xlsx',
    'Propuesta base MPP.xlsx',
    'Actividad LU HUC.xlsx',
    'Actividad LU HUC - Con Validacion - Formato Modelo.xlsx',
    'validacion_octubre_2025_resultados.xlsx',
]

# Variables clínicas que buscamos (diferentes nombres posibles)
variables_objetivo = {
    'Presión Arterial Sistólica': ['pa sistolica', 'pas', 'presion arterial sistolica', 'sistolica', 'pa_sistolica', 'presión sistolica'],
    'Presión Arterial Diastólica': ['pa diastolica', 'pad', 'presion arterial diastolica', 'diastolica', 'pa_diastolica', 'presión diastolica'],
    'Temperatura': ['temperatura', 'temp', 'temperatura en °c', 'temperatura c', 't°'],
    'Saturación Oxígeno': ['saturacion', 'sato2', 'sat o2', 'saturacion oxigeno', 'saturación', 'spo2'],
    'Frecuencia Cardíaca': ['fc', 'frecuencia cardiaca', 'frecuencia cardíaca', 'freq cardiaca', 'pulso'],
    'Frecuencia Respiratoria': ['fr', 'frecuencia respiratoria', 'freq respiratoria', 'respiratoria'],
    'Glasgow': ['glasgow', 'escala glasgow', 'gcs'],
    'FiO2': ['fio2', 'fio2', 'fi o2', 'fraccion inspirada'],
    'Ventilación Mecánica': ['ventilacion mecanica', 'ventilación', 'vm', 'ventilacion'],
    'Cirugía': ['cirugia', 'cirugía', 'cx', 'cirugia realizada'],
    'Hemodinamia': ['hemodinamia', 'hemodinamica', 'hd'],
    'Diálisis': ['dialisis', 'diálisis', 'hemodialisis'],
    'PCR': ['pcr', 'proteina c reactiva'],
    'Hemoglobina': ['hemoglobina', 'hb', 'hgb'],
    'Creatinina': ['creatinina', 'creat'],
    'Validación': ['validacion', 'validación', 'pertinente', 'resolucion', 'etiqueta', 'clasificacion']
}

def normalizar_nombre(nombre):
    """Normaliza nombre de columna para comparar"""
    if pd.isna(nombre):
        return ""
    return str(nombre).lower().strip().replace('_', ' ').replace('°', '').replace('  ', ' ')

def detectar_variable(columna_nombre, df, col_idx):
    """Detecta qué variable es basándose en nombre y contenido"""
    nombre_norm = normalizar_nombre(columna_nombre)
    
    # Intentar por nombre
    for var_obj, aliases in variables_objetivo.items():
        for alias in aliases:
            if alias in nombre_norm or nombre_norm in alias:
                return var_obj, 'nombre'
    
    # Intentar por contenido (valores únicos, rango, tipo)
    try:
        valores = df.iloc[:, col_idx].dropna()
        if len(valores) == 0:
            return None, None
            
        valores_unicos = valores.nunique()
        
        # Binarias (Si/No)
        if valores_unicos <= 3:
            muestra = valores.astype(str).str.lower().unique()
            if any(x in ['si', 'sí', 'no', 's', 'n'] for x in muestra):
                if 'cirug' in nombre_norm or 'cx' in nombre_norm:
                    return 'Cirugía', 'contenido'
                elif 'ventil' in nombre_norm or 'vm' in nombre_norm:
                    return 'Ventilación Mecánica', 'contenido'
                elif 'dialisis' in nombre_norm or 'hd' in nombre_norm:
                    return 'Diálisis', 'contenido'
        
        # Numéricas - por rango
        if pd.api.types.is_numeric_dtype(valores):
            min_val = valores.min()
            max_val = valores.max()
            
            # Temperatura (34-42°C)
            if 34 <= min_val and max_val <= 42 and max_val - min_val < 10:
                if 'temp' in nombre_norm:
                    return 'Temperatura', 'contenido'
            
            # SatO2 (50-100%)
            if 50 <= min_val and max_val <= 100 and valores.median() > 90:
                if 'sat' in nombre_norm or 'o2' in nombre_norm or 'ox' in nombre_norm:
                    return 'Saturación Oxígeno', 'contenido'
            
            # Presión Arterial Sistólica (80-250)
            if 80 <= min_val and max_val <= 250:
                if 'sist' in nombre_norm or 'pas' in nombre_norm:
                    return 'Presión Arterial Sistólica', 'contenido'
            
            # Presión Arterial Diastólica (40-150)
            if 40 <= min_val and max_val <= 150:
                if 'diast' in nombre_norm or 'pad' in nombre_norm:
                    return 'Presión Arterial Diastólica', 'contenido'
            
            # FC (30-200)
            if 30 <= min_val and max_val <= 200:
                if 'fc' in nombre_norm or 'card' in nombre_norm or 'pulso' in nombre_norm:
                    return 'Frecuencia Cardíaca', 'contenido'
            
            # FR (8-60)
            if 8 <= min_val and max_val <= 60 and max_val - min_val < 50:
                if 'fr' in nombre_norm or 'respir' in nombre_norm:
                    return 'Frecuencia Respiratoria', 'contenido'
            
            # Glasgow (3-15)
            if 3 <= min_val <= 15 and max_val <= 15:
                if 'glas' in nombre_norm or 'gcs' in nombre_norm:
                    return 'Glasgow', 'contenido'
        
        # Categóricas - validación
        if valores_unicos < 10:
            muestra_str = valores.astype(str).str.upper().unique()
            if any('PERTINENTE' in x for x in muestra_str):
                return 'Validación', 'contenido'
                
    except:
        pass
    
    return None, None

# Analizar cada archivo
resultados_archivos = []

for archivo in archivos:
    if not os.path.exists(archivo):
        continue
    
    print(f"\n{'='*100}")
    print(f"📄 {archivo}")
    print(f"{'='*100}")
    
    try:
        df = pd.read_excel(archivo, nrows=1000)  # Leer primeras 1000 filas para análisis
        
        print(f"\n   Filas: {len(df):,} (muestra)")
        print(f"   Columnas totales: {len(df.columns)}")
        
        # Detectar variables
        variables_encontradas = {}
        columnas_detectadas = []
        
        for idx, col in enumerate(df.columns):
            var_detectada, metodo = detectar_variable(col, df, idx)
            if var_detectada:
                variables_encontradas[var_detectada] = {
                    'columna_original': col,
                    'metodo': metodo,
                    'completitud': df[col].notna().sum() / len(df) * 100
                }
                columnas_detectadas.append(col)
        
        # Mostrar variables encontradas
        print(f"\n   ✅ Variables clínicas detectadas: {len(variables_encontradas)}")
        
        # Agrupar por tipo
        signos_vitales = ['Presión Arterial Sistólica', 'Presión Arterial Diastólica', 
                         'Temperatura', 'Saturación Oxígeno', 'Frecuencia Cardíaca', 
                         'Frecuencia Respiratoria', 'Glasgow']
        procedimientos = ['Ventilación Mecánica', 'Cirugía', 'Hemodinamia', 'Diálisis']
        laboratorios = ['PCR', 'Hemoglobina', 'Creatinina']
        
        sv_encontrados = [v for v in signos_vitales if v in variables_encontradas]
        proc_encontrados = [v for v in procedimientos if v in variables_encontradas]
        lab_encontrados = [v for v in laboratorios if v in variables_encontradas]
        tiene_validacion = 'Validación' in variables_encontradas
        
        print(f"\n      📊 Signos Vitales: {len(sv_encontrados)}/{len(signos_vitales)}")
        for var in sv_encontrados:
            info = variables_encontradas[var]
            print(f"         ✓ {var}: '{info['columna_original']}' ({info['completitud']:.0f}% completo)")
        
        if proc_encontrados:
            print(f"\n      💉 Procedimientos: {len(proc_encontrados)}/{len(procedimientos)}")
            for var in proc_encontrados:
                info = variables_encontradas[var]
                print(f"         ✓ {var}: '{info['columna_original']}' ({info['completitud']:.0f}% completo)")
        
        if lab_encontrados:
            print(f"\n      🧪 Laboratorios: {len(lab_encontrados)}/{len(laboratorios)}")
            for var in lab_encontrados:
                info = variables_encontradas[var]
                print(f"         ✓ {var}: '{info['columna_original']}' ({info['completitud']:.0f}% completo)")
        
        print(f"\n      🏷️  Validación: {'SÍ ✅' if tiene_validacion else 'NO ❌'}")
        if tiene_validacion:
            info = variables_encontradas['Validación']
            print(f"         Columna: '{info['columna_original']}'")
            print(f"         Completitud: {info['completitud']:.0f}%")
            
            # Mostrar distribución
            val_col = info['columna_original']
            dist = df[val_col].value_counts()
            print(f"         Distribución:")
            for val, count in dist.items():
                print(f"            - {val}: {count}")
        
        # Calcular score
        score_signos = len(sv_encontrados) / len(signos_vitales) * 100
        score_procedimientos = len(proc_encontrados) / len(procedimientos) * 100 if procedimientos else 0
        score_validacion = 100 if tiene_validacion else 0
        score_total = (score_signos * 0.5 + score_procedimientos * 0.3 + score_validacion * 0.2)
        
        print(f"\n   📈 SCORE DE CALIDAD:")
        print(f"      Signos Vitales: {score_signos:.0f}%")
        print(f"      Procedimientos: {score_procedimientos:.0f}%")
        print(f"      Tiene Validación: {'Sí' if tiene_validacion else 'No'}")
        print(f"      ═══════════════")
        print(f"      SCORE TOTAL: {score_total:.0f}%")
        
        resultados_archivos.append({
            'archivo': archivo,
            'filas': len(df),
            'columnas_totales': len(df.columns),
            'variables_detectadas': len(variables_encontradas),
            'signos_vitales': len(sv_encontrados),
            'procedimientos': len(proc_encontrados),
            'laboratorios': len(lab_encontrados),
            'tiene_validacion': tiene_validacion,
            'score': score_total,
            'variables_info': variables_encontradas
        })
        
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)[:80]}")

# RANKING FINAL
print(f"\n\n{'='*100}")
print("🏆 RANKING FINAL - MEJOR ARCHIVO PARA ENTRENAMIENTO")
print(f"{'='*100}\n")

resultados_archivos.sort(key=lambda x: x['score'], reverse=True)

print(f"{'Rank':<6} {'Archivo':<55} {'Filas':>8} {'Vars':>5} {'SV':>3} {'Valid':>6} {'Score':>6}")
print("-" * 100)

for i, r in enumerate(resultados_archivos, 1):
    valid_icon = "✅" if r['tiene_validacion'] else "❌"
    archivo_corto = r['archivo'][:53]
    
    print(f"{i:<6} {archivo_corto:<55} {r['filas']:>8,} {r['variables_detectadas']:>5} "
          f"{r['signos_vitales']:>3}/7 {valid_icon:>6} {r['score']:>5.0f}%")

# RECOMENDACIÓN
print(f"\n{'='*100}")
print("🎯 RECOMENDACIÓN FINAL")
print(f"{'='*100}\n")

if resultados_archivos:
    mejor = resultados_archivos[0]
    
    print(f"📌 MEJOR ARCHIVO: {mejor['archivo']}\n")
    print(f"   Razones:")
    print(f"   • {mejor['filas']:,} casos disponibles")
    print(f"   • {mejor['variables_detectadas']} variables clínicas detectadas")
    print(f"   • {mejor['signos_vitales']}/7 signos vitales completos")
    print(f"   • Validaciones: {'SÍ ✅' if mejor['tiene_validacion'] else 'NO ❌'}")
    print(f"   • Score de calidad: {mejor['score']:.0f}%")
    
    if mejor['tiene_validacion'] and mejor['score'] >= 70:
        print(f"\n   ✅ EXCELENTE para entrenamiento:")
        print(f"      • Tiene suficientes variables clínicas")
        print(f"      • Incluye etiquetas de validación")
        print(f"      • Datos suficientes para ML")
    elif mejor['score'] >= 70 and not mejor['tiene_validacion']:
        print(f"\n   ⚠️ BUENO pero requiere validaciones:")
        print(f"      • Buena calidad de datos clínicos")
        print(f"      • Falta columna de etiquetas (PERTINENTE/NO PERTINENTE)")
    else:
        print(f"\n   ⚠️ Calidad moderada")

print(f"\n{'='*100}")
