# 📁 ESTRUCTURA DEL PROYECTO

```
ley_urgencia/
│
├── 📂 data/                                    # DATOS
│   ├── Base.xlsx                              # 5,250 casos etiquetados
│   ├── form_MPP.xlsx                          # 1,253 casos (525 validados)
│   └── form_MPP.csv                           # Formato CSV alternativo
│
├── 📂 tools/                                   # HERRAMIENTAS
│   └── conversor_archivos.py                  # Convertir Excel/CSV/JSON
│
├── 📂 archive_excel_antiguos/                 # ARCHIVOS HISTÓRICOS
│   ├── Base MPP 2024-2025.xlsx
│   ├── Base MPP Actualizada.xlsx
│   ├── Base MPP mes Octubre 2025.xlsx
│   └── ... (otros archivos antiguos)
│
├── 📂 archive_modelos_antiguos/               # MODELOS ANTERIORES
│   ├── entrenar_balanceado.py
│   ├── entrenar_ensemble.py
│   └── ... (scripts de entrenamiento antiguos)
│
├── 📂 archive_scripts_analisis/               # ANÁLISIS/EXPERIMENTOS
│   ├── analizar_errores_detalle.py
│   ├── buscar_mejor_modelo.py
│   └── ... (scripts de análisis)
│
├── 🎯 SCRIPTS PRINCIPALES
│   ├── inicio.py                              # Menu interactivo
│   ├── entrenar_con_form_mpp.py              # ⭐ ENTRENAR MODELO
│   ├── evaluar_modelo_final.py               # ⭐ EVALUAR MODELO
│   ├── streamlit_app_v3.py                   # Interfaz web
│   └── ejecutar_streamlit.py                 # Launcher Streamlit
│
├── 🤖 MODELO ENTRENADO
│   ├── modelo_ley_urgencia.pkl               # Random Forest (500 árboles)
│   ├── preprocessor.pkl                      # Pipeline preprocesamiento
│   ├── scaler.pkl                            # StandardScaler
│   └── modelo_metadata.pkl                   # Metadata (fecha, métricas)
│
├── 📊 RESULTADOS
│   ├── resultados_form_mpp_FINAL.xlsx        # Predicciones completas
│   └── errores_form_mpp_FINAL.csv            # 102 errores
│
├── 📖 DOCUMENTACIÓN
│   ├── README.md                             # Guía principal
│   ├── RESUMEN_MODELO_FINAL.md              # Documentación técnica
│   ├── ESTRUCTURA.md                        # Este archivo
│   └── requirements.txt                     # Dependencias Python
│
└── 📂 __pycache__/                           # Cache Python (ignorar)
```

---

## 🎯 SCRIPTS PRINCIPALES (USO DIARIO)

### 1. inicio.py
Menu interactivo con todas las opciones.

**Uso:**
```bash
python inicio.py
```

### 2. entrenar_con_form_mpp.py
Entrena modelo combinando Base.xlsx + form_MPP.xlsx.

**Uso:**
```bash
python entrenar_con_form_mpp.py
```

**Genera:**
- modelo_ley_urgencia.pkl
- preprocessor.pkl
- scaler.pkl
- modelo_metadata.pkl

### 3. evaluar_modelo_final.py
Evalúa modelo en los 525 casos validados de form_MPP.xlsx.

**Uso:**
```bash
python evaluar_modelo_final.py
```

**Genera:**
- resultados_form_mpp_FINAL.xlsx
- errores_form_mpp_FINAL.csv

### 4. streamlit_app_v3.py
Interfaz web interactiva para predicciones.

**Uso:**
```bash
python ejecutar_streamlit.py
# o directamente:
streamlit run streamlit_app_v3.py
```

---

## 📂 CARPETAS DE ARCHIVO

### archive_excel_antiguos/
Archivos Excel que ya no se usan activamente:
- Bases MPP anteriores
- Archivos de actividad LU HUC
- Datasets antiguos

### archive_modelos_antiguos/
Scripts de entrenamiento obsoletos:
- Modelos ensemble antiguos
- Experimentos con balanceo
- Versiones robustas anteriores

### archive_scripts_analisis/
Scripts de análisis y experimentación:
- Análisis de errores
- Búsqueda de mejores configuraciones
- Diagnósticos de datos

---

## 🗂️ CONVENCIONES

**Archivos activos:** Raíz del proyecto
**Archivos antiguos:** Carpetas archive_*
**Datos:** Carpeta data/
**Herramientas:** Carpeta tools/

**Nomenclatura:**
- Scripts principales: nombre_descriptivo.py
- Resultados finales: *_FINAL.xlsx / .csv
- Archivos de modelo: modelo_*.pkl
- Documentación: *.md (Markdown)

---

## 🔄 FLUJO DE TRABAJO

```
1. Datos nuevos → data/
2. Entrenar → entrenar_con_form_mpp.py
3. Evaluar → evaluar_modelo_final.py
4. Revisar → resultados_form_mpp_FINAL.xlsx
5. Analizar errores → errores_form_mpp_FINAL.csv
6. Usar modelo → streamlit_app_v3.py
```

---

## 📌 ARCHIVOS CLAVE

**Para entrenar:**
- data/Base.xlsx (5,250 casos)
- data/form_MPP.xlsx (525 validados)
- entrenar_con_form_mpp.py

**Para predecir:**
- modelo_ley_urgencia.pkl
- preprocessor.pkl
- scaler.pkl
- evaluar_modelo_final.py

**Para entender:**
- README.md (guía general)
- RESUMEN_MODELO_FINAL.md (detalles técnicos)
- ESTRUCTURA.md (este archivo)

---

**Última actualización:** 6 de febrero de 2026
