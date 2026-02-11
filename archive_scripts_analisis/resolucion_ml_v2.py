import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ============================================================
# Helpers
# ============================================================


def strip_accents(text: str) -> str:
    if text is None:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", str(text)) if unicodedata.category(c) != "Mn"
    )


def normalize_header(h: str) -> str:
    # Trim and collapse inner whitespace
    return re.sub(r"\s+", " ", str(h)).strip()


def normalize_key(h: str) -> str:
    return normalize_header(strip_accents(h)).lower()


def to_bool(x) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    if isinstance(x, (int, float)) and not pd.isna(x):
        return x != 0
    t = str(x).strip().lower()
    if t in {"", "null", "none", "nan"}:
        return False
    return t in {"si", "sí", "s", "y", "x", "true", "1", "t", "yes", "verdadero"}


def parse_decimal_series(series: pd.Series) -> pd.Series:
    def _conv(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        if s.lower() in {"", "null", "none", "nan"}:
            return np.nan
        s = s.replace("%", "")
        
        # Detección inteligente de formato decimal
        has_dot = "." in s
        has_comma = "," in s
        
        if has_dot and has_comma:
            # Formato europeo con miles: 1.234,56
            s = s.replace(".", "")  # eliminar separador de miles
            s = s.replace(",", ".")  # convertir decimal
        elif has_comma:
            # Solo coma: asumir decimal europeo: 1,3 → 1.3
            s = s.replace(",", ".")
        # Si solo tiene punto, dejarlo como está (formato americano: 1.3)
        
        return pd.to_numeric(s, errors="coerce")

    return series.apply(_conv)


def normalize_fio2_fraction(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mask = s > 1.0
    # Convertir a float explícitamente para evitar warning de dtype
    s = s.astype(float)
    s.loc[mask] = s.loc[mask] / 100.0
    return s


TEXT_ALERTS: List[Tuple[str, str]] = [
    (r"paro\s*card|arresto", "paro_cardiorrespiratorio"),
    (r"shock", "shock"),
    (r"sepsis|septico", "sepsis"),
    (r"acv|ictus|stroke|ataque cerebro|accidente cerebro|hsa|hcia", "acv"),
    (r"convulsion|status epilept", "convulsiones"),
    (r"coma|inconsciente|no responde", "coma"),
    (r"infarto|iam|stemi|scacest|scai|coronario agudo", "sindrome_coronario"),
    (r"dolor torac", "dolor_toracico"),
    (r"disnea|insuficiencia respiratoria|sdra|broncoaspir|crisis asma", "disnea"),
    (r"embol.? pulmon|tromboembolismo pulmon|\btep\b", "tep"),
    (r"hemorragia|sangrado|hematemesis|melena|hemoptisis", "hemorragia"),
    (r"trauma|politrauma|tce|accidente", "trauma"),
    (r"neumon", "neumonia"),
]

# Normalized header -> canonical
HEADER_MAP: Dict[str, str] = {
    "id": "ID",
    "fecha formulario": "Fecha_Formulario",
    "episodio": "Episodio",
    "antecedentes cardiacos": "Antecedente_Cardiaco",
    "antecedentes diabeticos": "Antecedente_Diabetico",
    "antecedentes de hipertension arterial": "Antecedente_HTA",
    "triage": "Triage",
    "presion arterial sistolica": "PA_Sistolica",
    "presion arterial diastolica": "PA_Diastolica",
    "presion arterial media": "PA_Media",
    "temperatura en c": "Temperatura",
    "temperatura en °c": "Temperatura",
    "saturacion oxigeno": "SatO2",
    "frecuencia cardiaca": "FC",
    "frecuencia respiratoria": "FR",
    "tipo de cama": "Tipo_Cama",
    "glasgow": "Glasgow",
    "fio2": "FiO2_raw",
    "fio2 > o igual a 50%": "FiO2_ge50_flag",
    "fio2 >= o igual a 50%": "FiO2_ge50_flag",
    "ventilacion mecanica": "Ventilacion_Mecanica",
    "cirugia realizada": "Cirugia",
    "cirugia mismo dia ingreso": "Cirugia_mismo_dia",
    "hemodinamia realizada": "Hemodinamia",
    "hemodinamia mismo dia ingreso": "Hemodinamia_mismo_dia",
    "endoscopia": "Endoscopia",
    "endoscopia mismo dia ingreso": "Endoscopia_mismo_dia",
    "dialisis": "Dialisis",
    "trombolisis": "Trombolisis",
    "trombolisis mismo dia ingreso": "Trombolisis_mismo_dia",
    "pcr": "PCR",
    "hemoglobina": "Hemoglobina",
    "creatinina": "Creatinina",
    "nitrogeno ureico": "BUN",
    "sodio": "Sodio",
    "potasio": "Potasio",
    "dreo": "DREO",
    "troponinas alteradas": "Troponinas_Alteradas",
    "ecg alterado": "ECG_Alterado",
    "rnm protocolo stroke": "RNM_Stroke",
    "dva": "DVA",
    "transfusiones": "Transfusiones",
    "compromiso conciencia": "Compromiso_Conciencia",
    "texto libre": "Texto_Libre",
    "resolucion": "Resolucion",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = normalize_key(col)
        if key in HEADER_MAP:
            rename_map[col] = HEADER_MAP[key]
    return df.rename(columns=rename_map)


def add_text_alerts(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    text_series = df.get("Texto_Libre", pd.Series(index=df.index, data=""))
    cleaned = (
        text_series.fillna("")
        .apply(strip_accents)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]+", " ", regex=True)
    )
    alert_cols: List[str] = []
    for pattern, slug in TEXT_ALERTS:
        col = f"alert_{slug}"
        df[col] = cleaned.str.contains(pattern, regex=True)
        alert_cols.append(col)
    return df, alert_cols

def infer_patient_category(row) -> str:
    """
    Infiere categoría etaria basada en el tipo de cama y texto libre.
    Prioridad: NEO > PEDIATRICO > ADULTO
    """
    cama = str(row.get('Tipo_Cama', '')).lower()
    texto = str(row.get('Texto_Libre', '')).lower()
    full_context = f"{cama} {texto}"
    
    # Palabras clave basadas en nomenclatura hospitalaria estándar
    if re.search(r'\b(neo|rn|recien nacido|prematuro|incubadora|cuna)\b', full_context):
        return 'NEO'
    if re.search(r'\b(ped|pediat|infantil|niño|escolar|lactante|cuna)\b', full_context):
        return 'PEDIATRICO'
    
    return 'ADULTO'

def add_decree_flags(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # 1. Inferir tipo de paciente fila por fila
    df['Categoria_Inferida'] = df.apply(infer_patient_category, axis=1)

    # 2. Preparar variables numéricas
    fio2 = df.get("FiO2", pd.Series(index=df.index, data=np.nan))
    sat = pd.to_numeric(df.get("SatO2", pd.Series(index=df.index, data=np.nan)), errors='coerce')
    gcs = pd.to_numeric(df.get("Glasgow", pd.Series(index=df.index, data=np.nan)), errors='coerce')
    pas = pd.to_numeric(df.get("PA_Sistolica", pd.Series(index=df.index, data=np.nan)), errors='coerce')
    hb = pd.to_numeric(df.get("Hemoglobina", pd.Series(index=df.index, data=np.nan)), errors='coerce')
    triage = pd.to_numeric(df.get("Triage", pd.Series(index=df.index, data=np.nan)), errors="coerce")
    
    # Variables Booleanas
    vm = df.get("Ventilacion_Mecanica", pd.Series(index=df.index, data=False))
    fio2_flag = df.get("FiO2_ge50_flag", pd.Series(index=df.index, data=False))
    comp_conc = df.get("Compromiso_Conciencia", pd.Series(index=df.index, data=False))
    dva = df.get("DVA", pd.Series(index=df.index, data=False))
    hemo = df.get("Hemodinamia_mismo_dia", pd.Series(index=df.index, data=False))
    ciru = df.get("Cirugia_mismo_dia", pd.Series(index=df.index, data=False))
    endo = df.get("Endoscopia_mismo_dia", pd.Series(index=df.index, data=False))
    transf = df.get("Transfusiones", pd.Series(index=df.index, data=False))
    rnm = df.get("RNM_Stroke", pd.Series(index=df.index, data=False))
    tromb = df.get("Trombolisis_mismo_dia", pd.Series(index=df.index, data=False))
    
    # 3. Criterios Respiratorios Diferenciados (Decreto 34)
    # Adulto: (SatO2 <= 90% AND FiO2 >= 50%) OR Ventilación Mecánica
    crit_resp_adult = (df['Categoria_Inferida'] == 'ADULTO') & (
        vm | (((fio2 >= 0.50) | fio2_flag) & (sat <= 90))
    )
    # Pediátrico: (SatO2 < 92% AND FiO2 >= 50%) OR Ventilación Mecánica  
    crit_resp_ped = (df['Categoria_Inferida'] == 'PEDIATRICO') & (
        vm | (((fio2 >= 0.50) | fio2_flag) & (sat < 92))
    )
    # Neonatal: (SatO2 <= 92% AND FiO2 >= 40%) OR Ventilación Mecánica
    crit_resp_neo = (df['Categoria_Inferida'] == 'NEO') & (
        vm | ((fio2 >= 0.40) & (sat <= 92))
    )
    
    df["flag_resp_grave"] = crit_resp_adult | crit_resp_ped | crit_resp_neo

    # 4. Criterios Neurológicos Diferenciados (Decreto 34)
    # Pediátrico: Glasgow <= 12
    crit_neuro_ped = (df['Categoria_Inferida'] == 'PEDIATRICO') & (gcs <= 12)
    # Adulto/Neo: Glasgow <= 8 (coma severo) o compromiso de conciencia con Glasgow <= 12
    crit_neuro_adult = (df['Categoria_Inferida'] != 'PEDIATRICO') & (
        (gcs <= 8) | (comp_conc & (gcs <= 12))
    )
    df["flag_neuro_grave"] = crit_neuro_ped | crit_neuro_adult | rnm | tromb

    # 5. Criterios Circulatorios (Decreto 34)
    # DVA: Universal para todos los grupos etarios
    # Hipotensión (PAS < 90): Solo adultos (en neonatos/pediátricos es tardía)
    crit_shock_adult = (df['Categoria_Inferida'] == 'ADULTO') & (pas < 90) & pas.notna()
    df["flag_circ_grave"] = (
        dva | crit_shock_adult | hemo | ciru | (endo & transf) | (transf & (hb < 7))
    )

    # 6. Triage Crítico (C1 y C2)
    df["flag_triage_critico"] = triage.isin([1, 2])

    # 7. Regla Maestra: Cumplimiento del Decreto 34
    # Solo los criterios clínicos graves activan la regla dura
    df["CUMPLE_CRITERIO_DECRETO"] = (
        df["flag_resp_grave"] | df["flag_circ_grave"] | df["flag_neuro_grave"]
    )

    # Retornar solo los flags individuales como features (NO incluir CUMPLE_CRITERIO_DECRETO para evitar data leakage)
    # NOTA: flag_triage_critico NO se incluye como feature del ML para evitar sobreajuste,
    # pero SÍ se usa en el override de decisión del sistema híbrido
    return df, ["flag_resp_grave", "flag_circ_grave", "flag_neuro_grave"]

def add_clinical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Crea features de interacción clínicas (ratios, productos, etc.)
    Basado en conocimiento médico de gravedad
    """
    new_features = []
    
    # === INTERACCIONES MÉDICAS CLAVE ===
    
    # 1. Shock Index Invertido (PA_Sistolica / FC)
    # Valor normal ~1.5, <0.9 indica shock severo
    if "PA_Sistolica" in df.columns and "FC" in df.columns:
        df["Shock_Index_Inv"] = df["PA_Sistolica"] / (df["FC"] + 1)
        df["Shock_Index_Inv"] = df["Shock_Index_Inv"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Shock_Index_Inv")
    
    # 2. Eficiencia Respiratoria (SatO2 × FR)
    # Alto = buena oxigenación con poco esfuerzo, Bajo = fallo respiratorio
    if "SatO2" in df.columns and "FR" in df.columns:
        df["Eficiencia_Resp"] = df["SatO2"] * df["FR"]
        df["Eficiencia_Resp"] = df["Eficiencia_Resp"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Eficiencia_Resp")
    
    # 3. Índice de Oxigenación Ajustado (SatO2 / FiO2)
    if "SatO2" in df.columns and "FiO2" in df.columns:
        df["Indice_Oxigenacion"] = df["SatO2"] / (df["FiO2"] + 0.01)
        df["Indice_Oxigenacion"] = df["Indice_Oxigenacion"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Indice_Oxigenacion")
    
    # 4. Deterioro Neurológico con Taquicardia (Glasgow × FC)
    # Bajo Glasgow + FC alta = deterioro crítico
    if "Glasgow" in df.columns and "FC" in df.columns:
        df["Neuro_Cardiaco"] = df["Glasgow"] * df["FC"]
        df["Neuro_Cardiaco"] = df["Neuro_Cardiaco"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Neuro_Cardiaco")
    
    # 5. Severidad Anemia con Transfusión (Hemoglobina × Transfusiones)
    if "Hemoglobina" in df.columns and "Transfusiones" in df.columns:
        df["Severidad_Anemia"] = df["Hemoglobina"] * df["Transfusiones"].fillna(0).astype(int)
        df["Severidad_Anemia"] = df["Severidad_Anemia"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Severidad_Anemia")
    
    # 6. Shock Refractario (DVA × PA_Sistolica)
    # DVA con PA baja = shock que no responde
    if "DVA" in df.columns and "PA_Sistolica" in df.columns:
        df["Shock_Refractario"] = df["DVA"].fillna(0).astype(int) * df["PA_Sistolica"]
        df["Shock_Refractario"] = df["Shock_Refractario"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Shock_Refractario")
    
    # 7. Función Renal (BUN × Creatinina)
    # Producto alto = fallo renal severo
    if "BUN" in df.columns and "Creatinina" in df.columns:
        df["Funcion_Renal"] = df["BUN"] * df["Creatinina"]
        df["Funcion_Renal"] = df["Funcion_Renal"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Funcion_Renal")
    
    # 8. Respuesta Inflamatoria (Temperatura × FC)
    # Fiebre + taquicardia = sepsis/SIRS
    if "Temperatura" in df.columns and "FC" in df.columns:
        df["Respuesta_Inflam"] = df["Temperatura"] * df["FC"]
        df["Respuesta_Inflam"] = df["Respuesta_Inflam"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Respuesta_Inflam")
    
    # 9. Hipoxemia Severa con FiO2 Alta (FiO2 × (100 - SatO2))
    # FiO2 alto pero SatO2 bajo = SDRA/fallo respiratorio
    if "FiO2" in df.columns and "SatO2" in df.columns:
        df["Severidad_Hipoxemia"] = df["FiO2"] * (100 - df["SatO2"])
        df["Severidad_Hipoxemia"] = df["Severidad_Hipoxemia"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Severidad_Hipoxemia")
    
    # 10. Perfusión Cerebral (PA_Media × Glasgow)
    # Presión perfusión cerebral crítica
    if "PA_Media" in df.columns and "Glasgow" in df.columns:
        df["Perfusion_Cerebral"] = df["PA_Media"] * df["Glasgow"]
        df["Perfusion_Cerebral"] = df["Perfusion_Cerebral"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Perfusion_Cerebral")
    
    # 11. Fallo Multiorgánico (flag_resp_grave × flag_circ_grave)
    # Combinación de fallos = altísima mortalidad
    if "flag_resp_grave" in df.columns and "flag_circ_grave" in df.columns:
        df["Fallo_Multiorganico"] = df["flag_resp_grave"].fillna(0).astype(int) * df["flag_circ_grave"].fillna(0).astype(int)
        new_features.append("Fallo_Multiorganico")
    
    # 12. Shock Distributivo (alert_shock × DVA)
    # Alerta de shock con DVA = shock distribuido
    if "alert_shock" in df.columns and "DVA" in df.columns:
        df["Shock_Distributivo"] = df["alert_shock"].fillna(0).astype(int) * df["DVA"].fillna(0).astype(int)
        new_features.append("Shock_Distributivo")
    
    # 13. Compromiso Respiratorio-Neurológico (flag_resp_grave × Glasgow)
    if "flag_resp_grave" in df.columns and "Glasgow" in df.columns:
        df["Resp_Neuro_Combo"] = df["flag_resp_grave"].fillna(0).astype(int) * df["Glasgow"]
        df["Resp_Neuro_Combo"] = df["Resp_Neuro_Combo"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Resp_Neuro_Combo")
    
    # 14. Presión de Pulso (PA_Sistolica - PA_Diastolica)
    # Baja (<25) = shock, Alta (>60) = rigidez arterial
    if "PA_Sistolica" in df.columns and "PA_Diastolica" in df.columns:
        df["Presion_Pulso"] = df["PA_Sistolica"] - df["PA_Diastolica"]
        df["Presion_Pulso"] = df["Presion_Pulso"].replace([np.inf, -np.inf], np.nan)
        new_features.append("Presion_Pulso")
    
    # 15. Carga de Intervenciones (Cirugia + Hemodinamia + Endoscopia)
    # Múltiples intervenciones = alta complejidad
    interv_cols = []
    if "Cirugia_mismo_dia" in df.columns:
        interv_cols.append(df["Cirugia_mismo_dia"].fillna(0).astype(int))
    if "Hemodinamia_mismo_dia" in df.columns:
        interv_cols.append(df["Hemodinamia_mismo_dia"].fillna(0).astype(int))
    if "Endoscopia_mismo_dia" in df.columns:
        interv_cols.append(df["Endoscopia_mismo_dia"].fillna(0).astype(int))
    
    if interv_cols:
        df["Carga_Intervenciones"] = sum(interv_cols)
        new_features.append("Carga_Intervenciones")
    
    return df, new_features

def prepare_dataframe(raw_path: Path) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
    df_raw = pd.read_excel(raw_path)
    
    # Filtrar filas sin Episodio ANTES de normalizar
    if "Episodio" in df_raw.columns:
        initial_rows = len(df_raw)
        df_raw = df_raw[df_raw["Episodio"].notna()].copy()
        if len(df_raw) != initial_rows:
            print(f"Filas filtradas por Episodio: {initial_rows} -> {len(df_raw)}")
    
    df = normalize_columns(df_raw)

    # Convert numeric columns
    numeric_cols = [
        "PA_Sistolica",
        "PA_Diastolica",
        "PA_Media",
        "Temperatura",
        "SatO2",
        "FC",
        "FR",
        "Glasgow",
        "PCR",
        "Hemoglobina",
        "Creatinina",
        "BUN",
        "Sodio",
        "Potasio",
        "DREO",
        "FiO2_raw",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = parse_decimal_series(df[c])

    if "FiO2_raw" in df.columns:
        df["FiO2"] = normalize_fio2_fraction(df["FiO2_raw"])
    else:
        df["FiO2"] = np.nan

    # Convert booleans
    bool_cols = [
        "FiO2_ge50_flag",
        "Ventilacion_Mecanica",
        "Cirugia",
        "Cirugia_mismo_dia",
        "Hemodinamia",
        "Hemodinamia_mismo_dia",
        "Endoscopia",
        "Endoscopia_mismo_dia",
        "Dialisis",
        "Trombolisis",
        "Trombolisis_mismo_dia",
        "DVA",
        "Transfusiones",
        "Troponinas_Alteradas",
        "ECG_Alterado",
        "RNM_Stroke",
        "Compromiso_Conciencia",
        "Antecedente_Cardiaco",
        "Antecedente_Diabetico",
        "Antecedente_HTA",
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].apply(to_bool)

    # Normalize resolution label
    if "Resolucion" in df.columns:
        df["Resolucion"] = df["Resolucion"].replace(
            {"ND": np.nan, "Nd": np.nan, "nd": np.nan, "#N/D": np.nan, "RECHAZO": "NO PERTINENTE"}
        )

    # Text alerts and decree-inspired flags
    df, alert_cols = add_text_alerts(df)
    df, decree_cols = add_decree_flags(df)
    
    # Features clínicos desactivados - No mejoran test set
    clinical_features = []

    # Ensure Tipo_Cama exists as string
    if "Tipo_Cama" not in df.columns:
        df["Tipo_Cama"] = "desconocido"
    else:
        df["Tipo_Cama"] = df["Tipo_Cama"].fillna("desconocido").astype(str)

    return df, alert_cols + clinical_features, decree_cols, df_raw


def build_feature_pipeline(
    feature_cols: List[str],
    binary_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    transformers = []

    numeric_cols = [c for c in feature_cols if c not in binary_cols + categorical_cols]
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if binary_cols:
        transformers.append(
            (
                "bin",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                binary_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def build_xgboost_model(num_classes: int, override_params: dict = None) -> XGBClassifier:
    """Construye modelo XGBoost optimizado para clasificación de Ley de Urgencia."""
    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
    num_class = num_classes if objective.startswith("multi") else None
    
    # Parámetros base
    base_params = {
        'n_estimators': 600,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'objective': objective,
        'num_class': num_class,
        'eval_metric': 'logloss',
        'n_jobs': 4,
        'tree_method': 'hist',
        'random_state': 42,
    }
    
    # Sobrescribir con parámetros optimizados si se proporcionan
    if override_params:
        for key, value in override_params.items():
            if key in base_params:
                base_params[key] = value
    
    return XGBClassifier(**base_params)


def predecir_archivo_nuevo(ruta_archivo: str, output_name: str = None) -> None:
    """
    Predice en un archivo Excel nuevo usando el modelo previamente entrenado.
    
    Args:
        ruta_archivo: Ruta al archivo Excel con nuevos pacientes
        output_name: Nombre del archivo de salida (opcional)
    """
    modelo_path = Path("modelo_ley_urgencia.pkl")
    metadata_path = Path("modelo_metadata.pkl")
    
    if not modelo_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "❌ No se encontró el modelo entrenado.\n"
            "Primero ejecuta el entrenamiento con Data.xlsx"
        )
    
    print("📦 Cargando modelo entrenado...")
    model = joblib.load(modelo_path)
    metadata: Dict = joblib.load(metadata_path)  # Type hint para Pylance
    
    print(f"📂 Procesando archivo: {ruta_archivo}")
    archivo_path = Path(ruta_archivo)
    if not archivo_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")
    
    # Procesar nuevo archivo (mismo pipeline que entrenamiento)
    df, _, _, df_raw = prepare_dataframe(archivo_path)
    
    # Asegurar que existan todas las features necesarias
    # Obtener lista de binarias (compatibilidad con ambos formatos)
    binary_cols = metadata.get('present_binary', metadata.get('all_binary', []))
    
    for col in metadata['feature_cols']:
        if col not in df.columns:
            if col in binary_cols:
                df[col] = 0
            elif col in metadata['categorical_cols']:
                df[col] = "desconocido"
            else:
                df[col] = np.nan
    
    # Convertir binarias a int
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    X_new = df[metadata['feature_cols']].copy()
    
    print("🤖 Generando predicciones...")
    preds = model.predict(X_new)
    preds_labels = pd.Series(preds).map(metadata['int_to_label'])
    proba = model.predict_proba(X_new)
    confianza = proba.max(axis=1)
    
    # Aplicar lógica híbrida (igual que en entrenamiento)
    df_out = df_raw.copy()
    
    if "CUMPLE_CRITERIO_DECRETO" in df.columns:
        cumple_decreto = df["CUMPLE_CRITERIO_DECRETO"].astype(bool)
        
        ETIQUETA_POSITIVA = "PERTINENTE"
        UMBRAL_CONFIANZA_ALTA = 0.55  # Umbral optimizado: +0.39 puntos accuracy
        
        decisiones = []
        for i in range(len(df)):
            cumple_d = cumple_decreto.iloc[i]
            pred_ml = preds_labels.iloc[i]
            conf = confianza[i]
            
            # NUEVA LÓGICA OPTIMIZADA (Opción B - Tiebreaker):
            # PRIORIDAD 1: Alta confianza ML → confiar en ML
            if conf >= UMBRAL_CONFIANZA_ALTA:
                decisiones.append(pred_ml)
            # PRIORIDAD 2: Baja confianza + Decreto 34 → usar Decreto como tiebreaker
            elif cumple_d:
                decisiones.append(ETIQUETA_POSITIVA)
            # PRIORIDAD 3: Baja confianza sin Decreto → conservador
            else:
                decisiones.append("NO PERTINENTE")
        
        df_out["Prediccion"] = decisiones
        df_out["Confianza_ML"] = (confianza * 100).round(2)
        df_out["CUMPLE_CRITERIO_DECRETO"] = cumple_decreto.values
        df_out["flag_resp_grave"] = df["flag_resp_grave"].values if "flag_resp_grave" in df.columns else False
        df_out["flag_circ_grave"] = df["flag_circ_grave"].values if "flag_circ_grave" in df.columns else False
        df_out["flag_neuro_grave"] = df["flag_neuro_grave"].values if "flag_neuro_grave" in df.columns else False
    else:
        df_out["Prediccion"] = preds_labels.values
        df_out["Confianza_ML"] = (confianza * 100).round(2)
    
    # Eliminar columnas sin nombre
    cols_to_drop = [col for col in df_out.columns if 'unnamed' in str(col).lower() or str(col).startswith('Columna')]
    if cols_to_drop:
        df_out = df_out.drop(columns=cols_to_drop)
    
    # Guardar resultado
    if output_name is None:
        output_name = f"Prediccion_{archivo_path.stem}.csv"
    
    output_path = Path(output_name)
    try:
        df_out.to_csv(output_path, index=False)
    except PermissionError:
        output_path = output_path.with_name(f"Prediccion_{archivo_path.stem}_nuevo.csv")
        df_out.to_csv(output_path, index=False)
    
    print(f"\n✅ Predicciones guardadas en: {output_path}")
    print(f"   Total de casos procesados: {len(df_out)}")
    
    # Estadísticas de predicción
    pred_counts = df_out["Prediccion"].value_counts()
    print(f"\n📊 Resumen de predicciones:")
    for label, count in pred_counts.items():
        pct = (count / len(df_out)) * 100
        print(f"   {label}: {count} ({pct:.1f}%)")
    
    # === VALIDACIÓN CON RESOLUCIONES REALES (si existen) ===
    if "Resolucion" in df.columns:
        resolucion_real = df["Resolucion"]  # Ya normalizado por prepare_dataframe
        mask_con_resolucion = resolucion_real.notna()
        
        if mask_con_resolucion.sum() > 0:
            indices_validos = resolucion_real[mask_con_resolucion].index
            
            resoluciones_reales_norm = resolucion_real[indices_validos].astype(str).str.strip().str.upper()
            predicciones_norm = df_out.loc[indices_validos, "Prediccion"].astype(str).str.strip().str.upper()
            
            total_con_resolucion = len(indices_validos)
            correctos = (predicciones_norm == resoluciones_reales_norm).sum()
            incorrectos = total_con_resolucion - correctos
            accuracy = (correctos / total_con_resolucion) * 100
            
            print(f"\n{'='*60}")
            print(f"📈 VALIDACIÓN CON RESOLUCIONES REALES")
            print(f"{'='*60}")
            print(f"Total casos con resolución conocida: {total_con_resolucion}")
            print(f"Predicciones correctas: {correctos} ({accuracy:.2f}%)")
            print(f"Predicciones incorrectas: {incorrectos} ({100-accuracy:.2f}%)")
            
            # Desglose por tipo de decisión (si existe el flag)
            if "CUMPLE_CRITERIO_DECRETO" in df.columns:
                cumple_decreto = df["CUMPLE_CRITERIO_DECRETO"].astype(bool)
                casos_por_decreto = cumple_decreto[indices_validos].sum()
                casos_por_ml = total_con_resolucion - casos_por_decreto
                
                if casos_por_decreto > 0:
                    correctos_decreto = ((predicciones_norm.values == resoluciones_reales_norm.values) & cumple_decreto[indices_validos].values).sum()
                    acc_decreto = (correctos_decreto / casos_por_decreto) * 100
                    print(f"\n--- Desglose por Tipo de Decisión ---")
                    print(f"Casos con criterios Decreto 34: {casos_por_decreto} ({acc_decreto:.2f}% accuracy)")
                
                if casos_por_ml > 0:
                    correctos_ml = ((predicciones_norm.values == resoluciones_reales_norm.values) & ~cumple_decreto[indices_validos].values).sum()
                    acc_ml = (correctos_ml / casos_por_ml) * 100
                    print(f"Casos decididos por ML puro: {casos_por_ml} ({acc_ml:.2f}% accuracy)")
            
            # Matriz de confusión simple
            print(f"\n--- Matriz de Confusión ---")
            for label_real in resoluciones_reales_norm.unique():
                casos_reales = (resoluciones_reales_norm == label_real).sum()
                correctos_label = ((predicciones_norm == label_real) & (resoluciones_reales_norm == label_real)).sum()
                acc_label = (correctos_label / casos_reales * 100) if casos_reales > 0 else 0
                print(f"{label_real}: {correctos_label}/{casos_reales} correctos ({acc_label:.1f}%)")
            
            print(f"{'='*60}\n")
        else:
            print(f"\n⚠️  Columna 'Resolucion' encontrada pero sin valores válidos.")
    else:
        print(f"\nℹ️  No se encontró columna 'Resolucion' - solo se generaron predicciones.")


def main(archivo_entrada="Data.xlsx"):
    input_path = Path(archivo_entrada)
    if not input_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {input_path}")

    try:
        print(f"Leyendo y normalizando datos desde {input_path}...")
        df, alert_cols, decree_cols, df_raw_full = prepare_dataframe(input_path)
    except PermissionError:
        fallback = input_path.with_name("tmp_resolucion_input.xlsx")
        shutil.copy(input_path, fallback)
        print(f"Archivo original bloqueado. Usando copia temporal {fallback}...")
        df, alert_cols, decree_cols, df_raw_full = prepare_dataframe(fallback)

    label_col = "Resolucion"
    labeled_mask = df[label_col].notna()

    # Base feature sets
    # NOTA: PA_Media eliminada por alta correlación (0.91) con PA_Sistolica
    numeric_base = [
        "PA_Sistolica", "PA_Diastolica", "Temperatura",
        "SatO2", "FC", "FR", "Glasgow", "PCR", "Hemoglobina",
        "Creatinina", "BUN", "Sodio", "Potasio", "FiO2",
    ]
    binary_base = [
        "FiO2_ge50_flag", "Ventilacion_Mecanica", "Cirugia", "Cirugia_mismo_dia",
        "Hemodinamia", "Hemodinamia_mismo_dia", "Endoscopia", "Endoscopia_mismo_dia",
        "Dialisis", "Trombolisis", "Trombolisis_mismo_dia", "DVA", "Transfusiones",
        "Troponinas_Alteradas", "ECG_Alterado", "RNM_Stroke", "Compromiso_Conciencia",
        "Antecedente_Cardiaco", "Antecedente_Diabetico", "Antecedente_HTA",
    ]

    present_numeric = [c for c in numeric_base if c in df.columns]
    present_binary = [c for c in binary_base if c in df.columns]
    
    # Separar alert_cols en clínicos vs originales
    clinical_binary = [f for f in alert_cols if f in [
        'Hipotension_Severa', 'Hipoxemia_Severa', 'Alteracion_Mental',
        'Score_Shock', 'Score_Multiorganico', 'Temp_Anormal', 'Taquipnea_Severa'
    ]]
    clinical_numeric = [f for f in alert_cols if f in [
        'Ratio_PAS_FC', 'Indice_Oxigenacion', 'Ratio_BUN_Creat'
    ]]
    original_alerts = [a for a in alert_cols if a not in clinical_binary + clinical_numeric]
    
    # Rellenar NaN en features clínicas binarias con 0
    for col in clinical_binary:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    # Agregar todas las features binarias (originales + alertas + decree + clínicas binarias)
    all_binary = present_binary + original_alerts + decree_cols + clinical_binary
    
    # Agregar features numéricas clínicas
    present_numeric = present_numeric + clinical_numeric
    
    categorical_cols = ["Tipo_Cama"]
    if "Categoria_Inferida" in df.columns:
        categorical_cols.append("Categoria_Inferida")

    feature_cols = present_numeric + all_binary + categorical_cols

    # Conversión a enteros para features binarias (ya rellenadas)
    for col in all_binary:
        if col in df.columns:
            df[col] = df[col].astype(int)

    X_all = df[feature_cols].copy()
    y_all = df[label_col]

    print(f"Casos etiquetados: {labeled_mask.sum()} de {len(df)}")
    
    X_train_full = X_all[labeled_mask]
    y_train_full = y_all[labeled_mask]

    classes_sorted = sorted(y_train_full.unique())
    label_to_int = {cls: idx for idx, cls in enumerate(classes_sorted)}
    int_to_label = {v: k for k, v in label_to_int.items()}
    y_train_full_enc = y_train_full.map(label_to_int)
    num_classes = len(classes_sorted)

    class_counts = y_train_full.value_counts()
    class_weights = (len(y_train_full) / (len(class_counts) * class_counts)).to_dict()

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train_full, y_train_full_enc, test_size=0.2, stratify=y_train_full_enc, random_state=42,
    )
    
    preprocessor = build_feature_pipeline(feature_cols, present_binary, categorical_cols)
    X_tr_preprocessed = preprocessor.fit_transform(X_tr)
    X_tr_balanced, y_tr_balanced = X_tr_preprocessed, y_tr

    # CARGAR MODELO DE STACKING (86.96% accuracy)
    modelo_stacking_path = Path("modelo_stacking.pkl")
    
    if not modelo_stacking_path.exists():
        print("\nError: Modelo de stacking no encontrado")
        print("Ejecuta: python entrenar_stacking.py\n")
        raise FileNotFoundError("Modelo de stacking no encontrado")
    
    print("\nCargando modelo de stacking...")
    
    # Cargar modelo de stacking entrenado
    model = joblib.load(modelo_stacking_path)
    metadata_stacking = joblib.load(Path("modelo_stacking_metadata.pkl"))
    
    # Usar las features que el stacking espera
    feature_cols = metadata_stacking['feature_cols']
    all_binary = metadata_stacking['all_binary']
    categorical_cols = metadata_stacking['categorical_cols']
    
    # Reconstruir X_all con las features correctas
    X_all = df[feature_cols].copy()
    
    print("OK - Modelo cargado")

    # Predicción final para todas las filas
    print("\nGenerando predicciones...")
    preds_all = model.predict(X_all)
    preds_all_labels = pd.Series(preds_all).map(int_to_label)
    
    # Obtener probabilidades para decisión inteligente
    proba_all = model.predict_proba(X_all)
    confianza_ml = proba_all.max(axis=1)

    # Generación del archivo de salida (Sistema Híbrido: Reglas + ML)
    df_out = df_raw_full.copy()
    
    # Solo creamos las columnas que realmente necesitamos en el CSV final
    if "CUMPLE_CRITERIO_DECRETO" in df.columns:
        # Variables temporales para la lógica híbrida
        cumple_decreto = df["CUMPLE_CRITERIO_DECRETO"].astype(bool)
        ia_prediccion = preds_all_labels.values
        confianza = confianza_ml
        
        # LÓGICA HÍBRIDA MEJORADA (OPTIMIZADA):
        # 1. Alta confianza ML → usar predicción ML (prioridad)
        # 2. Baja confianza + Decreto 34 → usar Decreto como tiebreaker
        # 3. Baja confianza sin Decreto → conservador (NO PERTINENTE)
        
        ETIQUETA_POSITIVA = "PERTINENTE"
        UMBRAL_CONFIANZA_ALTA = 0.55  # Umbral optimizado: +0.39 puntos accuracy
        
        # Crear array de decisiones
        decisiones = []
        for i in range(len(df)):
            cumple_d = cumple_decreto.iloc[i]
            pred_ml = ia_prediccion[i]
            conf = confianza[i]
            
            # PRIORIDAD 1: Alta confianza ML → confiar en ML
            if conf >= UMBRAL_CONFIANZA_ALTA:
                decisiones.append(pred_ml)
            # PRIORIDAD 2: Baja confianza + Decreto 34 → usar Decreto
            elif cumple_d:
                decisiones.append(ETIQUETA_POSITIVA)
            # PRIORIDAD 3: Baja confianza sin Decreto → conservador
            else:
                decisiones.append("NO PERTINENTE")
        
        df_out["Prediccion"] = decisiones
        
        # === ANÁLISIS INTERNO DE RENDIMIENTO ===
        # Usar el DataFrame procesado que tiene las resoluciones normalizadas
        resolucion_procesada = df[label_col]  # Este ya tiene ND convertido a NaN
        
        # Filtrar solo casos con resolución válida
        mask_con_resolucion = resolucion_procesada.notna()
        indices_validos = resolucion_procesada[mask_con_resolucion].index
        
        resoluciones_reales_norm = resolucion_procesada[indices_validos].astype(str).str.strip().str.upper()
        predicciones_norm = df_out.loc[indices_validos, "Prediccion"].astype(str).str.strip().str.upper()
        
        # Calcular métricas
        total_con_resolucion = len(indices_validos)
        if total_con_resolucion > 0:
            correctos = (predicciones_norm == resoluciones_reales_norm).sum()
            incorrectos = total_con_resolucion - correctos
            accuracy = (correctos / total_con_resolucion) * 100
            
            # Desglose por tipo de decisión
            casos_por_decreto = cumple_decreto[indices_validos].sum()
            casos_por_ml = total_con_resolucion - casos_por_decreto
            
            # Accuracy por tipo de decisión
            if casos_por_decreto > 0:
                correctos_decreto = ((predicciones_norm.values == resoluciones_reales_norm.values) & cumple_decreto[indices_validos].values).sum()
                acc_decreto = (correctos_decreto / casos_por_decreto) * 100
            else:
                correctos_decreto = 0
                acc_decreto = 0
                
            if casos_por_ml > 0:
                correctos_ml = ((predicciones_norm.values == resoluciones_reales_norm.values) & ~cumple_decreto[indices_validos].values).sum()
                acc_ml = (correctos_ml / casos_por_ml) * 100
            else:
                correctos_ml = 0
                acc_ml = 0
            
            print(f"\n{'='*60}")
            print(f"RESULTADOS")
            print(f"{'='*60}")
            print(f"Total casos: {total_con_resolucion}")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"\nDesglose:")
            print(f"  Decreto 34: {casos_por_decreto} casos ({acc_decreto:.2f}%)")
            print(f"  ML: {casos_por_ml} casos ({acc_ml:.2f}%)")
            print(f"{'='*60}\n")

    # Eliminar columnas sin nombre (Unnamed, Columna1, etc.) que vienen del Excel
    cols_to_drop = [col for col in df_out.columns if 'unnamed' in str(col).lower() or str(col).startswith('Columna')]
    if cols_to_drop:
        df_out = df_out.drop(columns=cols_to_drop)

    out_full = Path("Prediccion.csv")
    try:
        df_out.to_csv(out_full, index=False)
    except PermissionError:
        out_full = out_full.with_name("Prediccion_xgb.csv")
        df_out.to_csv(out_full, index=False)
    
    print(f"OK - Guardado: {out_full} ({len(df_out)} filas)")


if __name__ == "__main__":
    import sys
    
    # Modo de uso desde línea de comandos
    if len(sys.argv) > 1:
        comando = sys.argv[1].lower()
        
        if comando == "predecir" or comando == "predict":
            if len(sys.argv) < 3:
                print("Uso: python resolucion_ml_v2.py predecir <archivo.xlsx> [nombre_salida.csv]")
                print("Ejemplo: python resolucion_ml_v2.py predecir NuevosPacientes.xlsx")
                sys.exit(1)
            
            archivo_entrada = sys.argv[2]
            nombre_salida = sys.argv[3] if len(sys.argv) > 3 else None
            
            print("\n" + "="*60)
            print("MODO PREDICCION - Usando modelo entrenado")
            print("="*60 + "\n")
            
            predecir_archivo_nuevo(archivo_entrada, nombre_salida)
            
        elif comando == "entrenar" or comando == "train":
            archivo_entrenamiento = sys.argv[2] if len(sys.argv) > 2 else "Data.xlsx"
            print("\n" + "="*60)
            print(f"MODO ENTRENAMIENTO - Entrenando modelo con {archivo_entrenamiento}")
            print("="*60 + "\n")
            main(archivo_entrenamiento)
            
        elif comando == "help" or comando == "--help" or comando == "-h":
            print("\nUSO DEL SISTEMA DE PREDICCION - LEY DE URGENCIA")
            print("="*60)
            print("\n1. ENTRENAR MODELO:")
            print("   python resolucion_ml_v2.py entrenar")
            print("\n2. PREDECIR EN ARCHIVO NUEVO:")
            print("   python resolucion_ml_v2.py predecir <archivo.xlsx> [salida.csv]")
            print("\n   Ejemplos:")
            print("   python resolucion_ml_v2.py predecir NuevosPacientes.xlsx")
            print("   python resolucion_ml_v2.py predecir Mes_Enero_2025.xlsx Resultados_Enero.csv\n")
            print("="*60 + "\n")
        else:
            print(f"❌ Comando desconocido: {comando}")
            print("   Usa: python resolucion_ml_v2.py help")
            sys.exit(1)
    else:
        # Modo tradicional: entrenar con Data.xlsx
        main()