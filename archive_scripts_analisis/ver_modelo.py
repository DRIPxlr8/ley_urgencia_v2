import joblib

# Cargar modelo y metadata
modelo = joblib.load('modelo_ley_urgencia.pkl')
metadata = joblib.load('modelo_metadata.pkl')

print("\n" + "="*80)
print("MODELO ACTUAL EN USO")
print("="*80)

print(f"\n📅 Última modificación: 14-01-2026 16:24:33")
print(f"📦 Tamaño: 5,985 KB (~6 MB)")

print(f"\n🔧 Tipo de modelo: {metadata.get('model_type', 'N/A')}")
print(f"🎯 Threshold optimizado: {metadata.get('threshold', 'N/A')}")

print(f"\n📊 Conjunto de entrenamiento:")
print(f"   - Entrenamiento: {metadata.get('n_train', 'N/A')} casos")
print(f"   - Validación: {metadata.get('n_val', 'N/A')} casos")

print(f"\n🔢 Features utilizadas: {len(metadata.get('features', []))} features")
print(f"   - Numéricas: {len(metadata.get('numeric_features', []))}")
print(f"   - Binarias: {len(metadata.get('binary_features', []))}")
print(f"   - Categóricas: {len(metadata.get('categorical_features', []))}")

print(f"\n🏷️  Clases: {metadata.get('classes', [])}")

# Ver estructura del pipeline
print(f"\n⚙️  Arquitectura del modelo:")
print(f"   Tipo: {type(modelo).__name__}")

if hasattr(modelo, 'steps'):
    print(f"\n   Pipeline con {len(modelo.steps)} etapas:")
    for i, (name, step) in enumerate(modelo.steps, 1):
        print(f"      {i}. {name}: {type(step).__name__}")
        
        # Si es VotingClassifier, mostrar estimadores
        if hasattr(step, 'estimators'):
            print(f"         ↳ Ensemble de {len(step.estimators)} modelos:")
            for est_name, est in step.estimators:
                print(f"            • {est_name}: {type(est).__name__}")

print("\n" + "="*80)
print("SCRIPT QUE GENERÓ ESTE MODELO")
print("="*80)

model_type = metadata.get('model_type', '')
if 'mpp_completo' in model_type:
    print("\n✅ entrenar_mpp_completo.py")
    print("   Modelo entrenado con datos MPP completos (sin octubre)")
elif 'ensemble' in model_type:
    print("\n✅ entrenar_ensemble.py")
    print("   Modelo ensemble con múltiples algoritmos")
elif 'mejorado' in model_type:
    print("\n✅ entrenar_modelo_mejorado.py")
    print("   Modelo con feature engineering avanzado")
elif 'robusto' in model_type:
    print("\n✅ entrenar_modelo_robusto.py")
    print("   Modelo robusto con manejo de datos faltantes")
else:
    print(f"\n⚠️  Tipo: {model_type}")

print("\n" + "="*80)
