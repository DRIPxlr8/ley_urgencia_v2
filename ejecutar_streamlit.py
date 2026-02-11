"""
Script para ejecutar la aplicación Streamlit con verificación de dependencias
"""
import subprocess
import sys

def verificar_streamlit():
    """Verifica si streamlit está instalado"""
    try:
        import streamlit
        print(f"✅ Streamlit instalado (versión {streamlit.__version__})")
        return True
    except ImportError:
        print("❌ Streamlit no está instalado")
        return False

def instalar_streamlit():
    """Instala streamlit"""
    print("\n📦 Instalando Streamlit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    print("✅ Streamlit instalado correctamente")

def ejecutar_app():
    """Ejecuta la aplicación Streamlit"""
    print("\n🚀 Iniciando aplicación Streamlit...")
    print("\n" + "="*60)
    print("La aplicación se abrirá en tu navegador automáticamente")
    print("Si no se abre, ve a: http://localhost:8501")
    print("Para detener la app: Ctrl+C")
    print("="*60 + "\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "scripts/streamlit_app_v3.py"])
    except KeyboardInterrupt:
        print("\n👋 Aplicación cerrada correctamente")

if __name__ == "__main__":
    print("="*60)
    print(" SISTEMA INTELIGENTE - LEY DE URGENCIA")
    print(" Verificando dependencias...")
    print("="*60)
    
    if not verificar_streamlit():
        respuesta = input("\n¿Deseas instalar Streamlit ahora? (s/n): ")
        if respuesta.lower() in ['s', 'si', 'yes', 'y']:
            instalar_streamlit()
        else:
            print("\n❌ No se puede ejecutar sin Streamlit")
            print("Instala manualmente con: pip install streamlit")
            sys.exit(1)
    
    ejecutar_app()
