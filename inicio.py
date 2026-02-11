"""
INICIO RÁPIDO - SISTEMA LEY DE URGENCIA
========================================

Menu de opciones para trabajar con el modelo
"""

import os
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    while True:
        clear_screen()
        print("="*70)
        print("  SISTEMA ML - LEY DE URGENCIA DECRETO 34")
        print("  UC CHRISTUS Chile")
        print("="*70)
        print()
        print("Seleccione una opción:")
        print()
        print("  1. Entrenar nuevo modelo (Base.xlsx + form_MPP.xlsx)")
        print("  2. Evaluar modelo actual en form_MPP.xlsx")
        print("  3. Abrir interfaz web (Streamlit)")
        print("  4. Ver resultados actuales")
        print("  5. Ver resumen del modelo")
        print()
        print("  0. Salir")
        print()
        print("="*70)
        
        opcion = input("\nIngrese su opción: ").strip()
        
        if opcion == "1":
            print("\n🔄 Entrenando modelo...")
            print("Esto puede tardar varios minutos.\n")
            os.system("python scripts/entrenar_con_form_mpp.py")
            input("\nPresione ENTER para continuar...")
            
        elif opcion == "2":
            print("\n📊 Evaluando modelo...")
            os.system("python scripts/evaluar_modelo_final.py")
            input("\nPresione ENTER para continuar...")
            
        elif opcion == "3":
            print("\n🌐 Abriendo interfaz web...")
            print("Se abrirá en tu navegador. Presiona Ctrl+C para detener.\n")
            os.system("python ejecutar_streamlit.py")
            
        elif opcion == "4":
            print("\n📂 Abriendo archivo de resultados...")
            if os.name == 'nt':  # Windows
                os.system("start resultados\\resultados_form_mpp_FINAL.xlsx")
            else:  # Linux/Mac
                os.system("xdg-open resultados/resultados_form_mpp_FINAL.xlsx")
            input("\nPresione ENTER para continuar...")
            
        elif opcion == "5":
            print("\n📖 Resumen del modelo:")
            print()
            with open("docs/RESUMEN_MODELO_FINAL.md", "r", encoding="utf-8") as f:
                content = f.read()
                # Mostrar solo primeras 50 líneas
                lines = content.split("\n")
                for line in lines[:50]:
                    print(line)
                if len(lines) > 50:
                    print("\n... (ver RESUMEN_MODELO_FINAL.md para más detalles)")
            input("\nPresione ENTER para continuar...")
            
        elif opcion == "0":
            print("\n👋 ¡Hasta luego!")
            break
            
        else:
            print("\n❌ Opción inválida. Intente nuevamente.")
            input("\nPresione ENTER para continuar...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 ¡Hasta luego!")
        sys.exit(0)
