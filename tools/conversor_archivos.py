"""
Conversor Universal de Archivos
===============================
Convierte entre CSV y XLSX.
Diseño moderno con CustomTkinter (bordes redondeados).
"""

import customtkinter as ctk
import pandas as pd
import os
from pathlib import Path
import threading
from tkinter import filedialog, messagebox

# Configurar tema
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ConversorUniversal(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Conversor")
        self.geometry("750x550")
        self.minsize(650, 500)
        
        # Variables
        self.archivos_seleccionados = []
        self.carpeta_destino = None  # Se establecerá cuando se seleccionen archivos
        self.formato_origen = ctk.StringVar(value="csv")
        self.formato_destino = ctk.StringVar(value="xlsx")
        
        # Colores personalizados
        self.card_color = "#2d3748"  # Gris azulado más claro
        self.accent_color = "#e94560"
        
        # Sync automático de formatos
        self.formato_origen.trace_add("write", self._sync_formato_destino)
        
        self._crear_interfaz()
    
    def _crear_interfaz(self):
        # Container principal con padding
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=30, pady=20)
        
        # === HEADER ===
        header = ctk.CTkLabel(main_frame,
                             text="⚡ Conversor Universal",
                             font=ctk.CTkFont(size=22, weight="bold"))
        header.pack(anchor="w", pady=(0, 15))
        
        # === SECCIÓN FORMATO ===
        formato_frame = ctk.CTkFrame(main_frame, fg_color=self.card_color, corner_radius=12)
        formato_frame.pack(fill="x", pady=(0, 10))
        
        formato_inner = ctk.CTkFrame(formato_frame, fg_color="transparent")
        formato_inner.pack(fill="x", padx=15, pady=12)
        
        ctk.CTkLabel(formato_inner,
                    text="FORMATO",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color="#888888").pack(anchor="w")
        
        # Grid de formatos
        formatos_row = ctk.CTkFrame(formato_inner, fg_color="transparent")
        formatos_row.pack(fill="x", pady=(8, 0))
        
        # Origen
        origen_col = ctk.CTkFrame(formatos_row, fg_color="transparent")
        origen_col.pack(side="left", expand=True, fill="x")
        
        ctk.CTkLabel(origen_col, text="Desde", font=ctk.CTkFont(size=10),
                    text_color="#aaaaaa").pack(anchor="w")
        
        origen_btns = ctk.CTkFrame(origen_col, fg_color="transparent")
        origen_btns.pack(anchor="w", pady=(5, 0))
        
        for fmt in ["csv", "xlsx"]:
            btn = ctk.CTkRadioButton(origen_btns, text=fmt.upper(),
                                     variable=self.formato_origen, value=fmt,
                                     font=ctk.CTkFont(size=12, weight="bold"),
                                     fg_color=self.accent_color,
                                     hover_color="#ff6b6b")
            btn.pack(side="left", padx=(0, 15))
        
        # Flecha
        ctk.CTkLabel(formatos_row, text="→",
                    font=ctk.CTkFont(size=20, weight="bold"),
                    text_color=self.accent_color).pack(side="left", padx=20)
        
        # Destino
        destino_col = ctk.CTkFrame(formatos_row, fg_color="transparent")
        destino_col.pack(side="left", expand=True, fill="x")
        
        ctk.CTkLabel(destino_col, text="Hacia", font=ctk.CTkFont(size=10),
                    text_color="#aaaaaa").pack(anchor="w")
        
        destino_btns = ctk.CTkFrame(destino_col, fg_color="transparent")
        destino_btns.pack(anchor="w", pady=(5, 0))
        
        for fmt in ["xlsx", "csv"]:
            btn = ctk.CTkRadioButton(destino_btns, text=fmt.upper(),
                                     variable=self.formato_destino, value=fmt,
                                     font=ctk.CTkFont(size=12, weight="bold"),
                                     fg_color=self.accent_color,
                                     hover_color="#ff6b6b")
            btn.pack(side="left", padx=(0, 15))
        
        # === SECCIÓN ARCHIVOS ===
        archivos_frame = ctk.CTkFrame(main_frame, fg_color=self.card_color, corner_radius=12)
        archivos_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        archivos_inner = ctk.CTkFrame(archivos_frame, fg_color="transparent")
        archivos_inner.pack(fill="both", expand=True, padx=15, pady=12)
        
        # Header archivos
        arch_header = ctk.CTkFrame(archivos_inner, fg_color="transparent")
        arch_header.pack(fill="x")
        
        ctk.CTkLabel(arch_header,
                    text="ARCHIVOS",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color="#888888").pack(side="left")
        
        self.count_label = ctk.CTkLabel(arch_header,
                                        text="0 seleccionados",
                                        font=ctk.CTkFont(size=11),
                                        text_color=self.accent_color)
        self.count_label.pack(side="left", padx=(10, 0))
        
        # Botones
        btns_frame = ctk.CTkFrame(arch_header, fg_color="transparent")
        btns_frame.pack(side="right")
        
        ctk.CTkButton(btns_frame, text="📂 Carpeta",
                     command=self.elegir_carpeta,
                     width=100, height=28,
                     font=ctk.CTkFont(size=11),
                     fg_color="#3d4a5c",
                     hover_color="#4d5a6c").pack(side="left", padx=(0, 8))
        
        ctk.CTkButton(btns_frame, text="📄 Archivos",
                     command=self.elegir_archivos,
                     width=100, height=28,
                     font=ctk.CTkFont(size=11),
                     fg_color="#3d4a5c",
                     hover_color="#4d5a6c").pack(side="left")
        
        # Lista de archivos
        self.lista_text = ctk.CTkTextbox(archivos_inner,
                                         font=ctk.CTkFont(family="Consolas", size=11),
                                         fg_color="#1e2530",
                                         corner_radius=8,
                                         height=120)
        self.lista_text.pack(fill="both", expand=True, pady=(10, 0))
        self.lista_text.insert("1.0", "  Selecciona archivos o carpeta...")
        self.lista_text.configure(state="disabled")
        
        # === SECCIÓN DESTINO ===
        destino_frame = ctk.CTkFrame(main_frame, fg_color=self.card_color, corner_radius=12)
        destino_frame.pack(fill="x", pady=(0, 15))
        
        destino_inner = ctk.CTkFrame(destino_frame, fg_color="transparent")
        destino_inner.pack(fill="x", padx=15, pady=12)
        
        ctk.CTkLabel(destino_inner,
                    text="GUARDAR EN",
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color="#888888").pack(anchor="w")
        
        dest_row = ctk.CTkFrame(destino_inner, fg_color="transparent")
        dest_row.pack(fill="x", pady=(8, 0))
        
        self.label_destino = ctk.CTkLabel(dest_row,
                                         text="(selecciona archivos primero)",
                                         font=ctk.CTkFont(family="Consolas", size=11),
                                         text_color="#888888")
        self.label_destino.pack(side="left", fill="x", expand=True, anchor="w")
        
        ctk.CTkButton(dest_row, text="Cambiar",
                     command=self.elegir_destino,
                     width=80, height=28,
                     font=ctk.CTkFont(size=11),
                     fg_color=self.accent_color,
                     hover_color="#ff6b6b").pack(side="right")
        
        # === BOTÓN CONVERTIR ===
        self.btn_convertir = ctk.CTkButton(main_frame,
                                          text="🚀 CONVERTIR ARCHIVOS",
                                          command=self.convertir,
                                          height=45,
                                          font=ctk.CTkFont(size=14, weight="bold"),
                                          fg_color=self.accent_color,
                                          text_color="#ffffff",
                                          text_color_disabled="#cccccc",
                                          hover_color="#ff6b6b",
                                          state="disabled")
        self.btn_convertir.pack(fill="x")
        
        # Barra de progreso (oculta)
        self.progress = ctk.CTkProgressBar(main_frame, mode="indeterminate",
                                           progress_color=self.accent_color)
    
    def _sync_formato_destino(self, *args):
        """Sincroniza formato destino: si origen es CSV -> destino XLSX y viceversa"""
        origen = self.formato_origen.get()
        if origen == "csv":
            self.formato_destino.set("xlsx")
        else:
            self.formato_destino.set("csv")
    
    def elegir_carpeta(self):
        ext = self.formato_origen.get()
        carpeta = filedialog.askdirectory(
            title=f"Seleccione carpeta con archivos {ext.upper()}",
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        
        if carpeta:
            archivos = [f for f in os.listdir(carpeta) if f.endswith(f'.{ext}')]
            if not archivos:
                messagebox.showwarning("Sin archivos", 
                                      f"No se encontraron archivos .{ext}")
                return
            self.archivos_seleccionados = [(carpeta, f) for f in archivos]
            # Establecer carpeta de destino como la carpeta seleccionada
            if not self.carpeta_destino or self.carpeta_destino != carpeta:
                self.carpeta_destino = carpeta
                self.label_destino.configure(text=carpeta, text_color="#cccccc")
            self.actualizar_lista()
    
    def elegir_archivos(self):
        ext = self.formato_origen.get()
        archivos = filedialog.askopenfilenames(
            title=f"Seleccione archivo(s) {ext.upper()}",
            filetypes=[(f"{ext.upper()}", f"*.{ext}"), ("Todos", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        
        if archivos:
            self.archivos_seleccionados = [
                (os.path.dirname(a), os.path.basename(a)) for a in archivos
            ]
            # Establecer carpeta de destino como la carpeta del primer archivo
            primera_carpeta = os.path.dirname(archivos[0])
            if not self.carpeta_destino or self.carpeta_destino != primera_carpeta:
                self.carpeta_destino = primera_carpeta
                self.label_destino.configure(text=primera_carpeta, text_color="#cccccc")
            self.actualizar_lista()
    
    def elegir_destino(self):
        carpeta = filedialog.askdirectory(title="Carpeta de destino")
        if carpeta:
            self.carpeta_destino = carpeta
            self.label_destino.configure(text=carpeta)
    
    def actualizar_lista(self):
        self.lista_text.configure(state="normal")
        self.lista_text.delete("1.0", "end")
        
        for carpeta, archivo in self.archivos_seleccionados:
            ruta = os.path.join(carpeta, archivo)
            try:
                size = os.path.getsize(ruta) / 1024 / 1024
                self.lista_text.insert("end", f"  {archivo}  •  {size:.2f} MB\n")
            except:
                self.lista_text.insert("end", f"  {archivo}\n")
        
        self.lista_text.configure(state="disabled")
        self.count_label.configure(text=f"{len(self.archivos_seleccionados)} seleccionados")
        self.btn_convertir.configure(state="normal")
    
    def convertir(self):
        if not self.archivos_seleccionados:
            messagebox.showwarning("Sin archivos", "No hay archivos seleccionados.")
            return
        
        self.btn_convertir.configure(state="disabled")
        self.progress.pack(fill="x", pady=(10, 0))
        self.progress.start()
        
        thread = threading.Thread(target=self._convertir_archivos)
        thread.start()
    
    def _convertir_archivos(self):
        os.makedirs(self.carpeta_destino, exist_ok=True)
        
        fmt_origen = self.formato_origen.get()
        fmt_destino = self.formato_destino.get()
        
        exitosos = 0
        errores = []
        
        for carpeta, archivo in self.archivos_seleccionados:
            ruta_origen = os.path.join(carpeta, archivo)
            nombre = os.path.splitext(archivo)[0]
            ruta_destino = os.path.join(self.carpeta_destino, f"{nombre}.{fmt_destino}")
            
            try:
                if fmt_origen == 'csv':
                    df = pd.read_csv(ruta_origen)
                else:
                    df = pd.read_excel(ruta_origen)
                
                if fmt_destino == 'csv':
                    df.to_csv(ruta_destino, index=False)
                else:
                    df.to_excel(ruta_destino, index=False, engine='openpyxl')
                
                exitosos += 1
            except Exception as e:
                errores.append(f"{archivo}: {str(e)[:40]}")
        
        self.progress.stop()
        self.progress.pack_forget()
        self.btn_convertir.configure(state="normal")
        
        if errores:
            msg = f"⚠️ Con errores\n\n✅ {exitosos} exitosos\n❌ {len(errores)} fallidos"
            self.after(0, lambda: messagebox.showwarning("Resultado", msg))
        else:
            msg = f"🎉 ¡Éxito!\n\n✅ {exitosos} archivo(s)\n📁 {self.carpeta_destino}"
            self.after(0, lambda: messagebox.showinfo("Completado", msg))
        
        if exitosos > 0:
            self.after(0, self._preguntar_abrir)
    
    def _preguntar_abrir(self):
        if messagebox.askyesno("Abrir", "¿Abrir carpeta de destino?"):
            os.startfile(self.carpeta_destino)


def main():
    try:
        import openpyxl
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'openpyxl'])
    
    app = ConversorUniversal()
    app.mainloop()


if __name__ == "__main__":
    main()
