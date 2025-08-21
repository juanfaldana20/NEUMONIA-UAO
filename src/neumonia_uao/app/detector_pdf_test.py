#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Versión de prueba para probar específicamente la funcionalidad de PDF
con datos simulados de predicción.
"""

from tkinter import *
from tkinter import ttk, font, filedialog, Entry
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image
import csv
import pyautogui
import tkcap
import img2pdf
import numpy as np
import time
import cv2
import os


def read_jpg_file(path):
    """Lee archivo JPG/PNG"""
    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2show = Image.fromarray(img_rgb)
        
        img2 = img.astype(float)
        img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
        img2 = np.uint8(img2)
        
        return img2, img2show
    except Exception as e:
        raise ValueError(f"Error leyendo imagen: {e}")


def simulate_prediction(array):
    """Simula una predicción exitosa para pruebas"""
    # Simular resultados
    label = "viral"
    proba = 87.5
    
    # Crear un mapa de calor simulado
    h, w = array.shape[:2] if len(array.shape) > 1 else (512, 512)
    
    # Redimensionar a 512x512 para consistencia
    img_resized = cv2.resize(array, (512, 512))
    
    # Crear mapa de calor simulado (gradiente circular)
    center = (256, 256)
    Y, X = np.ogrid[:512, :512]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Crear patrón de mapa de calor
    heatmap = np.exp(-(dist_from_center / 100)**2) * 255
    heatmap = heatmap.astype(np.uint8)
    
    # Aplicar colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superponer sobre la imagen original
    alpha = 0.6
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    superimposed = cv2.addWeighted(img_resized, 1 - alpha, heatmap_colored, alpha, 0)
    
    return (label, proba, superimposed)


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Detector de Neumonía - PRUEBA PDF")

        # BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA - PRUEBA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        # STRING VARIABLES
        self.ID = StringVar()
        self.result = StringVar()

        # INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        # IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        # BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir (SIMULADO)", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        # WIDGET POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=60, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=190, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        # FOCUS
        self.text1.focus_set()

        # VARIABLES
        self.array = None
        self.reportID = 0

        # RUN LOOP
        self.root.mainloop()

    def load_img_file(self):
        """Carga imagen desde archivo"""
        filepath = filedialog.askopenfilename(
            initialdir="C:/",
            title="Select image",
            filetypes=(
                ("All Images", "*.jpg;*.jpeg;*.png"),
                ("JPEG", "*.jpeg"),
                ("JPG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*"),
            ),
        )
        if filepath:
            print(f"Archivo seleccionado: {filepath}")
            try:
                print("Cargando imagen...")
                self.array, img2show = read_jpg_file(filepath)
                
                print(f"Imagen cargada: {img2show.size}")
                
                self.img1 = img2show.resize((250, 250), Image.LANCZOS)
                self.img1 = ImageTk.PhotoImage(self.img1)
                
                self.text_img1.delete(1.0, END)
                self.text_img1.image_create(END, image=self.img1)
                self.text_img1.insert(END, "\n")
                
                print("Imagen mostrada correctamente")
                self.button1["state"] = "enabled"
                
            except Exception as e:
                print(f"Error al cargar imagen: {e}")
                showinfo(title="Error", message=f"No se pudo cargar la imagen: {e}")

    def run_model(self):
        """Ejecuta predicción simulada"""
        try:
            print("Ejecutando predicción simulada...")
            self.label, self.proba, self.heatmap = simulate_prediction(self.array)
            
            self.img2 = Image.fromarray(self.heatmap.astype(np.uint8))
            self.img2 = self.img2.resize((250, 250), Image.LANCZOS)
            self.img2 = ImageTk.PhotoImage(self.img2)
            
            # Limpiar resultados anteriores
            self.text_img2.delete(1.0, END)
            self.text2.delete(1.0, END)
            self.text3.delete(1.0, END)
            
            self.text_img2.image_create(END, image=self.img2)
            self.text2.insert(END, self.label)
            self.text3.insert(END, "{:.2f}".format(self.proba) + "%")
            
            print(f"Predicción simulada completada: {self.label}, {self.proba:.2f}%")
            
        except Exception as e:
            print(f"Error en run_model: {e}")
            showinfo(title="Error", message=f"Error en la predicción: {e}")

    def save_results_csv(self):
        """Guarda resultados en CSV"""
        if not hasattr(self, 'label') or not hasattr(self, 'proba'):
            showinfo(title="Error", message="Primero debe realizar una predicción.")
            return
        
        try:
            with open("historial.csv", "a", encoding='utf-8') as csvfile:
                w = csv.writer(csvfile, delimiter="-")
                w.writerow([
                    self.text1.get(), 
                    self.label, 
                    "{:.2f}".format(self.proba) + "%"
                ])
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")
        except Exception as e:
            showinfo(title="Error", message=f"Error guardando CSV: {e}")

    def create_pdf(self):
        """Genera un PDF con la captura de pantalla de la interfaz."""
        try:
            # Verificar si hay datos para generar el reporte
            if not hasattr(self, 'label') or not hasattr(self, 'proba'):
                showinfo(title="Error", message="Primero debe realizar una predicción antes de generar el PDF.")
                return
            
            # Crear nombres de archivos únicos
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            temp_img_name = f"temp_reporte_{timestamp}.jpg"
            pdf_name = f"Reporte_Neumonia_PRUEBA_{timestamp}.pdf"
            
            print(f"Generando PDF: {pdf_name}")
            
            # Método 1: Usando tkcap
            try:
                cap = tkcap.CAP(self.root)
                cap.capture(temp_img_name)
                
                if os.path.exists(temp_img_name):
                    img = Image.open(temp_img_name)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(pdf_name, "PDF", resolution=100.0, save_all=True)
                    
                    # Limpiar archivo temporal
                    try:
                        os.remove(temp_img_name)
                    except:
                        pass
                    
                    showinfo(
                        title="PDF Generado", 
                        message=f"El PDF fue generado con éxito como:\n{pdf_name}\n\nUbicación: {os.path.abspath(pdf_name)}"
                    )
                    
                    print(f"PDF generado exitosamente: {pdf_name}")
                    self.reportID += 1
                    return
            except Exception as e:
                print(f"Error con tkcap: {e}")
            
            # Método 2: Usando pyautogui + img2pdf
            try:
                screenshot_name = f"screenshot_{timestamp}.png"
                
                print("Intentando método alternativo...")
                self.root.update()  # Asegurar que la ventana esté actualizada
                x = self.root.winfo_rootx()
                y = self.root.winfo_rooty()
                w = self.root.winfo_width()
                h = self.root.winfo_height()
                
                screenshot = pyautogui.screenshot(region=(x, y, w, h))
                screenshot.save(screenshot_name)
                
                with open(pdf_name, "wb") as f:
                    f.write(img2pdf.convert(screenshot_name))
                
                try:
                    os.remove(screenshot_name)
                except:
                    pass
                
                showinfo(
                    title="PDF Generado (Método Alternativo)",
                    message=f"El PDF fue generado con éxito:\n{pdf_name}"
                )
                
                print(f"PDF generado con método alternativo: {pdf_name}")
                self.reportID += 1
                
            except Exception as e:
                print(f"Error en método alternativo: {e}")
                showinfo(
                    title="Error",
                    message=f"No se pudo generar el PDF. Error: {e}"
                )
            
        except Exception as e:
            print(f"Error general en create_pdf: {e}")
            showinfo(title="Error", message=f"Error general: {e}")

    def delete(self):
        """Borra todos los datos"""
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(1.0, "end")
            self.text_img2.delete(1.0, "end")
            
            # Resetear variables
            self.array = None
            if hasattr(self, 'img1'):
                del self.img1
            if hasattr(self, 'img2'):
                del self.img2
            if hasattr(self, 'label'):
                del self.label
            if hasattr(self, 'proba'):
                del self.proba
            
            self.button1["state"] = "disabled"
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main():
    my_app = App()
    return 0


if __name__ == "__main__":
    main()
