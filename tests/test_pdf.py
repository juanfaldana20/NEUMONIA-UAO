#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pdf.py
Programa simple para probar la funcionalidad de generación de PDF
"""

from tkinter import *
from tkinter import ttk, font
from tkinter.messagebox import showinfo
from PIL import Image
import tkcap
import img2pdf
import pyautogui
import time
import os


class TestPDFApp:
    def __init__(self):
        self.root = Tk()
        self.root.title("Prueba de Generación PDF")
        self.root.geometry("400x300")
        
        # Crear algunos elementos de interfaz para hacer la captura interesante
        fonti = font.Font(weight="bold")
        
        Label(self.root, text="PRUEBA DE PDF", font=fonti).pack(pady=20)
        Label(self.root, text="Esta es una ventana de prueba").pack(pady=10)
        Label(self.root, text="Resultado: Normal").pack(pady=5)
        Label(self.root, text="Probabilidad: 85.3%").pack(pady=5)
        
        # Simular datos de predicción
        self.label = "Normal"
        self.proba = 85.3
        
        # Botón para generar PDF
        btn_pdf = Button(self.root, text="Generar PDF", command=self.create_pdf, 
                        bg="lightblue", font=fonti)
        btn_pdf.pack(pady=20)
        
        Label(self.root, text="Haz clic en 'Generar PDF' para probar").pack(pady=10)
        
        self.root.mainloop()
    
    def create_pdf(self):
        """Genera un PDF con la captura de pantalla de la interfaz."""
        try:
            # Crear nombres de archivos únicos
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            temp_img_name = f"temp_reporte_{timestamp}.jpg"
            pdf_name = f"Reporte_Test_{timestamp}.pdf"
            
            print(f"Generando PDF: {pdf_name}")
            
            # Método 1: Usando tkcap
            try:
                print("Intentando con tkcap...")
                cap = tkcap.CAP(self.root)
                cap.capture(temp_img_name)
                
                # Verificar que la captura se realizó
                if os.path.exists(temp_img_name):
                    img = Image.open(temp_img_name)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(pdf_name, "PDF", resolution=100.0)
                    
                    # Limpiar archivo temporal
                    try:
                        os.remove(temp_img_name)
                    except:
                        pass
                    
                    showinfo("Éxito", f"PDF generado con tkcap:\\n{pdf_name}")
                    print(f"PDF creado exitosamente: {pdf_name}")
                    return
                
            except Exception as e:
                print(f"Error con tkcap: {e}")
            
            # Método 2: Usando pyautogui + img2pdf
            try:
                print("Intentando con pyautogui...")
                screenshot_name = f"screenshot_{timestamp}.png"
                
                # Obtener posición y tamaño de la ventana
                self.root.update()  # Asegurar que la ventana esté actualizada
                x = self.root.winfo_rootx()
                y = self.root.winfo_rooty()
                w = self.root.winfo_width()
                h = self.root.winfo_height()
                
                print(f"Capturando región: x={x}, y={y}, w={w}, h={h}")
                
                # Capturar la región de la ventana
                screenshot = pyautogui.screenshot(region=(x, y, w, h))
                screenshot.save(screenshot_name)
                
                # Convertir a PDF usando img2pdf
                with open(pdf_name, "wb") as f:
                    f.write(img2pdf.convert(screenshot_name))
                
                # Limpiar archivo temporal
                try:
                    os.remove(screenshot_name)
                except:
                    pass
                
                showinfo("Éxito (Método Alternativo)", 
                        f"PDF generado con pyautogui + img2pdf:\\n{pdf_name}")
                print(f"PDF creado con método alternativo: {pdf_name}")
                return
                
            except Exception as e:
                print(f"Error con pyautogui: {e}")
            
            # Si ambos métodos fallan
            showinfo("Error", "No se pudo generar el PDF con ningún método")
            
        except Exception as e:
            print(f"Error general: {e}")
            showinfo("Error", f"Error general: {e}")


if __name__ == "__main__":
    TestPDFApp()
