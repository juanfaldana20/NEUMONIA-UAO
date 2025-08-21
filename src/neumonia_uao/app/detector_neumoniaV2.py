#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk, font, filedialog, Entry

from tkinter.messagebox import askokcancel, showinfo, WARNING
import getpass
from PIL import ImageTk, Image
import csv
import pyautogui
import tkcap
import img2pdf
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import backend as K
import pydicom as dicom
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
import cv2


def model_fun():
    """
    Función para cargar el modelo entrenado de neumonía.
    """
    try:
        # Intentar cargar con compile=False para evitar problemas de compatibilidad
        model = tf.keras.models.load_model('conv_MLP_84.h5', compile=False)
        # Recompilar el modelo si es necesario
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Modelo cargado correctamente")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegúrate de que el archivo conv_MLP_84.h5 existe en el directorio actual")
        print("Si el problema persiste, podrías necesitar reentrenar el modelo con la versión actual de TensorFlow")
        return None


def grad_cam(array):
    img = preprocess(array)
    model = model_fun()
    if model is None:
        # Si el modelo no se carga, retornar imagen original
        return cv2.resize(array, (512, 512))
    preds = model.predict(img)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    last_conv_layer = model.get_layer("conv10_thisone")
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img)
    for filters in range(64):
        conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
    # creating the heatmap
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # normalize
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img[:, :, ::-1]


def predict(array):
    #   1. call function to pre-process image: it returns image in batch format
    batch_array_img = preprocess(array)
    #   2. call function to load model and predict: it returns predicted class and probability
    model = model_fun()
    if model is None:
        return ("Error", 0.0, cv2.resize(array, (512, 512)))
    # model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
    predictions = model.predict(batch_array_img)
    prediction = np.argmax(predictions)
    proba = np.max(predictions) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    #   3. call function to generate Grad-CAM: it returns an image with a superimposed heatmap
    heatmap = grad_cam(array)
    return (label, proba, heatmap)


def read_dicom_file(path):
    img = dicom.read_file(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show


def read_jpg_file(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")
    
    # Convertir de BGR a RGB para PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2show = Image.fromarray(img_rgb)
    
    # Procesar imagen para el modelo
    img2 = img.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    
    return img2, img2show


def preprocess(array):
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        #   BOLD FONT
        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        #   LABELS
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        #   TWO STRING VARIABLES TO CONTAIN ID AND RESULT
        self.ID = StringVar()
        self.result = StringVar()

        #   TWO INPUT BOXES
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)

        #   GET ID
        self.ID_content = self.text1.get()

        #   TWO IMAGE INPUT BOXES
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        #   BUTTONS
        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        #   WIDGETS POSITIONS
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        #   FOCUS ON PATIENT ID
        self.text1.focus_set()

        #  se reconoce como un elemento de la clase
        self.array = None

        #   NUMERO DE IDENTIFICACIÓN PARA GENERAR PDF
        self.reportID = 0

        #   RUN LOOP
        self.root.mainloop()

    #   METHODS
    def load_img_file(self):
        filepath = filedialog.askopenfilename(
            initialdir="C:/",  # Cambiar a C:/ para Windows
            title="Select image",
            filetypes=(
                ("All Images", "*.jpg;*.jpeg;*.png;*.dcm"),  # Opción para todos los tipos
                ("JPEG", "*.jpeg"),
                ("JPG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("DICOM", "*.dcm"),
                ("All files", "*.*"),  # Opción para ver todos los archivos
            ),
        )
        if filepath:
            print(f"Archivo seleccionado: {filepath}")  # Debug
            try:
                # Determinar el tipo de archivo y usar la función correcta
                if filepath.lower().endswith('.dcm'):
                    print("Cargando como DICOM...")  # Debug
                    self.array, img2show = read_dicom_file(filepath)
                else:
                    print("Cargando como JPG/JPEG/PNG...")  # Debug
                    self.array, img2show = read_jpg_file(filepath)
                
                print(f"Imagen cargada: {img2show.size}")  # Debug
                
                # Usar LANCZOS en lugar de ANTIALIAS (deprecado)
                self.img1 = img2show.resize((250, 250), Image.LANCZOS)
                self.img1 = ImageTk.PhotoImage(self.img1)
                
                # Limpiar el widget de imagen antes de insertar nueva imagen
                self.text_img1.delete(1.0, END)
                self.text_img1.image_create(END, image=self.img1)
                self.text_img1.insert(END, "\n")  # Agregar salto de línea
                
                print("Imagen mostrada correctamente")  # Debug
                self.button1["state"] = "enabled"
                
            except Exception as e:
                print(f"Error al cargar imagen: {e}")
                showinfo(title="Error", message=f"No se pudo cargar la imagen: {e}")

    def run_model(self):
        self.label, self.proba, self.heatmap = predict(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        print("OK")
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        with open("historial.csv", "a") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            w.writerow(
                [self.text1.get(), self.label, "{:.2f}".format(self.proba) + "%"]
            )
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

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
            pdf_name = f"Reporte_Neumonia_{timestamp}.pdf"
            
            print(f"Generando PDF: {pdf_name}")
            
            # Capturar la pantalla usando tkcap
            cap = tkcap.CAP(self.root)
            img = cap.capture(temp_img_name)
            
            # Verificar que la captura se realizó correctamente
            import os
            if not os.path.exists(temp_img_name):
                raise Exception("No se pudo capturar la pantalla")
            
            # Abrir y procesar la imagen capturada
            img = Image.open(temp_img_name)
            
            # Asegurar que la imagen esté en modo RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Guardar como PDF directamente
            img.save(pdf_name, "PDF", resolution=100.0, save_all=True)
            
            # Limpiar archivo temporal
            try:
                os.remove(temp_img_name)
            except:
                pass  # No es crítico si no se puede eliminar
            
            # Incrementar ID para próximo reporte
            self.reportID += 1
            
            # Mostrar mensaje de éxito
            showinfo(
                title="PDF Generado", 
                message=f"El PDF fue generado con éxito como:\n{pdf_name}\n\nUbicación: {os.path.abspath(pdf_name)}"
            )
            
            print(f"PDF generado exitosamente: {pdf_name}")
            
        except Exception as e:
            print(f"Error generando PDF: {e}")
            # Intentar método alternativo usando img2pdf
            self._create_pdf_alternative()
    
    def _create_pdf_alternative(self):
        """Método alternativo para generar PDF usando img2pdf."""
        try:
            import pyautogui
            from tkinter import messagebox
            
            # Generar nombre único
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            screenshot_name = f"screenshot_{timestamp}.png"
            pdf_name = f"Reporte_Neumonia_{timestamp}.pdf"
            
            print("Intentando método alternativo para generar PDF...")
            
            # Tomar captura de pantalla de toda la ventana
            # Obtener posición y tamaño de la ventana
            x = self.root.winfo_rootx()
            y = self.root.winfo_rooty()
            w = self.root.winfo_width()
            h = self.root.winfo_height()
            
            # Capturar la región de la ventana
            screenshot = pyautogui.screenshot(region=(x, y, w, h))
            screenshot.save(screenshot_name)
            
            # Convertir imagen a PDF usando img2pdf
            with open(pdf_name, "wb") as f:
                f.write(img2pdf.convert(screenshot_name))
            
            # Limpiar archivo temporal
            import os
            try:
                os.remove(screenshot_name)
            except:
                pass
            
            self.reportID += 1
            
            showinfo(
                title="PDF Generado (Método Alternativo)",
                message=f"El PDF fue generado con éxito usando método alternativo:\n{pdf_name}"
            )
            
            print(f"PDF generado con método alternativo: {pdf_name}")
            
        except Exception as e:
            print(f"Error en método alternativo: {e}")
            showinfo(
                title="Error",
                message=f"No se pudo generar el PDF. Error: {e}\n\nVerifique que tenga permisos de escritura en el directorio."
            )

    def delete(self):
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(1.0, "end")
            self.text_img2.delete(1.0, "end")
            # Resetear variables de imagen
            self.array = None
            if hasattr(self, 'img1'):
                del self.img1
            if hasattr(self, 'img2'):
                del self.img2
            # Deshabilitar botón de predicción
            self.button1["state"] = "disabled"
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main():
    my_app = App()
    return 0


if __name__ == "__main__":
    main()
