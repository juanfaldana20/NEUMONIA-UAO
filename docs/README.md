## NEUMONIA-UAO - Detector de Neumonía con Deep Learning

Sistema de apoyo al diagnóstico médico para la detección automática de neumonía en radiografías de tórax utilizando Deep Learning y técnicas
de explicabilidad con Grad-CAM.


## Este proyecto implementa una herramienta de inteligencia artificial que integra visión computacional y aprendizaje automático (machine learning). Se busca la detección rápida de neumonía a partir de imágenes radiográficas de tórax en formato DICOM o JPG/PNG. La herramienta clasifica imágenes en tres categorías:

1.Neumonía bacteriana

2.Neumonía viral

3.Sin neumonía (normal)

Se implementa la técnica Grad-CAM (Gradient-weighted Class Activation Mapping) para generar mapas de calor que destacan las regiones de la imagen más relevantes para la clasificación, proporcionando explicabilidad al modelo. Así, una persona sin conocimientos médicos puede comprender mejor el resultado de la radiografía.

---

## Uso de la herramienta:

A continuación le explicaremos cómo empezar a utilizarla.

Requerimientos necesarios para el funcionamiento:

-compatilidad con python <= 3.10.x  

- Instale Anaconda para Windows siguiendo las siguientes instrucciones:
  https://docs.anaconda.com/anaconda/install/windows/

- Abra Anaconda Prompt y ejecute las siguientes instrucciones:

  conda create -n tf tensorflow

  conda activate tf

  cd UAO-Neumonia

  pip install -r requirements.txt

  python detector_neumonia.py

---

## Uso de la Interfaz Gráfica:

- Ingrese la cédula del paciente en la caja de texto
- Presione el botón 'Cargar Imagen', seleccione la imagen del explorador de archivos del computador (Imagenes de prueba en https://drive.google.com/drive/folders/1WOuL0wdVC6aojy8IfssHcqZ4Up14dy0g?usp=drive_link)
- Presione el botón 'Predecir' y espere unos segundos hasta que observe los resultados
- Presione el botón 'Guardar' para almacenar la información del paciente en un archivo excel con extensión .csv
- Presione el botón 'PDF' para descargar un archivo PDF con la información desplegada en la interfaz
- Presión el botón 'Borrar' si desea cargar una nueva imagen

---

## Arquitectura de archivos propuesta.
app.py

Contiene el diseño de la interfaz gráfica utilizando Tkinter para la detección de neumonía.

Los botones de la interfaz (Predecir, Cargar Imagen, Borrar, PDF, Guardar) llaman métodos contenidos en otros scripts modulares. 
Implementa una clase App que gestiona toda la lógica de la interfaz de usuario y la interacción con los módulos de procesamiento.

## integrator.py

Es un módulo que integra los demás scripts y retorna solamente lo necesario para ser visualizado en la interfaz gráfica.

Retorna la clase de neumonía (bacteriana, normal, viral), la probabilidad de confianza y una imagen con el mapa de calor generado
por Grad-CAM. Funciona como el coordinador principal que orquesta el flujo completo de procesamiento.

## read_img.py

Script que lee imágenes médicas en múltiples formatos (DICOM .dcm, JPEG, PNG) para visualizarlas en la interfaz gráfica.

Contiene funciones especializadas: read_dicom_file() para archivos médicos DICOM y read_jpg_file() para imágenes estándar. 
Además, convierte las imágenes a arreglos NumPy normalizados para su posterior preprocesamiento, asegurando compatibilidad entre diferentes formatos de entrada.

## preprocess_img.py

Script que recibe el arreglo proveniente de read_img.py y realiza las siguientes transformaciones optimizadas para el modelo de CNN:

•  Resize a 512x512 píxeles (tamaño estándar del modelo)
•  Conversión a escala de grises para reducir dimensionalidad
•  Ecualización del histograma con CLAHE (Contrast Limited Adaptive Histogram Equalization)
•  Normalización de la imagen entre 0 y 1 para estabilidad numérica
•  Conversión del arreglo de imagen a formato de batch (tensor) compatible con TensorFlow

## load_model.py

Script que gestiona la carga del modelo de red neuronal convolucional previamente entrenado conv_MLP_84.h5.

Implementa un sistema de caché global para evitar recargar el modelo múltiples veces, mejorando significativamente el rendimiento. 
Incluye configuración específica de TensorFlow para compatibilidad con Grad-CAM y manejo robusto de errores en caso de que el archivo del modelo no esté disponible.

## grad_cam.py

Script que implementa la técnica Grad-CAM para generar mapas de calor explicativos.

Recibe la imagen preprocesada, carga el modelo, obtiene la predicción y utiliza la capa convolucional conv10_thisone para calcular los gradientes.
Genera un mapa de calor superpuesto que resalta las regiones pulmonares más relevantes para la clasificación, proporcionando explicabilidad visual al
diagnóstico automático.

---
## Proyecto realizado por:



Daniel Santiago Muñoz Jimenez - https://github.com/danielmunozji
Juan Fernando Aldana - https://github.com/juanfaldana20
Juan Camilo Leon - https://github.com/juanca23meca
Alejandro Bolaños - https://github.com/AlejandroBoPe



---

## Proyecto original realizado por:

Isabella Torres Revelo - https://github.com/isa-tr
Nicolas Diaz Salazar - https://github.com/nicolasdiazsalazar
