import cv2
import numpy as np
import pydicom as dicom
from PIL import Image

def read_image(path):
    """
    Lee una imagen desde la ruta especificada y la devuelve en escala de grises.
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen en la ruta: {path}")
    return image

def read_dicom_file(path):
    """
    Lee un archivo DICOM y lo convierte para su uso en el modelo.
    
    Args:
        path: Ruta al archivo DICOM
        
    Returns:
        tuple: (img_RGB, img2show) - imagen procesada y imagen para mostrar
    """
    img = dicom.read_file(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show

def read_jpg_file(path):
    """
    Lee un archivo JPG/JPEG/PNG y lo convierte para su uso en el modelo.
    
    Args:
        path: Ruta al archivo de imagen
        
    Returns:
        tuple: (img2, img2show) - imagen procesada y imagen para mostrar
    """
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
