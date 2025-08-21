import cv2
import numpy as np

def preprocess(array):
    """
    Preprocesa la imagen para el modelo de predicción.
    
    Args:
        array: Array numpy de la imagen de entrada
        
    Returns:
        Array numpy preprocesado listo para el modelo
    """
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array
