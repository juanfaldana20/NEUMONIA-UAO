import cv2

def read_image(path):
    """
    Lee una imagen desde la ruta especificada y la devuelve en escala de grises.
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen en la ruta: {path}")
    return image
