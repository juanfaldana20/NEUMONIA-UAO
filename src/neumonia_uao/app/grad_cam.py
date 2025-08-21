import numpy as np
import cv2
from tensorflow.keras import backend as K
from .preprocess_img import preprocess
from .load_model import model_fun

def grad_cam(array):
    """
    Genera un mapa de calor Grad-CAM sobre la imagen de entrada.
    """
    img = preprocess(array)
    model = model_fun()
    if model is None:
        return cv2.resize(array, (512, 512))
    
    preds = model.predict(img)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    
    # Intenta encontrar la capa si el nombre es 'conv10_thisone'
    try:
        last_conv_layer = model.get_layer("conv10_thisone")
    except ValueError:
        print("Capa 'conv10_thisone' no encontrada, usando la última capa conv.")
        # Método alternativo: encontrar la última capa de convolución automáticamente
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer = layer
                break
        if not 'last_conv_layer' in locals():
            print("No se encontró ninguna capa de convolución.")
            return cv2.resize(array, (512, 512))
    
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img)
    
    for filters in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
        
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    # Evita la división por cero si el heatmap es todo ceros
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.addWeighted(transparency, 0.5, img2, 0.5, 0) # Usar addWeighted para mejor control
    superimposed_img = superimposed_img.astype(np.uint8)
    
    return superimposed_img[:, :, ::-1]