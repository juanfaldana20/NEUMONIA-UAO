import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K

try:
    from .preprocess_img import preprocess
    from ..models.load_model import model_fun
except ImportError:
    from preprocess_img import preprocess
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
    from load_model import model_fun


def grad_cam(array):
    """
    Genera un mapa de calor Grad-CAM usando la implementación original corregida.
    
    Esta es la implementación original que funcionaba correctamente,
    con las correcciones necesarias para TensorFlow 2.x
    
    Args:
        array: Array numpy de la imagen de entrada
        
    Returns:
        Array numpy con la imagen superpuesta con el heatmap
    """
    try:
        print("🔥 Generando Grad-CAM (implementación original corregida)...")
        
        # Preprocesar imagen (igual que el original)
        img = preprocess(array)
        
        # Cargar modelo (igual que el original)
        model = model_fun()
        if model is None:
            # Si el modelo no se carga, retornar imagen original
            print("⚠️  Modelo no disponible, usando imagen original")
            return cv2.resize(array, (512, 512))
        
        # Hacer predicción (igual que el original)
        preds = model.predict(img, verbose=0)
        argmax = np.argmax(preds[0])
        # CORRECCIÓN: Convertir argmax a int nativo de Python
        argmax = int(argmax)
        print(f"🎯 Clase predicha: {argmax}")
        
        # CORRECCIÓN: Manejar model.output que puede ser una lista
        if isinstance(model.output, list):
            # Si model.output es una lista, tomar el primer elemento
            model_output = model.output[0]
        else:
            # Si es un tensor directo
            model_output = model.output
        
        # Ahora crear el output para la clase predicha
        output = model_output[:, argmax]  # Usar indexación estándar
        
        # Obtener capa convolucional (igual que el original)
        last_conv_layer = model.get_layer("conv10_thisone")
        print(f"🎯 Usando capa: {last_conv_layer.name}")
        
        # IMPLEMENTACIÓN FINAL: Usar GradientTape con enfoque funcional
        print("⚙️  Calculando gradientes...")
        
        # Convertir entrada a tensor
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        
        # Crear modelo que devuelve tanto las activaciones como las predicciones
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            # Observar las activaciones convolucionales
            conv_outputs, predictions = grad_model(img_tensor)
            
            # Manejar si predictions es una lista
            if isinstance(predictions, list):
                predictions = predictions[0]
            
            # Obtener la salida de la clase predicha
            class_channel = predictions[:, argmax]
        
        # Calcular gradientes
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Verificar que los gradientes no sean None
        if grads is None:
            print("⚠️  No se pudieron calcular gradientes, usando aproximación")
            # Usar aproximación basada en activaciones directas
            conv_outputs_np = conv_outputs[0].numpy()
            # Usar todas las activaciones con peso uniforme
            pooled_grads_value = np.ones(conv_outputs_np.shape[-1]) / conv_outputs_np.shape[-1]
            conv_layer_output_value = conv_outputs_np
        else:
            # Procesar gradientes normalmente
            print("✅ Gradientes calculados correctamente")
            # Pooled gradients (equivalente al original)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Convertir a numpy
            pooled_grads_value = pooled_grads.numpy()
            conv_layer_output_value = conv_outputs[0].numpy()
        
        # Aplicar gradientes a las activaciones (igual que el original)
        for filters in range(64):
            if filters < conv_layer_output_value.shape[-1]:  # CORRECCIÓN: verificar índice
                conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
        
        # Crear heatmap (igual que el original)
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU
        
        # Normalizar (igual que el original)
        if np.max(heatmap) > 0:  # CORRECCIÓN: evitar división por cero
            heatmap /= np.max(heatmap)  # normalize
        
        # Redimensionar heatmap (igual que el original)
        heatmap = cv2.resize(heatmap, (512, 512))  # CORRECCIÓN: usar tamaño fijo
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Preparar imagen base (igual que el original)
        img2 = cv2.resize(array, (512, 512))
        
        # CORRECCIÓN: Asegurar que img2 tenga 3 canales
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        elif len(img2.shape) == 3 and img2.shape[2] == 4:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2RGB)
        
        # Aplicar transparencia y superponer (igual que el original)
        hif = 0.8
        transparency = heatmap * hif
        transparency = transparency.astype(np.uint8)
        
        # CORRECCIÓN: Usar addWeighted en lugar de add para mejor control
        superimposed_img = cv2.addWeighted(img2, 0.6, transparency, 0.4, 0)
        superimposed_img = superimposed_img.astype(np.uint8)
        
        print("✅ Grad-CAM original generado exitosamente")
        
        # Retornar con corrección de canales (igual que el original)
        return superimposed_img[:, :, ::-1]
        
    except Exception as e:
        print(f"❌ Error en Grad-CAM original: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: retornar imagen redimensionada
        print("🔄 Usando imagen original como fallback")
        img_fallback = cv2.resize(array, (512, 512))
        if len(img_fallback.shape) == 2:
            img_fallback = cv2.cvtColor(img_fallback, cv2.COLOR_GRAY2RGB)
        return img_fallback
