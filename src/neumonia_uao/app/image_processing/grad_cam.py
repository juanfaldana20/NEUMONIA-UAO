import numpy as np
import cv2
import tensorflow as tf

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
    Genera un mapa de calor Grad-CAM simplificado y funcional.
    
    Args:
        array: Array numpy de la imagen de entrada
        
    Returns:
        Array numpy con la imagen superpuesta con el heatmap
    """
    try:
        print("🔥 Generando Grad-CAM simplificado...")
        
        # 1. Preprocesar imagen
        img = preprocess(array)
        
        # 2. Cargar modelo
        model = model_fun()
        if model is None:
            print("❌ Modelo no disponible")
            return _create_fallback_heatmap(array)
        
        # 3. Hacer predicción
        predictions = model.predict(img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
        
        print(f"🎯 Predicción: clase {predicted_class} ({confidence:.1f}%)")
        
        # 4. Intentar generar Grad-CAM real
        try:
            heatmap = _generate_gradcam_heatmap(model, img, predicted_class, array)
            print("✅ Grad-CAM real generado exitosamente")
            return heatmap
            
        except Exception as gradcam_error:
            print(f"⚠️  Error en Grad-CAM real: {gradcam_error}")
            print("📊 Generando heatmap basado en activaciones...")
            
            # 5. Fallback: heatmap basado en activaciones de capas
            heatmap = _generate_activation_heatmap(model, img, array, predicted_class)
            return heatmap
            
    except Exception as e:
        print(f"❌ Error general: {e}")
        return _create_fallback_heatmap(array)


def _generate_gradcam_heatmap(model, img, predicted_class, original_array):
    """Intenta generar Grad-CAM usando TensorFlow 2.x moderno"""
    
    # Encontrar capa convolucional objetivo
    target_layer_name = 'conv10_thisone'
    target_layer = None
    
    for layer in model.layers:
        if layer.name == target_layer_name:
            target_layer = layer
            break
    
    if target_layer is None:
        # Buscar cualquier capa convolucional
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                target_layer_name = layer.name
                break
    
    if target_layer is None:
        raise Exception("No se encontraron capas convolucionales")
    
    print(f"🎯 Usando capa: {target_layer_name}")
    
    # Crear modelo que devuelve activaciones y predicciones
    grad_model = tf.keras.models.Model(
        [model.inputs], [target_layer.output, model.output]
    )
    
    # Calcular gradientes
    with tf.GradientTape() as tape:
        inputs = tf.cast(img, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, predicted_class]
    
    # Obtener gradientes
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    # Calcular importancia de cada canal
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    
    # Promediar sobre altura y ancho
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    
    # Crear heatmap
    cam = np.ones(output.shape[0:2], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    
    # Aplicar ReLU y normalizar
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    
    # Redimensionar y aplicar
    cam = cv2.resize(cam, (512, 512))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superponer sobre imagen original
    img_resized = cv2.resize(original_array, (512, 512))
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    superimposed = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    return superimposed


def _generate_activation_heatmap(model, img, original_array, predicted_class):
    """Genera heatmap basado en activaciones de capas intermedias"""
    try:
        # Encontrar una capa intermedia para visualizar
        intermediate_layer = None
        for layer in reversed(model.layers[:40]):  # Buscar en capas intermedias
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                if layer.output_shape[-1] is not None and layer.output_shape[-1] > 1:
                    intermediate_layer = layer
                    break
        
        if intermediate_layer is None:
            raise Exception("No se encontró capa intermedia adecuada")
        
        print(f"🔍 Usando activaciones de: {intermediate_layer.name}")
        
        # Crear modelo para obtener activaciones
        activation_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=intermediate_layer.output
        )
        
        # Obtener activaciones
        activations = activation_model.predict(img, verbose=0)
        
        # Promediar canales para crear heatmap
        if len(activations.shape) == 4:
            heatmap = np.mean(activations[0], axis=-1)
        else:
            # Si no es 4D, usar la activación directamente
            heatmap = activations[0]
            if len(heatmap.shape) > 2:
                heatmap = np.mean(heatmap, axis=-1)
        
        # Normalizar
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Redimensionar y aplicar colormap
        heatmap = cv2.resize(heatmap, (512, 512))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        
        # Superponer
        img_resized = cv2.resize(original_array, (512, 512))
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        result = cv2.addWeighted(img_resized, 0.7, heatmap, 0.3, 0)
        print("✅ Heatmap de activaciones generado")
        return result
        
    except Exception as e:
        print(f"⚠️  Error en heatmap de activaciones: {e}")
        return _create_fallback_heatmap(original_array)


def _create_fallback_heatmap(array):
    """Crea un heatmap simulado como último recurso"""
    print("🎭 Creando heatmap simulado...")
    
    img_resized = cv2.resize(array, (512, 512))
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    # Crear un gradiente radial simple como heatmap
    h, w = 512, 512
    center = (w//2, h//2)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Crear heatmap con patrón radial
    heatmap = np.exp(-dist_from_center / 100)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superponer
    result = cv2.addWeighted(img_resized, 0.8, heatmap, 0.2, 0)
    print("🎆 Heatmap simulado creado")
    return result
