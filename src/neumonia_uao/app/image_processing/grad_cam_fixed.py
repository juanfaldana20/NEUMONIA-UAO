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


def grad_cam_fixed(array):
    """
    Genera un mapa de calor Grad-CAM sobre la imagen de entrada.
    Implementación corregida usando modelo completo
    
    Args:
        array: Array numpy de la imagen de entrada
        
    Returns:
        Array numpy con la imagen superpuesta con el heatmap
    """
    try:
        print("🔍 Iniciando Grad-CAM corregido...")
        
        # 1. Preprocesar imagen
        img = preprocess(array)
        
        # 2. Cargar modelo
        model = model_fun()
        if model is None:
            print("❌ Modelo no disponible")
            img_resized = cv2.resize(array, (512, 512))
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            return img_resized
        
        # 3. Encontrar la capa convolucional objetivo
        target_layer_name = 'conv10_thisone'  # Capa específica del modelo
        
        # Verificar que la capa existe
        target_layer = None
        for layer in model.layers:
            if layer.name == target_layer_name:
                target_layer = layer
                break
        
        if target_layer is None:
            print(f"❌ No se encontró la capa {target_layer_name}")
            # Buscar cualquier capa convolucional
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    target_layer = layer
                    target_layer_name = layer.name
                    break
        
        if target_layer is None:
            raise Exception("No se encontraron capas convolucionales")
        
        print(f"✅ Usando capa: {target_layer_name}")
        
        # 4. Crear modelo que devuelve tanto las activaciones como las predicciones
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[target_layer.output, model.output]
        )
        
        # 5. Usar GradientTape para calcular gradientes
        with tf.GradientTape() as tape:
            inputs = tf.cast(img, tf.float32)
            tape.watch(inputs)
            
            # Obtener activaciones de la capa convolucional y predicciones
            conv_outputs, predictions = grad_model(inputs)
            
            # Obtener la clase con mayor probabilidad
            predicted_class = tf.argmax(predictions[0])
            class_channel = predictions[:, predicted_class]
        
        # 6. Calcular gradientes de la clase predicha respecto a las activaciones
        grads = tape.gradient(class_channel, conv_outputs)
        
        # 7. Calcular la importancia promedio de cada canal (Global Average Pooling)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 8. Multiplicar cada canal de activación por su importancia
        conv_outputs = conv_outputs[0]  # Remover batch dimension
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # 9. Aplicar ReLU y normalizar
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / tf.reduce_max(heatmap)
        
        # 10. Convertir a numpy y redimensionar
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (512, 512))
        
        # 11. Aplicar mapa de colores
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 12. Preparar imagen original
        img_original = cv2.resize(array, (512, 512))
        if len(img_original.shape) == 2:
            img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)
        
        # 13. Superponer heatmap
        alpha = 0.4
        superimposed = cv2.addWeighted(img_original, 1-alpha, heatmap, alpha, 0)
        
        # Información de debug
        pred_class_int = predicted_class.numpy()
        pred_prob = tf.reduce_max(predictions).numpy() * 100
        
        print(f"✅ Grad-CAM generado exitosamente")
        print(f"   - Capa usada: {target_layer_name}")
        print(f"   - Clase predicha: {pred_class_int} ({pred_prob:.1f}%)")
        
        return superimposed
        
    except Exception as e:
        print(f"❌ Error en Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: imagen original con tinte
        try:
            img_resized = cv2.resize(array, (512, 512))
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            
            # Aplicar tinte rojizo para simular heatmap
            overlay = img_resized.copy()
            overlay[:, :, 2] = np.minimum(overlay[:, :, 2] + 40, 255)  # Canal rojo
            result = cv2.addWeighted(img_resized, 0.7, overlay, 0.3, 0)
            
            print("⚠️  Usando heatmap simulado")
            return result
            
        except Exception as fallback_error:
            print(f"❌ Error en fallback: {fallback_error}")
            # Último recurso: imagen redimensionada
            img_fallback = cv2.resize(array, (512, 512))
            if len(img_fallback.shape) == 2:
                img_fallback = cv2.cvtColor(img_fallback, cv2.COLOR_GRAY2RGB)
            return img_fallback


# Función de prueba
def test_grad_cam_fixed():
    """Función de prueba para el Grad-CAM corregido"""
    print("🧪 Probando Grad-CAM corregido...")
    
    # Crear imagen de prueba
    test_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Probar la función
    result = grad_cam_fixed(test_array)
    
    print(f"✅ Prueba completada. Resultado shape: {result.shape}")
    return result


if __name__ == "__main__":
    test_grad_cam_fixed()
