#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de diagnóstico para entender la estructura del modelo
y corregir el error de Grad-CAM
"""

import tensorflow as tf
import numpy as np
import os

# Configurar TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_soft_device_placement(True)

# Cambiar al directorio del modelo
os.chdir('src/neumonia_uao/app')

def analizar_modelo():
    try:
        print("=== DIAGNÓSTICO DEL MODELO ===")
        
        # Cargar el modelo
        print("1. Cargando modelo...")
        model = tf.keras.models.load_model('conv_MLP_84.h5', compile=False)
        print("✅ Modelo cargado exitosamente")
        
        # Mostrar información básica
        print(f"\n2. Información básica:")
        print(f"   - Número de capas: {len(model.layers)}")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        
        # Listar todas las capas
        print(f"\n3. Estructura de capas:")
        conv_layers = []
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__
            
            # Obtener output_shape de manera segura
            try:
                output_shape = layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'
            except:
                output_shape = 'N/A'
                
            print(f"   {i:2d}: {layer.name:25} | {layer_type:20} | Output: {output_shape}")
            
            # Identificar capas convolucionales
            if 'conv' in layer.name.lower() or 'Conv' in layer_type:
                conv_layers.append((i, layer.name, output_shape))
        
        print(f"\n4. Capas convolucionales encontradas:")
        for i, name, shape in conv_layers:
            print(f"   {i:2d}: {name:25} | Shape: {shape}")
        
        # Información sobre la última capa convolucional
        if conv_layers:
            last_conv = conv_layers[-1]
            print(f"\n5. Última capa convolucional:")
            print(f"   - Índice: {last_conv[0]}")
            print(f"   - Nombre: {last_conv[1]}")
            print(f"   - Shape: {last_conv[2]}")
            
            # Acceder a la capa
            last_conv_layer = model.layers[last_conv[0]]
            print(f"   - Tipo: {type(last_conv_layer).__name__}")
        
        # Probar una predicción simple
        print(f"\n6. Prueba de predicción:")
        test_input = np.random.random((1, 512, 512, 1)).astype(np.float32)
        print(f"   - Input shape: {test_input.shape}")
        
        try:
            prediction = model.predict(test_input, verbose=0)
            print(f"   - Predicción exitosa")
            print(f"   - Output shape: {prediction.shape}")
            print(f"   - Output sample: {prediction[0][:5]}")
        except Exception as e:
            print(f"   - Error en predicción: {e}")
        
        # Información sobre el backend de Keras
        print(f"\n7. Información del backend:")
        print(f"   - Keras backend: {tf.keras.backend.backend()}")
        print(f"   - TensorFlow version: {tf.__version__}")
        
        return model, conv_layers
        
    except Exception as e:
        print(f"❌ Error en diagnóstico: {e}")
        import traceback
        traceback.print_exc()
        return None, []

def probar_grad_cam_simple(model, conv_layers):
    """Probar Grad-CAM con la implementación más simple posible"""
    if not model or not conv_layers:
        print("No se puede probar Grad-CAM sin modelo o capas conv")
        return
    
    print(f"\n=== PRUEBA DE GRAD-CAM ===")
    
    try:
        from tensorflow.keras import backend as K
        
        # Usar la última capa convolucional
        last_conv = conv_layers[-1]
        last_conv_layer = model.get_layer(last_conv[1])
        print(f"Usando capa: {last_conv_layer.name}")
        
        # Crear input de prueba
        test_input = np.random.random((1, 512, 512, 1)).astype(np.float32)
        
        # Predicción
        preds = model.predict(test_input, verbose=0)
        print(f"Predicción shape: {preds.shape}")
        
        # Clase predicha
        predicted_class = np.argmax(preds[0])
        print(f"Clase predicha: {predicted_class}")
        
        # Obtener el output de la clase predicha
        class_output = model.output[:, predicted_class]
        print(f"Class output shape: {class_output.shape}")
        
        # Gradientes
        grads = K.gradients(class_output, last_conv_layer.output)[0]
        print(f"Gradients obtenidos: {grads is not None}")
        
        # Gradientes promedio
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        print(f"Pooled gradients: {pooled_grads.shape}")
        
        # Función para obtener los valores
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        
        # Ejecutar la función
        pooled_grads_value, conv_layer_output_value = iterate([test_input])
        print(f"Pooled grads value shape: {pooled_grads_value.shape}")
        print(f"Conv layer output shape: {conv_layer_output_value.shape}")
        
        print("✅ Grad-CAM básico funciona correctamente")
        
    except Exception as e:
        print(f"❌ Error en Grad-CAM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model, conv_layers = analizar_modelo()
    probar_grad_cam_simple(model, conv_layers)
