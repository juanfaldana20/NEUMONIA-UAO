#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Archivo principal de ejecución para el Detector de Neumonía

Este archivo configura el entorno y ejecuta la aplicación modularizada.
"""

import sys
import os
import tensorflow as tf

# Configuración de TensorFlow
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# Agregar las rutas necesarias al PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'src', 'neumonia_uao', 'app')
models_dir = os.path.join(app_dir, 'models')
image_processing_dir = os.path.join(app_dir, 'image_processing')
utils_dir = os.path.join(app_dir, 'utils')
gui_dir = os.path.join(app_dir, 'gui')

# Agregar directorios al path
sys.path.insert(0, app_dir)
sys.path.insert(0, models_dir)
sys.path.insert(0, image_processing_dir)
sys.path.insert(0, utils_dir)
sys.path.insert(0, gui_dir)

# Cambiar al directorio de la aplicación para que pueda encontrar el modelo
os.chdir(app_dir)

# Ahora importar y ejecutar la aplicación
try:
    from gui import main
    print("=== DETECTOR DE NEUMONÍA UAO ===")
    print("Iniciando aplicación...")
    main()
except Exception as e:
    print(f"Error al iniciar la aplicación: {e}")
    import traceback
    traceback.print_exc()
