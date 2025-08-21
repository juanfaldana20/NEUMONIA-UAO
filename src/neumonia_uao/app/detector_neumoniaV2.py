#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detector de Neumonía - Archivo Principal Modularizado

Este archivo sirve como punto de entrada principal que importa 
todas las funcionalidades desde los módulos específicos.

Uso:
    python detector_neumoniaV2.py
"""

import tensorflow as tf

# Configuración de TensorFlow para compatibilidad
# tf.compat.v1.disable_eager_execution()  # Comentado - puede causar problemas
# tf.compat.v1.experimental.output_all_intermediates(True)  # Comentado

# Importar funciones desde módulos
try:
    # Intentar importación relativa (cuando se ejecuta como módulo)
    from .models import model_fun
    from .image_processing import preprocess, grad_cam, read_dicom_file, read_jpg_file, read_image
    from .utils import predict
    from .gui import App, main
except ImportError:
    # Importación absoluta (cuando se ejecuta directamente)
    from models import model_fun
    from image_processing import preprocess, grad_cam, read_dicom_file, read_jpg_file, read_image
    from utils import predict
    from gui import App, main


if __name__ == "__main__":
    main()
