# Estructura Modular del Detector de Neumonía

## Descripción General
Este proyecto está organizado de manera modular para facilitar el mantenimiento, testing y extensibilidad del código.

## Estructura de Directorios

```
src/neumonia_uao/app/
├── detector_neumoniaV2.py      # Punto de entrada principal
├── models/                     # Módulo de modelos de ML
│   ├── __init__.py
│   └── load_model.py           # Función model_fun()
├── image_processing/           # Módulo de procesamiento de imágenes
│   ├── __init__.py
│   ├── grad_cam.py            # Generación de heatmaps Grad-CAM
│   ├── preprocess_img.py      # Preprocesamiento de imágenes
│   └── read_img.py            # Lectura de archivos DICOM/JPG/PNG
├── utils/                     # Módulo de utilidades
│   ├── __init__.py
│   └── integrator.py          # Función predict()
└── gui/                       # Módulo de interfaz gráfica
    ├── __init__.py
    └── gui.py                 # Interfaz Tkinter (App, main)
```

## Funciones Principales por Módulo

### 1. Models (`models/`)
- **model_fun()**: Carga el modelo de TensorFlow entrenado

### 2. Image Processing (`image_processing/`)
- **preprocess(array)**: Preprocesa imágenes para el modelo
- **grad_cam(array)**: Genera mapas de calor Grad-CAM
- **read_dicom_file(path)**: Lee archivos DICOM
- **read_jpg_file(path)**: Lee archivos JPG/JPEG/PNG
- **read_image(path)**: Función unificada de lectura de imágenes

### 3. Utils (`utils/`)
- **predict(array)**: Función principal de predicción que integra todo el pipeline

### 4. GUI (`gui/`)
- **App**: Clase principal de la interfaz gráfica
- **main()**: Función principal para ejecutar la aplicación

## Uso

### Como módulo
```python
# Importar desde el archivo principal
from detector_neumoniaV2 import *

# O importar módulos específicos
from models import model_fun
from image_processing import preprocess, grad_cam
from utils import predict
from gui import App, main
```

### Ejecución directa
```bash
python detector_neumoniaV2.py
```

## Ventajas de esta Estructura

1. **Separación de responsabilidades**: Cada módulo tiene una función específica
2. **Mantenibilidad**: Es fácil modificar o extender funcionalidades específicas
3. **Testability**: Se pueden hacer pruebas unitarias de cada módulo
4. **Reutilización**: Los módulos pueden ser reutilizados en otros proyectos
5. **Escalabilidad**: Fácil agregar nuevas funcionalidades sin afectar el código existente

## Dependencias por Módulo

- **models/**: tensorflow, keras
- **image_processing/**: opencv-cv2, numpy, pydicom, PIL
- **utils/**: numpy, opencv-cv2 
- **gui/**: tkinter, PIL, csv, pyautogui, tkcap, img2pdf

## Archivos de Configuración

- Cada módulo tiene su archivo `__init__.py` que exporta las funciones principales
- Las importaciones relativas permiten que los módulos se comuniquen entre sí
- El archivo principal `detector_neumoniaV2.py` actúa como punto de entrada y orquestador

## Pasos Realizados en la Modularización

1. ✅ Análisis de estructura actual y planificación
2. ✅ Lectura y comprensión de todos los archivos Python 
3. ✅ Organización de funciones de carga de modelo en `models/`
4. ✅ Organización de funciones de procesamiento de imagen en `image_processing/`
5. ✅ Organización de funciones utilitarias en `utils/`
6. ✅ Organización de interfaz gráfica en `gui/`
7. ✅ Actualización de imports en archivo principal
8. ✅ Verificación de sintaxis y estructura modular
9. ✅ Limpieza de archivos duplicados
10. ✅ Documentación de la nueva estructura
