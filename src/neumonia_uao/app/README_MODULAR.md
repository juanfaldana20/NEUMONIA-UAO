# Estructura Modular del Detector de Neumonía

Esta es la documentación de la nueva estructura modular del detector de neumonía V2.

## Estructura de Carpetas

```
src/neumonia_uao/app/
├── detector_neumoniaV2.py          # Archivo principal (punto de entrada)
├── models/                         # Módulo para manejo de modelos ML
│   ├── __init__.py
│   └── load_model.py              # Carga y compilación del modelo
├── image_processing/              # Módulo para procesamiento de imágenes
│   ├── __init__.py
│   ├── preprocess_img.py         # Preprocesamiento de imágenes
│   ├── grad_cam.py               # Generación de mapas de calor
│   └── read_img.py               # Lectura de archivos DICOM/JPG
├── utils/                        # Módulo de utilidades
│   ├── __init__.py
│   └── integrator.py             # Función principal de predicción
├── gui/                          # Módulo de interfaz gráfica
│   ├── __init__.py
│   └── gui.py                    # Clase App y funcionalidades GUI
└── README_MODULAR.md            # Este archivo
```

## Descripción de Módulos

### 1. `models/`
- **load_model.py**: Contiene la función `model_fun()` que carga el modelo de TensorFlow desde el archivo `.h5`

### 2. `image_processing/`
- **preprocess_img.py**: Función `preprocess()` que prepara las imágenes para el modelo
- **grad_cam.py**: Función `grad_cam()` que genera mapas de calor para visualización
- **read_img.py**: Funciones para leer archivos DICOM y JPG/PNG

### 3. `utils/`
- **integrator.py**: Función `predict()` que coordina todo el proceso de predicción

### 4. `gui/`
- **gui.py**: Clase `App` que maneja toda la interfaz gráfica de usuario

## Uso

### Ejecutar la aplicación:
```bash
python detector_neumoniaV2.py
```

### Importar módulos específicos:
```python
from .models import model_fun
from .image_processing import preprocess, grad_cam
from .utils import predict
from .gui import App, main
```

## Ventajas de la Modularización

1. **Separación de responsabilidades**: Cada módulo tiene una función específica
2. **Reutilización**: Los módulos pueden ser importados independientemente
3. **Mantenimiento**: Más fácil de mantener y actualizar
4. **Testing**: Cada módulo puede ser probado por separado
5. **Escalabilidad**: Fácil agregar nuevas funcionalidades

## Archivos Originales vs Modulares

| Funcionalidad | Archivo Original | Nuevo Módulo |
|---------------|------------------|--------------|
| Carga de modelo | detector_neumoniaV2.py | models/load_model.py |
| Preprocesamiento | detector_neumoniaV2.py | image_processing/preprocess_img.py |
| Grad-CAM | detector_neumoniaV2.py | image_processing/grad_cam.py |
| Lectura imágenes | detector_neumoniaV2.py | image_processing/read_img.py |
| Predicción | detector_neumoniaV2.py | utils/integrator.py |
| Interfaz GUI | detector_neumoniaV2.py | gui/gui.py |
