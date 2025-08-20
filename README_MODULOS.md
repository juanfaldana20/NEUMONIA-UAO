# Detector de Neumonía - Estructura Modularizada

## Descripción

Este proyecto ha sido refactorizado para seguir una arquitectura modular que separa las diferentes funcionalidades en módulos específicos, mejorando la mantenibilidad, reutilización y testing del código.

## Estructura de Módulos

### 1. `read_img.py`
**Propósito**: Lectura de imágenes médicas en diferentes formatos.

**Funcionalidades**:
- Lectura de archivos DICOM (`.dcm`)
- Lectura de archivos JPG, JPEG, PNG
- Conversión automática para visualización y procesamiento
- Normalización de datos de imagen

**Funciones principales**:
- `read_image(path)`: Función principal que detecta automáticamente el tipo de archivo
- `read_dicom_file(path)`: Lectura específica de archivos DICOM
- `read_jpg_file(path)`: Lectura de archivos de imagen estándar

### 2. `preprocess_img.py`
**Propósito**: Preprocesamiento de imágenes para el modelo de IA.

**Funcionalidades**:
- Redimensionado a 512x512 píxeles
- Conversión a escala de grises
- Ecualización del histograma con CLAHE
- Normalización entre 0 y 1
- Conversión a formato batch (tensor)

**Funciones principales**:
- `preprocess_image(array)`: Preprocesamiento completo para el modelo
- `preprocess_for_visualization(array)`: Preprocesamiento solo para visualización
- `validate_input_array(array)`: Validación de entrada

### 3. `load_model.py`
**Propósito**: Carga y gestión del modelo de red neuronal.

**Funcionalidades**:
- Carga del modelo `conv_MLP_84.h5`
- Caché del modelo para evitar cargas múltiples
- Recompilación automática para compatibilidad
- Información del modelo

**Funciones principales**:
- `load_pneumonia_model()`: Carga principal del modelo
- `get_model_summary()`: Obtiene resumen del modelo
- `get_model_input_shape()`: Forma de entrada esperada
- `is_model_loaded()`: Verifica si el modelo está cargado

### 4. `grad_cam.py`
**Propósito**: Generación de mapas de calor con Grad-CAM.

**Funcionalidades**:
- Implementación completa de Grad-CAM
- Superposición de mapas de calor sobre imágenes originales
- Manejo automático de capas convolucionales
- Mapas de calor alternativos en caso de fallos

**Funciones principales**:
- `generate_grad_cam(image_array)`: Generación principal de Grad-CAM
- `overlay_heatmap(original_img, heatmap_img)`: Superposición de mapas de calor
- `get_available_conv_layers()`: Lista de capas convolucionales disponibles

### 5. `integrator.py`
**Propósito**: Módulo principal que integra todas las funcionalidades.

**Funcionalidades**:
- Pipeline completo de procesamiento
- Interfaz unificada para la detección
- Manejo de errores centralizado
- Compatibilidad con código existente

**Funciones principales**:
- `process_medical_image(image_path)`: Procesamiento completo desde archivo
- `process_image_array(image_array)`: Procesamiento desde array
- `predict(image_array)`: Función de compatibilidad
- `get_prediction_details(result)`: Detalles de todas las predicciones

**Clase**:
- `PneumoniaDetectionResult`: Encapsula los resultados de la detección

### 6. `detector_neumonia.py` (Refactorizado)
**Propósito**: Interfaz gráfica principal usando los módulos separados.

**Cambios principales**:
- Importación de funciones desde módulos separados
- Eliminación de código duplicado
- Uso de la función unificada `read_image()`
- Mantenimiento de la misma interfaz de usuario

## Uso de los Módulos

### Uso Básico
```python
from integrator import process_medical_image

# Procesar imagen médica
result = process_medical_image("mi_radiografia.dcm")

print(f"Predicción: {result.predicted_class}")
print(f"Probabilidad: {result.probability:.2f}%")

# La imagen con heatmap está disponible en result.heatmap_image
```

### Uso Avanzado
```python
from integrator import process_image_array, get_prediction_details
from read_img import read_image

# Leer imagen manualmente
image_array, display_image = read_image("mi_imagen.jpg")

# Procesar array directamente
result = process_image_array(image_array)

# Obtener detalles de todas las clases
details = get_prediction_details(result)
for clase, probabilidad in details.items():
    print(f"{clase}: {probabilidad:.2f}%")
```

### Uso Individual de Módulos
```python
# Usar módulos por separado
from read_img import read_image
from preprocess_img import preprocess_image
from load_model import load_pneumonia_model
from grad_cam import generate_grad_cam

# Leer imagen
image_array, _ = read_image("imagen.dcm")

# Preprocesar
processed = preprocess_image(image_array)

# Cargar modelo y predecir
model = load_pneumonia_model()
predictions = model.predict(processed)

# Generar Grad-CAM
heatmap = generate_grad_cam(image_array)
```

## Ejecutar la Aplicación

### Interfaz Gráfica
```bash
python detector_neumonia.py
```

### Ejemplo de Uso
```bash
python ejemplo_uso.py
```

## Ventajas de la Estructura Modular

1. **Separación de Responsabilidades**: Cada módulo tiene una función específica
2. **Reutilización**: Los módulos pueden usarse independientemente
3. **Mantenimiento**: Fácil localización y corrección de errores
4. **Testing**: Cada módulo puede probarse por separado
5. **Escalabilidad**: Fácil agregar nuevas funcionalidades
6. **Legibilidad**: Código más organizado y comprensible

## Compatibilidad

La refactorización mantiene total compatibilidad con el código existente:
- La interfaz gráfica funciona exactamente igual
- Las funciones principales mantienen la misma firma
- Los resultados son idénticos al código original

## Requisitos

- Python 3.x
- TensorFlow
- OpenCV
- PIL/Pillow
- NumPy
- pydicom
- tkinter (para interfaz gráfica)
- tkcap (para generación de PDFs)

## Archivos del Proyecto

```
UAO-Neumonia/
├── detector_neumonia.py      # Interfaz gráfica principal (refactorizada)
├── integrator.py             # Módulo integrador principal
├── read_img.py              # Lectura de imágenes médicas
├── preprocess_img.py        # Preprocesamiento de imágenes
├── load_model.py            # Carga del modelo de IA
├── grad_cam.py              # Generación de mapas de calor
├── ejemplo_uso.py           # Ejemplo de uso de los módulos
├── README_MODULOS.md        # Esta documentación
├── conv_MLP_84.h5          # Modelo de red neuronal (requerido)
└── main.py                 # Archivo principal alternativo
```

## Notas Importantes

1. **Modelo Requerido**: El archivo `conv_MLP_84.h5` debe estar presente en el directorio del proyecto
2. **Configuración TensorFlow**: Se mantiene la configuración de compatibilidad para TensorFlow
3. **Cache del Modelo**: El modelo se carga una vez y se mantiene en cache para mejorar el rendimiento
4. **Manejo de Errores**: Cada módulo incluye manejo robusto de errores con mensajes informativos
