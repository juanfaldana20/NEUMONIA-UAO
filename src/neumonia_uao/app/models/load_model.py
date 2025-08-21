import tensorflow as tf

def model_fun():
    """
    Función para cargar el modelo entrenado de neumonía.
    """
    try:
        print("Cargando modelo...")
        # Intentar cargar con compile=False para evitar problemas de compatibilidad
        model = tf.keras.models.load_model('conv_MLP_84.h5', compile=False)
        print("Modelo cargado correctamente")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegúrate de que el archivo conv_MLP_84.h5 existe en el directorio actual")
        print("Si el problema persiste, podrías necesitar reentrenar el modelo con la versión actual de TensorFlow")
        return None
