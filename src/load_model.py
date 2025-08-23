

from manejo_avisos import configurar_logs

configurar_logs()# necesario para eliminar avisos molestos 

import tensorflow as tf


_MODEL = None
_TF_CONFIGURED = False


def _ensure_tf_configured():
    global _TF_CONFIGURED
    if not _TF_CONFIGURED:
        try:
            # Configuración global de TF para compatibilidad con K.gradients/K.function
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.experimental.output_all_intermediates(True)
        except Exception:
            # Si no está disponible en esta versión, continuamos igualmente
            pass
        _TF_CONFIGURED = True


def model_fun():
    """
    Función para cargar el modelo entrenado de neumonía.
    Usa cache para evitar recargar el modelo múltiples veces.
    """
    global _MODEL
    _ensure_tf_configured()

    if _MODEL is not None:
        return _MODEL

    try:
        model = tf.keras.models.load_model('conv_MLP_84.h5', compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Modelo cargado correctamente")
        _MODEL = model
        return _MODEL
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegúrate de que el archivo conv_MLP_84.h5 existe en el directorio actual")
        return None
