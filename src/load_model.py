import tensorflow as tf

def model_fun():
    """
    Función para cargar el modelo entrenado de neumonía.
    """
    try:
        model = tf.keras.models.load_model('conv_MLP_84.h5', compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Modelo cargado correctamente")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegúrate de que el archivo conv_MLP_84.h5 existe en el directorio actual")
        return None

model = model_fun()