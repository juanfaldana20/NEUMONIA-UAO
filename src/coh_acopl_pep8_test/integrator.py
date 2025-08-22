import numpy as np
from preprocess_img import preprocess
from load_model import model_fun
from grad_cam import grad_cam


def predict(array):
    batch_array_img = preprocess(array)
    model = model_fun()
    if model is None:
        raise RuntimeError("No se pudo cargar el modelo 'conv_MLP_84.h5'.")

    preds = model.predict(batch_array_img)
    prediction = int(np.argmax(preds))
    proba = float(np.max(preds)) * 100.0

    if prediction == 0:
        label = "bacteriana"
    elif prediction == 1:
        label = "normal"
    else:
        label = "viral"

    heatmap = grad_cam(array)
    return (label, proba, heatmap)
