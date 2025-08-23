import os
import warnings
import logging

def configurar_logs():
    """Configura los logs y warnings para TensorFlow y Python."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Ocultar INFO y WARNING de TF
    warnings.filterwarnings("ignore")  # Ignorar warnings no críticos
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
