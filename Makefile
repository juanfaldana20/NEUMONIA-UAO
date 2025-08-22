# Detectar sistema operativo
ifeq ($(OS),Windows_NT)
    PYTHON = .venv\Scripts\python.exe
    RM = powershell -Command "Remove-Item -Recurse -Force __pycache__, *.pyc, *.pyo, *.log -ErrorAction SilentlyContinue"
else
    PYTHON = .venv/bin/python
    RM = rm -rf __pycache__ *.pyc *.pyo *.log
endif

# Ejecutar la aplicación
run:
	$(PYTHON) app.py

# Instalar dependencias
install:
	$(PYTHON) -m pip install -r requirements.txt

# Limpiar archivos temporales
clean:
	$(RM)
