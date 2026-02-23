FROM python:3.11-slim

WORKDIR /app

# Instalar la librería de sistema libgomp1 requerida por LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar configuración de requerimientos de Python
COPY requirements.txt .

# Instalar los paquetes de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el grueso de archivos del proyecto, modelos y CSVs
COPY . .

# Comando principal de Railway dictado por Procfile
CMD ["python", "bot.py"]
