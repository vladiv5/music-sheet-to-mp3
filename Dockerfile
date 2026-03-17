FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and FluidSynth
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    fluidsynth \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Start the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]