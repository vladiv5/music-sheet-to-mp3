# I use the official NVIDIA image with CUDA 11.8 and cuDNN pre-installed
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# I prevent interactive prompts during apt-get installations
ENV DEBIAN_FRONTEND=noninteractive

# I install Python, pip, system dependencies, and the SoundFont in one step
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libglib2.0-0 \
    libgl1 \
    fluidsynth \
    fluid-soundfont-gm \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# I set my working directory inside the container
WORKDIR /app

# I copy the dependencies file
COPY requirements.txt .

# I install the packages, ensuring a clean onnxruntime-gpu environment
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 uninstall -y onnxruntime onnxruntime-gpu && \
    pip3 install --no-cache-dir onnxruntime-gpu==1.16.3

# I create a tiny dummy image and run oemer on it to trigger the automatic checkpoint download
RUN python3 -c "from PIL import Image; Image.new('RGB', (10, 10)).save('dummy.png')" && \
    oemer dummy.png || true && \
    rm dummy.png

# I copy the rest of my application code
COPY . .

# I expose the default Streamlit port
EXPOSE 8501

# I command the container to start the Streamlit server
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]