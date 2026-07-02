FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install dependencies for OpenCV, deepface, etc.
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --ignore-installed blinker deepface tf-keras opencv-python-headless
