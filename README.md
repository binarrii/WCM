# WCM Face Recognition

## Docker

Start the CPU image:

```bash
docker compose up --build
```

Start the CUDA-enabled image on a host with NVIDIA Container Toolkit:

```bash
docker compose -f compose.yaml -f compose.cuda.yaml up --build
```

The CUDA override installs `tensorflow[and-cuda]`, exposes all NVIDIA GPUs to
the API container, and enables TensorFlow GPU memory growth.

Check whether TensorFlow can see the GPU:

```bash
docker compose -f compose.yaml -f compose.cuda.yaml exec api \
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
