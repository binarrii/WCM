# WCM Face Recognition

## Docker

Start the CPU image:

```bash
docker compose up --build
```

Start the CUDA-enabled image on a host with NVIDIA Container Toolkit:

```bash
docker compose -f docker-compose.yaml -f docker-compose.cuda.yaml up --build
```

The CUDA override installs `tensorflow[and-cuda]`, exposes all NVIDIA GPUs to
the API container, and enables TensorFlow GPU memory growth.
