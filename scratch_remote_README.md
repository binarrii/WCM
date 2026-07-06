# DeepFace API with PGVector & Load Balancing

This directory contains the Docker Compose configuration to deploy the [DeepFace API](https://hub.docker.com/r/serengil/deepface) backed by `pgvector` as the vector database, optimized with **Nginx Load Balancing** and **NVIDIA MPS** for hardware-level GPU VRAM isolation.

## Architecture Highlights

- **Load Balancing (Nginx)**: Distributes incoming requests on port `5000` across 2 identical `deepface-api` container replicas to maximize concurrency.
- **Hardware-Level GPU VRAM Limit (NVIDIA MPS)**: 
  Each DeepFace API instance is hard-limited to 12GB of VRAM (25% of a 48GB GPU) using NVIDIA's Multi-Process Service (MPS). This prevents out-of-memory errors and allows multiple instances to share a single GPU efficiently.
- **Vector Database**: PostgreSQL with the `pgvector` extension is used to store and query facial embeddings.

## Services Overview

1. **deepface-nginx** (`nginx:alpine`)
   - **Port**: `5000` (Mapped to `0.0.0.0:5000`)
   - **Description**: Reverse proxy that round-robins traffic to the backend `deepface-api` instances.

2. **deepface-api** (`deepface-gpu:latest`) - **(2 Replicas)**
   - **Environment**: 
     - `DEEPFACE_DATABASE_TYPE=pgvector` (Enables native pgvector engine instead of fallback postgres engine)
     - `DEEPFACE_CONNECTION_DETAILS` configured for PostgreSQL.
     - `TF_FORCE_GPU_ALLOW_GROWTH=true` 
     - `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0.25` (12GB VRAM cap via MPS)
   - **IPC**: Mapped to host (`ipc: host`) to communicate with the host's NVIDIA MPS daemon.
   - **Database Note**: With the `pgvector` engine enabled, DeepFace dynamically generates specialized tables (e.g., `embeddings_facenet512_retinaface_aligned_raw`) using native `vector(N)` columns instead of the generic `embeddings` and `embeddings_index` tables.

3. **pgvector** (`pgvector/pgvector:pg15`)
   - **Port**: `5434` (Mapped to host port `5434`, internal `5432`)
   - **Credentials**: `deepface_user` / `deepface_pass`

## Directory Structure

```text
/opt/binarii/DeepFace/
├── docker-compose.yml       # Docker Compose configuration file
├── Dockerfile               # Custom Dockerfile for GPU acceleration
├── nginx.conf               # Nginx load balancer configuration
├── pgvector-init/           
│   └── init.sql             # SQL script to create the pgvector extension on startup
└── README.md                # This documentation file
```

## Management Commands

To manage the services, navigate to `/opt/binarii/DeepFace` and run the following commands (requires `sudo`):

- **Start NVIDIA MPS Daemon (Required after host reboot):**
  ```bash
  sudo nvidia-cuda-mps-control -d
  ```

- **Start/Restart Services:**
  ```bash
  sudo docker compose up -d --remove-orphans
  ```

- **Check status:**
  ```bash
  sudo docker compose ps
  ```

- **View logs:**
  ```bash
  # All services
  sudo docker compose logs -f

  # Only API logs (will interleave logs from both replicas)
  sudo docker compose logs -f deepface-api
  ```

- **Stop services:**
  ```bash
  sudo docker compose down
  ```

## API Testing

You can verify the API is running by sending a request to port 5000. Nginx will route the request to one of the replicas:

```bash
curl -X GET http://127.0.0.1:5000/
```

For full DeepFace API usage, refer to the [DeepFace Documentation](https://github.com/serengil/deepface).
