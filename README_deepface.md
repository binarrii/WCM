# DeepFace API with PGVector

This directory contains the Docker Compose configuration to deploy the official [DeepFace API](https://hub.docker.com/r/serengil/deepface) along with `pgvector` as the vector database backend.

## Services Overview

1. **deepface-api** (`serengil/deepface:latest`)
   - **Port**: `5000` (Mapped to `0.0.0.0:5000`)
   - **Environment**: Connected to PostgreSQL via `DB_URI=postgresql://deepface_user:deepface_pass@pgvector:5432/deepface`
   - **Description**: Exposes REST API endpoints for face recognition, verification, and analysis.

2. **pgvector** (`pgvector/pgvector:pg15`)
   - **Port**: `5434` (Mapped to host port `5434`, internal `5432`)
   - **Credentials**:
     - DB Name: `deepface`
     - User: `deepface_user`
     - Password: `deepface_pass`
   - **Volumes**: Data is persisted in the `postgres_data` docker volume.
   - **Initialization**: Automatically creates the `vector` extension on startup via `./pgvector-init/init.sql`.

## Directory Structure

```text
/opt/binarii/DeepFace/
├── docker-compose.yml       # Docker Compose configuration file
├── pgvector-init/           
│   └── init.sql             # SQL script to create the pgvector extension on startup
└── README.md                # This documentation file
```

## Management Commands

To manage the services, navigate to `/opt/binarii/DeepFace` and run the following commands (requires `sudo`):

- **Start in background:**
  ```bash
  sudo docker compose up -d
  ```

- **Check status:**
  ```bash
  sudo docker compose ps
  ```

- **View logs:**
  ```bash
  # All services
  sudo docker compose logs -f

  # Only API logs
  sudo docker compose logs -f deepface-api
  ```

- **Stop services:**
  ```bash
  sudo docker compose down
  ```

## API Testing

You can verify the API is running by sending a request to port 5000:

```bash
curl -X GET http://127.0.0.1:5000/
```

For full DeepFace API usage, refer to the [DeepFace Documentation](https://github.com/serengil/deepface).
