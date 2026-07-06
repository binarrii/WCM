import tensorflow as tf
import os

# Get the limit from environment variable, default to 12GB
limit_mb = int(os.environ.get("TF_MEMORY_LIMIT_MB", 12288))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)])
        print(f"✅ Successfully set GPU memory limit to {limit_mb} MB for all GPUs", flush=True)
    except RuntimeError as e:
        print(f"⚠️ Failed to set GPU memory limit: {e}", flush=True)

# Important: After setting the TF config, import the DeepFace app factory
from deepface.api.src.app import create_app
app = create_app()
