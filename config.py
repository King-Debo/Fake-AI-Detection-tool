# Import os module for accessing environment variables
import os

# Define some configuration settings and parameters for the app

# The secret key for the app, which can be set as an environment variable or generated randomly
SECRET_KEY = os.environ.get("SECRET_KEY") or os.urandom(24)

# The upload folder for storing the media files, which can be set as an environment variable or as a static folder
UPLOADED_MEDIA_DEST = os.environ.get("UPLOADED_MEDIA_DEST") or "static/uploads"

# The maximum content length for uploading files, which can be set as an environment variable or as 16 MB
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH") or 16 * 1024 * 1024)

# The learning rate for fine-tuning the models, which can be set as an environment variable or as 1e-4
LEARNING_RATE = float(os.environ.get("LEARNING_RATE") or 1e-4)

# The number of epochs for fine-tuning the models, which can be set as an environment variable or as 10
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS") or 10)

# The batch size for fine-tuning the models, which can be set as an environment variable or as 32
BATCH_SIZE = int(os.environ.get("BATCH_SIZE") or 32)