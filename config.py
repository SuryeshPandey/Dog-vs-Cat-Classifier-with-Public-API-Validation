import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Corrected paths
IMAGE_DIR = os.path.join(BASE_DIR, "images", "images")  # Where raw images are located
PROCESSED_DIR = os.path.join(BASE_DIR, "images", "processed")  # Where processed images will be saved
METADATA_FILE = os.path.join(BASE_DIR, "metadata.csv")

# (Optional) Any external API keys or constants can go here
