"""
Configuration parameters for fault injection experiments with TensorFlow 2.19.0

This module contains training hyperparameters, model configurations, and system settings
for the fault injection framework.
"""

# Training Parameters
EPOCHS = 80
BATCH_SIZE = 1024
VALID_BATCH_SIZE = 1000
PER_REPLICA_BATCH_SIZE = 128
PER_REPLICA_VALID_BATCH_SIZE = 125

# Model Configuration
NUM_CLASSES = 10
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
CHANNELS = 3

# Directory Configuration
GOLDEN_MODEL_DIR = "ISCA_AE_CKPT/"
OUTPUT_DIR = "outputs/"
LOG_DIR = "logs/"

# TensorFlow 2.19.0 Specific Settings
MIXED_PRECISION = False  # Enable mixed precision training
XLA_COMPILE = False      # Enable XLA compilation for performance
MEMORY_GROWTH = True     # Enable GPU memory growth

# Backward compatibility aliases
image_height = IMAGE_HEIGHT
image_width = IMAGE_WIDTH
channels = CHANNELS
golden_model_dir = GOLDEN_MODEL_DIR
