"""Stable test configuration independent of a developer's local .env file."""

import os


os.environ["AI_ENVIRONMENT"] = "test"
os.environ["AI_MODEL_BACKEND"] = "simulated"
os.environ["AI_MODEL_FALLBACK_BACKEND"] = "simulated"
os.environ["AI_ALLOW_BACKEND_FALLBACK"] = "true"
os.environ["AI_RECOGNITION_BACKEND"] = "hash"
os.environ["AI_RECOGNITION_ALLOW_FALLBACK"] = "true"
