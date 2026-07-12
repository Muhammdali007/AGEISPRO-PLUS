"""Stable test configuration independent of a developer's local .env file."""

import os


os.environ["ENVIRONMENT"] = "test"
os.environ["API_RECOGNITION_BACKEND"] = "hash"
os.environ["API_RECOGNITION_ALLOW_FALLBACK"] = "true"
