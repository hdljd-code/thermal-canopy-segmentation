"""Public entry point for the MSB model configuration."""

from .model_api import _load_model_class


def get_model_class():
    return _load_model_class("msb")
