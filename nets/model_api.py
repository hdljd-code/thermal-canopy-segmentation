from importlib import import_module


MODEL_MODULES = {
    "asf": "nets.asf",
    "msb": "nets.msb",
    "asf_msb": "nets.asf_msb",
}
PROTECTED_PROVIDER = "nets.protected_models"
MODEL_UNAVAILABLE_MESSAGE = "The requested model is unavailable."


def _load_model_class(model_name):
    try:
        provider = import_module(PROTECTED_PROVIDER)
    except ModuleNotFoundError as error:
        if error.name != PROTECTED_PROVIDER:
            raise
        raise RuntimeError(MODEL_UNAVAILABLE_MESSAGE) from None

    factory = getattr(provider, "get_model_class", None)
    if not callable(factory):
        raise RuntimeError(MODEL_UNAVAILABLE_MESSAGE)
    return factory(model_name)


def get_model_class(model_name):
    if model_name not in MODEL_MODULES:
        raise ValueError("Unsupported model: {}".format(model_name))

    model_module = import_module(MODEL_MODULES[model_name])
    return model_module.get_model_class()
