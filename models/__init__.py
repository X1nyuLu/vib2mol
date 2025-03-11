# Define model_types
model_registry = {}


def register_model(cls):
    """register_model.

    Add an argument to the model_registry.
    Use this as a decorator on classes defined in the rest of the directory.

    """
    model_registry[cls.__name__] = cls
    return cls


def get_model_class(model):
    return model_registry[model]


def build_model(model, **kwargs):
    """build_model"""
    return get_model_class(model)(**kwargs)
