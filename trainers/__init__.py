import importlib

# Define function_registry
function_registry = {}

def register_function(name):
    """register_function.

    Add a function to the function_registry.
    Use this as a decorator on functions defined in the rest of the directory.

    Args:
        name (str): The name under which the function will be registered.
    """
    def decorator(func):
        function_registry[name] = func
        return func
    return decorator


def get_function(function_name):
    """Get a function from the registry by name."""
    return function_registry.get(function_name)


def launch_training(function_name, **kwargs):
    """Build and call a function from the registry by name."""
    func = get_function(function_name)
    if func:
        return func(**kwargs)
    else:
        raise ValueError(f"Function '{function_name}' not found in registry.")


from .spt import *
from .matching import *
from .rxn import *
