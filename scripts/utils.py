import importlib.util
import os


def load_custom_function(file_path, func_name='reward_func'):
    if file_path is None:
        raise ValueError("No path provided for custom reward function.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Specified file {file_path} does not exist.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    custom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)

    if not hasattr(custom_module, func_name):
        raise AttributeError(f"The function '{func_name}' is not found in the specified file.")

    return getattr(custom_module, func_name)