import importlib
from typing import Optional

def is_module_available(module_name: str) -> bool:
    """
    Checks if a Python module is available for import.
    """
    try:
        importlib.import_module(module_name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False

# Dependency flags
HAS_HDBSCAN = is_module_available("hdbscan")
HAS_OPENAI = is_module_available("openai")
HAS_FOLIUM = is_module_available("folium")
HAS_MATPLOTLIB = is_module_available("matplotlib")
HAS_LANGDETECT = is_module_available("langdetect")
HAS_SENTENCE_TRANSFORMERS = is_module_available("sentence_transformers")
HAS_ALPHASHAPE = is_module_available("alphashape")
HAS_SHAPELY = is_module_available("shapely")
HAS_SKLEARN = is_module_available("sklearn")
HAS_SCIPY = is_module_available("scipy")
HAS_FASTAPI = is_module_available("fastapi")
HAS_UVICORN = is_module_available("uvicorn")
