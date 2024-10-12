import importlib
import pkgutil

# Dynamically import all submodules in this package
for _, module_name, _ in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    importlib.import_module(module_name)

# Optionally define __all__
__all__ = [name for _, name, _ in pkgutil.iter_modules(__path__)]
