# from typeguard._exceptions import TypeCheckError
import hashlib
import json
import warnings
from dataclasses import dataclass
from dataclasses import Field
from dataclasses import field
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

from filelock import FileLock
from typeguard import check_type

T = TypeVar("T")


def hash_key(s: str) -> str:
    """Generate a short hash from a string."""
    hash_value = hashlib.md5(s.encode()).hexdigest()
    return hash_value


def check_initialized(func):
    """function decorator to check if class is initialized."""

    def wrapper(self, *args, **kwargs):
        if not self.initialized:
            raise RuntimeError(
                f"Class {self.__class__.__name__} not initialized. Call .lazy_init() first."
            )
        return func(self, *args, **kwargs)

    return wrapper


@dataclass(kw_only=True)
class LazyInitializable:
    lazy: bool = field(default=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    @property
    def initialized(self) -> bool:
        """Whether the object has been initialized."""
        return self._initialized

    def __lazy_post_init__(self, *args, **kwargs):
        pass

    def validate_config(self):
        pass

    def validate_config_post_init(self):
        pass

    def lazy_init(self, *args, **kwargs):
        self._initialized = True
        self.__lazy_post_init__(*args, **kwargs)
        self.validate_config_post_init()

    def __post_init__(self):
        self.validate_config()
        if not self.lazy:
            self.lazy_init()


class Configurable(LazyInitializable):
    """Class for configurable dataclass objects."""
    _registry: Dict[str, Any] = {}

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        camel_case_name = cls.__name__
        # convert camel case to snake case
        return ''.join([
            '_' + i.lower() if i.isupper() else i for i in camel_case_name
        ]).lstrip('_')

    @property
    def config(self) -> Dict[str, Any]:
        """Get all configuration attributes."""
        config_dict = {}
        for f in fields(self):
            if f.metadata.get("type") == "config":
                config_dict[f.name] = getattr(self, f.name)
        return config_dict

    @staticmethod
    def config_fields(class_name: str) -> List[Field]:
        """Get the fields that are config fields for a given class."""
        return [
            f for f in fields(Configurable._registry[class_name])
            if f.metadata.get("type") == "config"
        ]

    @property
    def config_hash(self) -> str:
        """Hash of the configuration, taking into account sub-configs."""
        config_dict = self.to_dict()
        return Configurable.hash_config_dict(config_dict)

    @staticmethod
    def hash_config_dict(config_dict: Dict[str, Any]) -> str:
        """Hash of the configuration, taking into account sub-configs."""
        hash_dict = {}
        for key, value in config_dict.items():
            # check if value is a Configurable object
            # either via name lookup or checking if "class_type" is in the dict
            if isinstance(value, dict) and "class_type" in value:
                class_type = value["class_type"]
                assert class_type in Configurable._registry, f"Class {class_type} not registered"
                hash_dict[key] = Configurable.hash_config_dict(value)
            else:
                hash_dict[key] = value

        return hash_key(json.dumps(hash_dict, sort_keys=True))

    @classmethod
    def config_name(cls) -> str:
        """Name of the configuration file."""
        return f"{cls.config_prefix()}_config.json"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # assert cls.__name__ not in cls._registry, f"Duplicate Config class {cls.__name__}"
        if cls.__name__ in cls._registry:
            warnings.warn(
                f"Duplicate Config class {cls.__name__}. Overwriting existing class in registry."
            )
        Configurable._registry[cls.__name__] = cls

    def to_dict(self) -> Dict[str, Any]:
        config_dict = self.config
        for key, value in config_dict.items():
            # handle configurable objects
            if isinstance(value, Configurable):
                config_dict[key] = value.to_dict()

        # add type to config
        config_dict["class_type"] = self.__class__.__name__
        return config_dict

    def clone(self: T) -> T:
        """Clone the object."""
        return self.from_dict(self.to_dict(), lazy=self.lazy)

    def save(self, path: Path) -> Path:
        """Save configuration to disk."""
        path.mkdir(parents=True, exist_ok=True)

        lock = FileLock(path / "lock.lock", timeout=10)
        lock.acquire()
        try:
            nested_config_dict = self.to_dict()
            config_path = path / self.__class__.config_name()
            with open(config_path, 'w') as f:
                json.dump(nested_config_dict, f, indent=4)

            self._recursive_save_additional(path)
        finally:
            lock.release(force=True)
        return config_path

    def _recursive_save_additional(self, path: Path):
        self._save_additional(path)
        for key, value in self.config.items():
            if isinstance(value, Configurable):
                # create a new path for additional save
                sub_path = path / f"{key}/"
                value._recursive_save_additional(sub_path)

    def _save_additional(self, path: Path):
        """Hook for additional saving of objects besides the config."""

    @classmethod
    def validate_config_dict(cls, config_dict: Dict[str, Any]) -> bool:
        """Validate the config dictionary.

        Args:
            cls: The class to validate against
            config_dict: Dictionary containing configuration values

        Returns:
            bool: True if config is valid, False otherwise
        """
        # Get all config fields for this class
        if "class_type" not in config_dict:
            return False
        class_name = config_dict["class_type"]
        config_fields = Configurable.config_fields(class_name)

        # Check that all required fields are present
        for f in config_fields:
            if f.name not in config_dict:
                if f.default is None and f.default_factory is None:
                    return False
                continue

            # If field exists, validate its type
            value = config_dict[f.name]

            # Handle nested Configurable objects
            if isinstance(value, dict) and "class_type" in value:
                # Verify the class type exists in registry
                class_type = value["class_type"]
                if class_type not in Configurable._registry:
                    return False

                # Recursively validate nested config
                nested_cls = Configurable._registry[class_type]
                if not nested_cls.validate_config_dict(value):
                    return False

            # Check type matches field type annotation for non-nested values
            # else:
            #     try:
            #         if isinstance(value, f.type):
            #             continue
            #     except TypeError as e:
            #         # complex types (like Literals, etc.) can not be checked with isinstance
            #         # however simple types like int, float, str, etc. can be checked with check_type
            #         if check_type(value, f.type):
            #             continue
            #     return False
            else:
                try:
                    check_type(argname=f.name,
                               value=value,
                               expected_type=f.type)
                except Exception as e:
                    print(e)
                    return False

        return True

    @classmethod
    def from_dict(
        cls,
        config: Dict[str, Any],
        kwargs: Optional[Dict[str, Any]] = None,
        lazy: Optional[bool] = None,
        path: Optional[Path] = None,
    ) -> "Configurable":
        """Load object from dictionary."""
        config = config.copy()
        initialized_config = {}
        assert "class_type" in config, "Config must contain a class_type"
        object_type = config.pop("class_type")

        for key, value in config.items():
            if isinstance(value, dict) and value.get("class_type"):
                object_class = Configurable._registry[value["class_type"]]
                initialized_config[key] = object_class.from_dict(value,
                                                                 kwargs=kwargs,
                                                                 lazy=lazy)
            else:
                initialized_config[key] = value
        instance_kwargs = kwargs.get(object_type,
                                     {}) if kwargs is not None else {}
        if lazy is not None:
            instance_kwargs["lazy"] = lazy
        instance = cls(**initialized_config, **instance_kwargs)
        if path is not None:
            instance.locked_recursive_load_additional(path)
        return instance

    @classmethod
    def load(cls, path: Union[Path, str], **kwargs) -> "Configurable":
        """Load object from disk."""
        path = Path(path)
        config_path = path / cls.config_name()
        with open(config_path, 'r') as f:
            config = json.load(f)
        instance = cls.from_dict(config, path=path, **kwargs)
        return instance

    def _load_additional(self, path: Path):
        """Hook for post-initialization of loaded objects."""

    def _recursive_load_additional(self, path: Path):
        self._load_additional(path)
        for key, value in self.config.items():
            if isinstance(value, Configurable):
                # create a new path for additional save
                sub_path = path / f"{key}/"
                value._recursive_load_additional(sub_path)

    def locked_recursive_load_additional(self, path: Path, timeout: int = 10):
        lock = FileLock(path / "lock.lock", timeout=timeout)
        lock.acquire()
        try:
            self._recursive_load_additional(path)
        finally:
            lock.release(force=True)
