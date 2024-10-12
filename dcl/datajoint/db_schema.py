import datajoint as dj
import datajoint_pytypes.fields

__all__ = ["Schema", "schema"]


class Schema(dj.Schema):

    def _definition(self, cls):
        """Convert a python based table definition into a datajoint table.

        Decorate a class with this function to add a ``definition`` attribute.
        You still need to add the ``@schema`` constructor used for datajoint.
        """

        primary_keys = []
        secondary_keys = []
        # Traverse the MRO to gather all attributes
        # If we don't do this, we will miss fields defined in superclasses
        for base_cls in cls.mro():
            if base_cls is object:  # Stop at the top of the hierarchy
                continue
            for key, value in base_cls.__dict__.items():
                if isinstance(value, datajoint_pytypes.fields.Field):
                    if value.primary_key:
                        primary_keys.append(value.name(key))
                    else:
                        secondary_keys.append(value.name(key))

        primary = "\n".join(primary_keys)
        secondary = "\n".join(secondary_keys)

        if cls.__doc__ is not None:
            doc = cls.__doc__.strip()
        else:
            doc = ""

        definition = [
            f"# {doc}" if len(doc) > 0 else "", primary, "---", secondary
        ]

        definition = "\n".join(definition)
        return definition

    def __call__(self, cls, *, context=None):
        cls.definition = self._definition(cls)
        cls = super().__call__(cls, context=context)
        return cls


schema = Schema
