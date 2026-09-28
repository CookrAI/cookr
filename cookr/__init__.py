TRIGGER = "cookr"
__version__ = "0.1.0"


def __getattr__(name):
    # lazy: importing the data pipeline must not pull torch/diffusers
    if name in ("Cookr", "build_prompt"):
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
