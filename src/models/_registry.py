import torch.nn as nn
from pydantic import ValidationError

from utils import Registry

model_registry = Registry()


def get_model(
    config,
) -> type[nn.Module]:
    model_name = config.model.model_name.lower()

    if model_name not in model_registry.keys:
        raise KeyError(
            f'Model "{model_name}" name not found. Available models: [ {", ".join(registry.keys())} ]'
        )
    model = model_registry[model_name]

    # verify model config
    try:
        _ = model.config_schema(**config.model.__dict__)  # pyright: ignore [reportAttributeAccessIssue]
    except ValidationError as e:
        for error in e.errors():
            print(error, "\n")
        quit()

    return model
