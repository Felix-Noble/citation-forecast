from typing import cast, NamedTuple
import torch.nn as nn
import importlib

class RegistryEntry(NamedTuple):
    module_path: str 
    model_class: str

def get_model(model_name: str) -> type[nn.Module]:
    model_name = model_name.lower()
    name_registry = {
        'r_rnn' : ['real_rnn', 'real_recurrentnn']
    }

    for name, alts in name_registry.items():
        if model_name in alts:
            model_name = name
            break

    registry: dict[str, RegistryEntry] = {
        'r_rnn': RegistryEntry('src.models.real_rnn', 'R_RNN'),
    }
    
    if model_name not in registry.keys():
        raise KeyError(f'Model "{model_name}" name not registered, check model registry')

    model_registry = registry[model_name]

    module = importlib.import_module(model_registry.module_path)
    model = cast(type[nn.Module], getattr(module, model_registry.model_class))
    if not issubclass(model, nn.Module): # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f'Module [{model_registry.module_path}] exports model [{model_registry.model_class}] not subclass of torch.nn.Module')
    return model 
