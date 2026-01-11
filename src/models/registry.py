import torch.nn as nn
import importlib
import inspect
from pathlib import Path

def build_registry() -> dict[str, type[nn.Module]]:
    module_path = ('src' + str(Path(__file__).parent).split('src')[-1]).replace('/', '.')
    current_path = Path(__file__)
    module_names = [str(f.stem) for f in Path(__file__).parent.glob('*.py') if f != current_path]

    registry: dict[str, type[nn.Module]] = {}
    for module_name in module_names:
        module = importlib.import_module(module_path + '.' + str(module_name))

        components = [clss for name,clss in inspect.getmembers(module, inspect.isclass) if hasattr(clss, 'MODEL_NAME')]
        if len(components) > 1:
            raise ValueError (f'Found two models in {module_name}, only one model class per module (file) expected')
        if not components:
            raise ValueError(f"No valid classes found in '{module_name}', ensure model classes have 'MODEL_NAME' attribute")
        model = components[0]
        model_name = getattr(model, 'MODEL_NAME')
        
        registry[model_name] = model
    
    return registry

def get_model(model_name: str) -> type[nn.Module]:
    model_name = model_name.lower()
    alias_registry = {
        'r_rnn' : ['real_rnn', 'real_recurrent']
    }

    for name, alts in alias_registry.items():
        if model_name in alts:
            model_name = name
            break

    registry: dict[str, type[nn.Module]] = build_registry()
    if model_name not in registry.keys():
        raise KeyError(f'Model "{model_name}" name not found. Available models: [ {", ".join(registry.keys())} ]')
    model = registry[model_name]
    # TODO: check model config with pyright here

    return model 
