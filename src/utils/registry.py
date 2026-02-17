from typing import Callable

class Registry:
    """ Registry class
        Stores {key, callable} pairs through decoration
    """
    def __init__(self):
        self._map: dict[str, Callable] = {}
    
    def __getitem__(self, key) -> Callable:
        """ Access to values via key indexing """
        return self._map[key]

    def get(self, key) -> Callable:
        """ Alias for __get__item 
            Consistency with dict class """
        return self.__getitem__(key)
    
    @property
    def keys(self) -> list[str]:
        """ List available keys
            Allows easier membership checking"""
        return list(self._map.keys()) 
    
    def _register(
            self, 
            name: str,
            obj: Callable,
                  ) -> None:
        """ Adds key to map
            Checks if key already in map """

        assert name not in self.keys, f'{name} already in registry'
        self._map.update({name: obj})

    def register(self, obj_or_name: Callable | str) -> Callable:
        """ Decorator for _register
            Allows both argument and no-argument decoration """

        if callable(obj_or_name):
            self._register(obj_or_name.__name__.lower(), obj_or_name)
            return obj_or_name

        elif isinstance(obj_or_name, str):
            def _wrapper(obj):
                self._register(obj_or_name, obj)
                return obj
            return _wrapper

        else:
            raise TypeError(f'obj_or_name must be str or Callable, got {obj_or_name} which is type {type(obj_or_name)}')

    def __call__(self, obj_or_name: str | Callable) -> Callable:
        """ Instance callable alias for register """
        return self.register(obj_or_name) 

if __name__ == '__main__':
    registry = Registry()

    @registry
    class Test1:
        test = 1
        def __call__(self):
            return 'sucess 1'

    @registry.register
    class Test2:
        test=2
        def __call__(self):
            return 'sucess 2'

    @registry.register('new_name')
    class TestName:
        def __call__(self):
            return 'sucess 3'

    print(registry._map)
    print(registry['test1'])
    print(registry['test1']()())
    print(registry['test2'])
    print(registry['test2']()())
    print(registry['new_name'])
    print(registry['new_name']()())
    

