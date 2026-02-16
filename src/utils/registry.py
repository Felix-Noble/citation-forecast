class Registry:
    def __init__(self):
        self._map = {}
    
    def __getitem__(self, key):
        return self._map[key]

    def get(self, key):
        return self.__getitem__(key)
    
    @property
    def keys(self):
        return list(self._map.keys()) 

    def register(self, name: str | None=None):
        def _wrapper(obj):
            assert name not in self.keys, f'{name} already in registry'
            if  name is not None:
                registry_name = name
            else:
                registry_name = obj.__name__
            self._map.update({registry_name.lower(): obj})
            return obj
        return _wrapper
    
    def __call__(self, name=None):
        return self.register(name)

if __name__ == '__main__':
    registry = Registry()

    @registry()
    class Example:
        test = 'yes'

    print(registry._map)
    print(registry['Example'])
    print(registry['Example']())
