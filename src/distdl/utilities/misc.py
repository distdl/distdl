from torch.autograd.function import _ContextMethodMixin


class Bunch(dict):
    def __getattribute__(self, key):
        try:
            return self[key]
        except KeyError:
            try:
                return object.__getattribute__(self, key)
            except AttributeError:
                raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class DummyContext(Bunch, _ContextMethodMixin):

    pass
