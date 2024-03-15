import types
import dataclasses


class OvermindObjectRef(str):
    __slots__ = ()

    def __repr__(self):
        return f'OvermindObjectRef({super().__repr__()})'


def _deepfreeze(v):
    if isinstance(v, list):
        return tuple(_deepfreeze(i) for i in v)
    elif isinstance(v, dict):
        return tuple(
            (_deepfreeze(k), _deepfreeze(v))
            for k, v in v.items()
        )
    elif hasattr(v, '__dataclass_fields__'):
        return dataclasses.astuple(v)
    else:
        return v


def display_of(fn, kwargs):
        if isinstance(fn, types.MethodType):
            if isinstance(fn.__self__, type):
                ty = fn.__self__
            else:
                ty = type(fn.__self__)
            fndisp = f'{ty.__module__}.{ty.__name__}.{fn.__name__}'
        else:
            fndisp = f'{fn.__module__}.{fn.__name__}'

        args_disp = [f'{k}={repr(v)}' for k, v in kwargs.items()]

        disp = f'{fndisp}({", ".join(args_disp)})'
        return disp


def key_of(fn, kwargs):
    return (fn, _deepfreeze(kwargs))
