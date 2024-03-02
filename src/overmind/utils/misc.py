# -*- coding: utf-8 -*-

# -- stdlib --
from functools import wraps
from typing import Any
import types

# -- third party --
# -- own --

# -- code --
LOOKUP_TABLE: Dict[Any, Any] = {}


def hook_lookup(o: Any) -> Any:
    return LOOKUP_TABLE.get(o)


def hook(module, name=None):
    def inner(hooker):
        funcname = name or hooker.__name__
        hookee: Any = getattr(module, funcname)

        assert hookee not in LOOKUP_TABLE

        if isinstance(hookee, types.MethodType) and isinstance(hookee.__self__, type):
            hookee = hookee.__func__
            @classmethod
            @wraps(hookee)
            def real_hooker(cls, *args, **kwargs):
                return hooker(types.MethodType(hookee, cls), *args, **kwargs)

        else:
            @wraps(hookee)
            def real_hooker(*args, **kwargs):
                return hooker(hookee, *args, **kwargs)

        LOOKUP_TABLE[real_hooker] = hookee
        LOOKUP_TABLE[hookee] = real_hooker

        setattr(module, funcname, real_hooker)
        return real_hooker

    return inner
